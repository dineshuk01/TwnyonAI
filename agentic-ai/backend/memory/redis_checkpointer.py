import json
import os
from typing import Any, Optional, Sequence, AsyncIterator

import redis.asyncio as aioredis
from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
)

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# A separate Redis client for binary data (no decode_responses).
# The checkpointer stores serialized bytes, so we must NOT decode them as UTF-8.
_binary_redis_client = aioredis.from_url(REDIS_URL, decode_responses=False)

# Key-naming helpers
def _checkpoint_key(thread_id: str, checkpoint_ns: str, checkpoint_id: str) -> str:
    return f"cp:{thread_id}:{checkpoint_ns}:{checkpoint_id}"

def _blob_key(thread_id: str, checkpoint_ns: str, channel: str, version: str) -> str:
    return f"blob:{thread_id}:{checkpoint_ns}:{channel}:{version}"

def _writes_key(thread_id: str, checkpoint_ns: str, checkpoint_id: str, task_id: str, idx: int) -> str:
    return f"writes:{thread_id}:{checkpoint_ns}:{checkpoint_id}:{task_id}:{idx}"

def _index_key(thread_id: str, checkpoint_ns: str) -> str:
    """Sorted-set key that maps checkpoint_id → score (timestamp order) for listing."""
    return f"cpidx:{thread_id}:{checkpoint_ns}"


class AsyncRedisCheckpointer(BaseCheckpointSaver):
    """
    An asynchronous LangGraph checkpointer backed by Redis.

    Uses the built-in ``self.serde`` (``JsonPlusSerializer``) to serialise
    checkpoint data – the same approach that the official ``InMemorySaver``
    uses internally.  This avoids ``pickle`` entirely and therefore never
    hits the "Can't get local object" error.
    """

    TTL = 86400  # 24 h

    # ------------------------------------------------------------------
    # Sync stubs (this app always uses the async variants)
    # ------------------------------------------------------------------
    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        raise NotImplementedError("Use async methods")

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        raise NotImplementedError("Use async methods")

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        raise NotImplementedError("Use async methods")

    # ------------------------------------------------------------------
    # aget_tuple
    # ------------------------------------------------------------------
    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        thread_id: str = config["configurable"]["thread_id"]
        checkpoint_ns: str = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = get_checkpoint_id(config)

        try:
            if not checkpoint_id:
                # Fetch the latest checkpoint_id from the sorted-set index
                idx_key = _index_key(thread_id, checkpoint_ns)
                latest = await _binary_redis_client.zrevrange(idx_key, 0, 0)
                if not latest:
                    return None
                checkpoint_id = latest[0].decode() if isinstance(latest[0], bytes) else latest[0]

            key = _checkpoint_key(thread_id, checkpoint_ns, checkpoint_id)
            raw = await _binary_redis_client.get(key)
            if raw is None:
                return None

            stored = json.loads(raw)

            # Deserialize checkpoint & metadata
            checkpoint_data: Checkpoint = self.serde.loads_typed(
                (stored["cp_type"], stored["cp_data"].encode("latin-1"))
            )

            metadata_data: CheckpointMetadata = self.serde.loads_typed(
                (stored["md_type"], stored["md_data"].encode("latin-1"))
            )

            parent_checkpoint_id: Optional[str] = stored.get("parent_id")

            # Reconstruct channel_values from blobs
            channel_values: dict[str, Any] = {}
            channel_versions = checkpoint_data.get("channel_versions", {})
            for ch, ver in channel_versions.items():
                bk = _blob_key(thread_id, checkpoint_ns, ch, str(ver))
                blob_raw = await _binary_redis_client.get(bk)
                if blob_raw is not None:
                    blob_stored = json.loads(blob_raw)
                    if blob_stored["type"] != "empty":
                        channel_values[ch] = self.serde.loads_typed(
                            (blob_stored["type"], blob_stored["data"].encode("latin-1"))
                        )

            # Reconstruct pending writes
            pending_writes: list[tuple[str, str, Any]] = []
            writes_pattern = f"writes:{thread_id}:{checkpoint_ns}:{checkpoint_id}:*"
            write_keys = await _binary_redis_client.keys(writes_pattern)
            for wk in sorted(write_keys):
                w_raw = await _binary_redis_client.get(wk)
                if w_raw is not None:
                    w_stored = json.loads(w_raw)
                    val = self.serde.loads_typed(
                        (w_stored["v_type"], w_stored["v_data"].encode("latin-1"))
                    )
                    pending_writes.append((w_stored["task_id"], w_stored["channel"], val))

            return CheckpointTuple(
                config={
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": checkpoint_id,
                    }
                },
                checkpoint={**checkpoint_data, "channel_values": channel_values},
                metadata=metadata_data,
                pending_writes=pending_writes,
                parent_config=(
                    {
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": parent_checkpoint_id,
                        }
                    }
                    if parent_checkpoint_id
                    else None
                ),
            )

        except Exception as e:
            print(f"Error loading checkpoint from redis: {e}")
            return None

    # ------------------------------------------------------------------
    # aput
    # ------------------------------------------------------------------
    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        thread_id: str = config["configurable"]["thread_id"]
        checkpoint_ns: str = config["configurable"].get("checkpoint_ns", "")
        parent_checkpoint_id: Optional[str] = config["configurable"].get("checkpoint_id")

        c = checkpoint.copy()
        channel_values: dict[str, Any] = c.pop("channel_values", {})  # type: ignore[misc]

        try:
            # Persist per-channel blobs for new/changed channels
            for ch, ver in new_versions.items():
                bk = _blob_key(thread_id, checkpoint_ns, ch, str(ver))
                if ch in channel_values:
                    blob_type, blob_data = self.serde.dumps_typed(channel_values[ch])
                    blob_stored = json.dumps({"type": blob_type, "data": blob_data.decode("latin-1")})
                else:
                    blob_stored = json.dumps({"type": "empty", "data": ""})
                await _binary_redis_client.setex(bk, self.TTL, blob_stored.encode())

            # Persist checkpoint (without channel_values) and metadata
            cp_type, cp_data = self.serde.dumps_typed(c)
            from langgraph.checkpoint.base import get_checkpoint_metadata
            md_type, md_data = self.serde.dumps_typed(
                get_checkpoint_metadata(config, metadata)
            )

            stored = json.dumps({
                "cp_type": cp_type,
                "cp_data": cp_data.decode("latin-1"),
                "md_type": md_type,
                "md_data": md_data.decode("latin-1"),
                "parent_id": parent_checkpoint_id,
            })

            key = _checkpoint_key(thread_id, checkpoint_ns, checkpoint["id"])
            await _binary_redis_client.setex(key, self.TTL, stored.encode())

            # Add to the sorted-set index so aget_tuple can find the latest
            idx_key = _index_key(thread_id, checkpoint_ns)
            await _binary_redis_client.zadd(idx_key, {checkpoint["id"]: 1.0})
            await _binary_redis_client.expire(idx_key, self.TTL)

        except Exception as e:
            print(f"Error saving checkpoint to redis: {e}")

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }

    # ------------------------------------------------------------------
    # aput_writes
    # ------------------------------------------------------------------
    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        thread_id: str = config["configurable"]["thread_id"]
        checkpoint_ns: str = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id: str = config["configurable"]["checkpoint_id"]

        try:
            for idx, (channel, value) in enumerate(writes):
                wk = _writes_key(thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
                v_type, v_data = self.serde.dumps_typed(value)
                w_stored = json.dumps({
                    "task_id": task_id,
                    "channel": channel,
                    "v_type": v_type,
                    "v_data": v_data.decode("latin-1"),
                    "task_path": task_path,
                })
                await _binary_redis_client.setex(wk, self.TTL, w_stored.encode())
        except Exception as e:
            print(f"Error saving writes to redis: {e}")

    # ------------------------------------------------------------------
    # alist
    # ------------------------------------------------------------------
    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """List checkpoints for a thread, newest first."""
        if config is None:
            return

        thread_id: str = config["configurable"]["thread_id"]
        checkpoint_ns: str = config["configurable"].get("checkpoint_ns", "")
        idx_key = _index_key(thread_id, checkpoint_ns)
        ids = await _binary_redis_client.zrevrange(idx_key, 0, -1)

        count = 0
        for raw_id in ids:
            cp_id = raw_id.decode() if isinstance(raw_id, bytes) else raw_id
            if before:
                before_id = get_checkpoint_id(before)
                if before_id and cp_id >= before_id:
                    continue

            tuple_config: RunnableConfig = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": cp_id,
                }
            }
            item = await self.aget_tuple(tuple_config)
            if item is not None:
                yield item
                count += 1
                if limit and count >= limit:
                    break
