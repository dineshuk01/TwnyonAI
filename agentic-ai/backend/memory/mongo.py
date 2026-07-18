import os
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
from memory.redis_client import get_cache, set_cache, delete_cache, invalidate_pattern

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
client = AsyncIOMotorClient(MONGODB_URI)
db = client.agentdb

agent_memory_coll = db.agent_memory
session_history_coll = db.session_history
checkpoints_coll = db.checkpoints
users_coll = db.users

async def read_memory(key: str, user_email: str) -> str | None:
    cache_key = f"agent_memory:{user_email}:{key}"
    cached_val = await get_cache(cache_key)
    if cached_val is not None:
        return cached_val

    doc = await agent_memory_coll.find_one({"key": key, "user_email": user_email})
    if doc:
        val = doc.get("value")
        await set_cache(cache_key, val, 3600)
        return val
    return None

async def write_memory(key: str, value: str, user_email: str) -> None:
    await agent_memory_coll.update_one(
        {"key": key, "user_email": user_email},
        {"$set": {"value": value, "updatedAt": datetime.utcnow()}},
        upsert=True
    )
    cache_key = f"agent_memory:{user_email}:{key}"
    await set_cache(cache_key, value, 3600)
    await delete_cache(f"list_memory:{user_email}")

async def list_memory(user_email: str) -> list[dict]:
    cache_key = f"list_memory:{user_email}"
    cached_list = await get_cache(cache_key)
    if cached_list is not None:
        return cached_list

    cursor = agent_memory_coll.find({"user_email": user_email})
    memories = []
    async for doc in cursor:
        memories.append({
            "key": doc.get("key"),
            "value": doc.get("value"),
            "updatedAt": doc.get("updatedAt").isoformat() if doc.get("updatedAt") else None,
            "user_email": doc.get("user_email")
        })
    await set_cache(cache_key, memories, 3600)
    return memories

async def clear_memory(user_email: str) -> None:
    await agent_memory_coll.delete_many({"user_email": user_email})
    await delete_cache(f"list_memory:{user_email}")
    await invalidate_pattern(f"agent_memory:{user_email}:*")

async def save_checkpoint(thread_id: str, state: dict) -> None:
    await checkpoints_coll.update_one(
        {"thread_id": thread_id},
        {"$set": {"state": state, "updatedAt": datetime.utcnow()}},
        upsert=True
    )

async def load_checkpoint(thread_id: str) -> dict | None:
    doc = await checkpoints_coll.find_one({"thread_id": thread_id})
    if doc:
        return doc.get("state")
    return None

async def save_session(session_id: str, messages: list[dict]) -> None:
    await session_history_coll.update_one(
        {"session_id": session_id},
        {"$set": {"messages": messages, "createdAt": datetime.utcnow()}},
        upsert=True
    )

async def load_session(session_id: str) -> list[dict]:
    doc = await session_history_coll.find_one({"session_id": session_id})
    if doc:
        return doc.get("messages", [])
    return []

async def create_user(name: str, email: str, password_hash: str) -> None:
    await users_coll.update_one(
        {"email": email},
        {"$set": {"name": name, "password_hash": password_hash, "createdAt": datetime.utcnow()}},
        upsert=True
    )
    await delete_cache(f"user:{email}")

async def get_user_by_email(email: str) -> dict | None:
    cache_key = f"user:{email}"
    cached_user = await get_cache(cache_key)
    if cached_user:
        return cached_user
        
    doc = await users_coll.find_one({"email": email})
    if doc:
        doc["_id"] = str(doc["_id"])
        if "createdAt" in doc and isinstance(doc["createdAt"], datetime):
            doc["createdAt"] = doc["createdAt"].isoformat()
        await set_cache(cache_key, doc, 3600)
    return doc
