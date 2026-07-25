import os
import json
import redis.asyncio as aioredis
from dotenv import load_dotenv

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# Create a global Redis connection pool
redis_client = aioredis.from_url(
    REDIS_URL,
    decode_responses=True
)

async def get_cache(key: str):
    """Retrieve and deserialize a JSON object from Redis."""
    try:
        data = await redis_client.get(key)
        if data:
            return json.loads(data)
    except Exception as e:
        print(f"Redis get_cache error: {e}")
    return None

async def set_cache(key: str, value: any, ttl_seconds: int = 3600):
    """Serialize and store an object in Redis with a TTL."""
    try:
        data = json.dumps(value)
        await redis_client.setex(key, ttl_seconds, data)
    except Exception as e:
        print(f"Redis set_cache error: {e}")

async def delete_cache(key: str):
    """Delete a specific key from Redis."""
    try:
        await redis_client.delete(key)
    except Exception as e:
        print(f"Redis delete_cache error: {e}")

async def invalidate_pattern(pattern: str):
    """Delete all keys matching a specific pattern."""
    try:
        keys = await redis_client.keys(pattern)
        if keys:
            await redis_client.delete(*keys)
    except Exception as e:
        print(f"Redis invalidate_pattern error: {e}")
