from typing import Dict

from services.redis.redis_cache import RedisEmbeddingCache


class CacheService:
    """Service for handling Redis cache operations"""

    def __init__(self):
        self.cache = RedisEmbeddingCache()

    def get_cache_info(self) -> Dict:
        """Get cache information and statistics"""
        try:
            return self.cache.get_cache_info()
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def clear_cache(self) -> Dict:
        """Clear all embeddings from cache"""
        try:
            cleared = self.cache.clear_embeddings()
            return {
                "message": f"Cleared {cleared} embeddings from cache",
                "cleared": cleared,
            }
        except Exception as e:
            return {"error": str(e)}

    def health_check(self) -> Dict:
        """Perform cache health check"""
        try:
            cache_info = self.get_cache_info()
            return {"status": "healthy", "redis": cache_info}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}


cache_service = CacheService()
