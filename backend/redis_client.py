import os
import redis
import pickle
import functools
import hashlib
from config import get_config

class RedisClient:
    """
    Singleton custom Redis client wrapper.
    Handles connection, basic get/set/delete operations,
    and automatic serialization/deserialization of Python objects via pickle.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RedisClient, cls).__new__(cls)
            cls._instance._init_redis()
        return cls._instance
        
    def _init_redis(self):
        config = get_config()
        redis_url = config.CACHE_REDIS_URL
        try:
            self.client = redis.from_url(redis_url)
            # Simple ping to verify connection, if it fails it raises an exception
            self.client.ping()
        except Exception as e:
            print(f"Warning: Failed to connect to Redis at {redis_url}. Falling back to no-cache. Error: {e}")
            self.client = None
            
    def get(self, key):
        if not self.client: return None
        try:
            data = self.client.get(key)
            if data:
                return pickle.loads(data)
        except Exception as e:
            print(f"Redis get error for key {key}: {e}")
        return None
        
    def set(self, key, value, timeout=3600):
        if not self.client: return False
        try:
            data = pickle.dumps(value)
            if timeout:
                self.client.setex(key, timeout, data)
            else:
                self.client.set(key, data)
            return True
        except Exception as e:
            print(f"Redis set error for key {key}: {e}")
            return False
            
    def delete(self, key):
        if not self.client: return False
        try:
            self.client.delete(key)
            return True
        except Exception as e:
            print(f"Redis delete error for key {key}: {e}")
            return False

    def clear_all(self):
        if not self.client: return False
        try:
            self.client.flushdb()
            return True
        except Exception as e:
            print(f"Redis flushdb error: {e}")
            return False


def redis_cache(timeout=3600, key_prefix="cache:"):
    """
    Decorator to cache function results in Redis.
    Generates a unique key based on function name, module, and arguments.
    """
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            rc = RedisClient()
            if not rc.client:
                # If Redis is unavailable, just call the function
                return f(*args, **kwargs)
            
            # Create a string representation of args/kwargs for hashing
            # Note: Complex objects in args (like self) might not stringify safely or uniquely for caching,
            # so be cautious when using this decorator on class methods without customizing the key.
            # Usually it's best applied to functions with simple argument types.
            key_parts = [f.__module__, f.__name__]
            
            # For class methods, args[0] is often `self`. We can skip it or use its class name.
            safe_args = []
            for arg in args:
                if hasattr(arg, '__class__') and not isinstance(arg, (int, float, str, bool, tuple, frozenset)):
                    safe_args.append(arg.__class__.__name__)
                else:
                    safe_args.append(str(arg))
            
            if safe_args:
                key_parts.append(str(safe_args))
            if kwargs:
                key_parts.append(str(sorted(kwargs.items())))
            
            raw_key = ":".join(key_parts)
            hashed_key = hashlib.md5(raw_key.encode('utf-8')).hexdigest()
            full_key = f"{key_prefix}{hashed_key}"
            
            cached_value = rc.get(full_key)
            if cached_value is not None:
                return cached_value
                
            result = f(*args, **kwargs)
            rc.set(full_key, result, timeout=timeout)
            return result
        return wrapper
    return decorator
