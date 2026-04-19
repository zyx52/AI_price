import redis
import os
try:
    r = redis.from_url("redis://127.0.0.1:6379/0", socket_connect_timeout=1)
    if r.ping():
        print("REDIS_ALREADY_RUNNING")
    else:
        print("REDIS_NOT_RESPONDING")
except Exception:
    print("REDIS_NOT_FOUND")
