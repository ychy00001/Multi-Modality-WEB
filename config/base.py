import os
from config.conf import Config

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config = Config(os.path.join(BASE_DIR, ".env"))
MINIO_HOST: str = config('MINIO_HOST', cast=str, default='http://minio:9000')
MINIO_RESOURCE_HOST: str=config('MINIO_RESOURCE_HOST', cast=str, default='http://minio:9000')
MINIO_ACCESS_KEY: str = config('MINIO_ACCESS_KEY', cast=str, default='empty')
MINIO_SECRET_KEY: str = config('MINIO_SECRET_KEY', cast=str, default='empty')

VLLM_ENDPOINT: str = config('VLLM_ENDPOINT', cast=str, default='http://172.17.0.1:41176')