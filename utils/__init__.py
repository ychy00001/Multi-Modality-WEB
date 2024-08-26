from utils.minio_util import MiniUtil
from config import MINIO_HOST, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_RESOURCE_HOST

minio_util = MiniUtil(MINIO_HOST, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, resource_server=MINIO_RESOURCE_HOST)