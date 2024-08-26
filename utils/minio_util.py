import os
import io
import uuid

from minio import Minio
from minio.error import S3Error
from datetime import timedelta,datetime
from tqdm import tqdm
from minio.deleteobjects import DeleteObject
from concurrent.futures import as_completed, ThreadPoolExecutor
from urllib.parse import urlunsplit
from enum import Enum

class ViewContentTypeEnum(Enum):
    DEFAULT = ("default", "application/octet-stream")
    PNG = ("png", "image/png")
    JPEG = ("jpeg", "image/jpeg")
    JPG = ("jpg", "image/jpeg")
    GIF = ("gif", "image/gif")
    WBMP = ("wbmp", "image/vnd.wap.wbmp")
    TIFF = ("tiff", "image/tiff")
    JFIF = ("jfif", "image/jpeg")
    TIF = ("tif", "image/tiff")
    FAX = ("fax", "image/fax")
    JPE = ("jpe", "image/jpeg")
    NET = ("net", "image/pnetvue")
    RP = ("rp", "image/vnd.rn-realpix")
    ICO = ("ico", "image/x-icon")

    def __init__(self, name, mime_type):
        self._name = name
        self.mime_type = mime_type


def get_mime_type_from_path(file_path):
    # 获取文件扩展名
    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower().lstrip('.')  # 去除点号并转为小写
    # 根据扩展名查找枚举成员
    for member in ViewContentTypeEnum:
        if member._name == file_extension:
            return member.mime_type
    # 如果没有找到匹配项，则返回默认值
    return ViewContentTypeEnum.DEFAULT.mime_type


def add_date_prefix_to_filename(filename):
    # 获取当前日期
    current_date = datetime.now()

    # 提取年月日
    year = str(current_date.year)
    month = str(current_date.month).zfill(2)  # 保证月份为两位数
    day = str(current_date.day).zfill(2)  # 保证日期为两位数

    # 构造带有日期前缀的完整路径
    date_prefix_path = os.path.join(year, month, day, filename)
    return date_prefix_path

class MiniUtil(object):
    client = None
    policy = '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"AWS":["*"]},"Action":["s3:GetBucketLocation","s3:ListBucket"],"Resource":["arn:aws:s3:::%s"]},{"Effect":"Allow","Principal":{"AWS":["*"]},"Action":["s3:GetObject"],"Resource":["arn:aws:s3:::%s/*"]}]}'

    def __new__(cls, *args, **kwargs):
        if not cls.client:
            cls.client = object.__new__(cls)
        return cls.client

    def __init__(self, service, access_key, secret_key, resource_server=None, secure=False, section_size=10, t_max=3):
        '''
        实例化参数
        :param service: 服务器地址
        :param access_key: access_key
        :param secret_key: secret_key
        :param secure: secure
        :param section_size: 切片大小mb
        :param t_max: 线程池大小
        '''
        self.service = service
        self.client = Minio(service, access_key=access_key, secret_key=secret_key, secure=secure)
        self.size = section_size * 1024 * 1024
        self.processPool = ThreadPoolExecutor(max_workers=t_max)
        if resource_server is None:
            resource_server = ("https://" if secure else "http://") + service
        self.resource_server = resource_server

    def exists_bucket(self, bucket_name):
        """
        判断桶是否存在
        :param bucket_name: 桶名称
        :return:
        """
        return self.client.bucket_exists(bucket_name=bucket_name)

    def create_bucket(self, bucket_name: str, is_policy: bool = True):
        """
        创建桶 + 赋予策略
        :param bucket_name: 桶名
        :param is_policy: 策略
        :return:
        """
        if self.exists_bucket(bucket_name=bucket_name):
            return False
        else:
            self.client.make_bucket(bucket_name=bucket_name)
        if is_policy:
            policy = self.policy % (bucket_name, bucket_name)
            self.client.set_bucket_policy(bucket_name=bucket_name, policy=policy)
        return True

    def get_bucket_list(self):
        """
        列出存储桶
        :return:
        """
        buckets = self.client.list_buckets()
        bucket_list = []
        for bucket in buckets:
            bucket_list.append(
                {"bucket_name": bucket.name, "create_time": bucket.creation_date}
            )
        return bucket_list

    def remove_bucket(self, bucket_name):
        """
        删除桶
        :param bucket_name:
        :return:
        """
        try:
            self.client.remove_bucket(bucket_name=bucket_name)
        except S3Error as e:
            print("[error]:", e)
            return False
        return True

    def bucket_list_files(self, bucket_name, prefix):
        """
        列出存储桶中所有对象
        :param bucket_name: 同名
        :param prefix: 前缀
        :return:
        """
        try:
            files_list = self.client.list_objects(bucket_name=bucket_name, prefix=prefix, recursive=True)
            for obj in files_list:
                print(obj.bucket_name, obj.object_name.encode('utf-8'), obj.last_modified,
                      obj.etag, obj.size, obj.content_type)
        except S3Error as e:
            print("[error]:", e)

    def bucket_policy(self, bucket_name):
        """
        列出桶存储策略
        :param bucket_name:
        :return:
        """
        try:
            policy = self.client.get_bucket_policy(bucket_name)
        except S3Error as e:
            print("[error]:", e)
            return None
        return policy

    def download_file(self, bucket_name, file, file_path, stream=1024 * 32):
        """
        从bucket 下载文件 + 写入指定文件
        :return:
        """
        try:
            data = self.client.get_object(bucket_name, file)
            with open(file_path, "wb") as fp:
                for d in data.stream(stream):
                    fp.write(d)
        except S3Error as e:
            print("[error]:", e)

    def fget_file(self, bucket_name, file, file_path):
        """
        下载保存文件保存本地
        :param bucket_name:
        :param file:
        :param file_path:
        :return:
        """
        self.client.fget_object(bucket_name, file, file_path)

    def get_section_data(self, bucket_name, file_name, start, size):
        '''
        获取切片数据
        :param bucket_name:
        :param file_name:
        :param start:
        :param size:
        :return:
        '''
        data = {'start': start, 'data': None}
        try:
            obj = self.client.get_object(bucket_name=bucket_name, object_name=file_name, offset=start, length=size)
            data = {'start': start, 'data': obj}

        except Exception as e:
            print('=============', e)

        return data

    def get_file_object(self, bucket_name, object_name):
        """
        获取文件对象
        :param bucket_name:
        :param file:
        :return:
        """
        pool_arr = []
        file_data = io.BytesIO()
        try:
            stat_obj = self.client.stat_object(bucket_name=bucket_name, object_name=object_name)
            total_length = stat_obj.size
            size = self.size
            total_page = self.get_page_count(total_length, size)

            total = 0
            for chunck in range(1, total_page + 1):
                start = (chunck - 1) * size
                if chunck == total_page:
                    size = total_length - total
                thread_item = self.processPool.submit(self.get_section_data, bucket_name, object_name, start, size)
                pool_arr.append(thread_item)

            for key, thread_res in tqdm(enumerate(as_completed(pool_arr)), unit='MB', unit_scale=True,
                                        unit_divisor=1024 * 1024, ascii=True, total=len(pool_arr), ncols=50):
                try:
                    _res = thread_res.result()
                    file_data.seek(_res['start'])
                    file_data.write(_res['data'].read())
                except Exception as e:
                    print(e)

        except Exception as e:
            print(e)

        return file_data.getvalue()

    def get_object_list(self, bucket_name):
        objects = []
        try:
            objects = self.client.list_objects(bucket_name)
        except Exception as e:
            print(e)

        return objects

    def get_page_count(self, total, per_page):
        """
        计算分页总数
        :param total: 记录总数
        :param per_page: 每页记录数
        :return: 分页总数
        """
        page_count = total // per_page
        if total % per_page != 0:
            page_count += 1
        return page_count

    def copy_file(self, bucket_name, file, file_path):
        """
        拷贝文件（最大支持5GB）
        :param bucket_name:
        :param file:
        :param file_path:
        :return:
        """
        self.client.copy_object(bucket_name, file, file_path)

    def upload_file(self, bucket_name, file, file_path, content_type):
        """
        上传文件 + 写入
        :param bucket_name: 桶名
        :param file: 文件名
        :param file_path: 本地文件路径
        :param content_type: 文件类型
        :return:
        """
        try:
            # Make bucket if not exist.
            found = self.client.bucket_exists(bucket_name)
            if not found:
                print("Bucket '{}' is not exists".format(bucket_name))
                self.client.make_bucket(bucket_name)

            with open(file_path, "rb") as file_data:
                file_stat = os.stat(file_path)
                self.client.put_object(bucket_name, file, file_data, file_stat.st_size, content_type=content_type)

        except S3Error as e:
            print("[error]:", e)

    def upload_object(self, bucket_name, file, file_data, content_type='binary/octet-stream'):
        """
        上传文件 + 写入
        :param bucket_name: 桶名
        :param file: 文件名
        :param file_data: bytes
        :param content_type: 文件类型 默认是appliction/octet-stream
        :return:
        """
        try:
            # Make bucket if not exist.
            found = self.client.bucket_exists(bucket_name)
            if not found:
                print("Bucket '{}' is not exists".format(bucket_name))
                self.client.make_bucket(bucket_name)

            buffer = io.BytesIO(file_data)
            st_size = len(file_data)
            self.client.put_object(bucket_name, file, buffer, st_size, content_type=content_type)
        except S3Error as e:
            print("[error]:", e)

    def fput_file(self, bucket_name, file_path: str, file_name: str = None):
        """
        上传文件
        :param bucket_name: 桶名
        :param file_path: 本地文件路径
        :param file_name: 文件名
        :return:
        """

        if file_path.startswith("."):
            file_path = os.path.abspath(file_path)
        try:
            if file_name is None:
                file_name = str(uuid.uuid4()) + "." + file_path.split('.')[-1]
            file_name = add_date_prefix_to_filename(file_name)
            content_type = get_mime_type_from_path(file_path)
            # Make bucket if not exist.
            found = self.client.bucket_exists(bucket_name)
            if not found:
                self.client.make_bucket(bucket_name)
            else:
                print("Bucket '{}' already exists".format(bucket_name))

            self.client.fput_object(bucket_name, file_name, file_path, content_type)
            return os.path.join(self.resource_server, bucket_name, file_name)
        except S3Error as e:
            print("[error]:", e)

    def stat_object(self, bucket_name, file, log=True):
        """
        获取文件元数据
        :param bucket_name:
        :param file:
        :return:
        """
        res = None
        try:
            data = self.client.stat_object(bucket_name, file)
            res = data
            if log:
                print(data.bucket_name)
                print(data.object_name)
                print(data.last_modified)
                print(data.etag)
                print(data.size)
                print(data.metadata)
                print(data.content_type)
        except S3Error as e:
            if log:
                print("[error]:", e)

        return res

    def remove_file(self, bucket_name, file):
        """
        移除单个文件
        :return:
        """
        self.client.remove_object(bucket_name, file)

    def remove_files(self, bucket_name, file_list):
        """
        删除多个文件
        :return:
        """
        delete_object_list = [DeleteObject(file) for file in file_list]
        for del_err in self.client.remove_objects(bucket_name, delete_object_list):
            print("del_err", del_err)

    def presigned_get_file(self, bucket_name, file, days=7):
        """
        生成一个http GET操作 签证URL
        :return:
        """
        return self.client.presigned_get_object(bucket_name, file, expires=timedelta(days=days))
