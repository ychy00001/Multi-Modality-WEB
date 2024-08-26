from utils import minio_util

if __name__ == '__main__':
    result = minio_util.fput_file("inference", "./test.jpg")
    print(result)