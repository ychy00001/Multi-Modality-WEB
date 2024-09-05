import os
import shutil


def clear_directory(directory):
    """
    清空指定目录下的所有文件和子目录。
    :param directory: 要清空的目录路径
    """
    # 检查目录是否存在
    if not os.path.exists(directory):
        print(f"目录 '{directory}' 不存在。")
        return

    # 遍历目录中的所有文件和子目录
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            # 如果是文件则删除
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            # 如果是目录则递归删除
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"删除 {file_path} 时失败: {e}")


if __name__ == '__main__':
    # 使用示例
    directory_to_clear = '/path/to/your/directory'
    clear_directory(directory_to_clear)
    print(f"目录 '{directory_to_clear}' 已经清空。")