import os
import shutil


def create_or_clear_directory(dir_path):
    """创建或清空指定目录"""
    # 如果目录存在，则清空它
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        # 遍历目录中的所有内容
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            try:
                # 根据内容类型删除文件或目录
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    else:
        # 如果目录不存在，则创建它
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")