import time

def timer_decorator(func):
    """
    这个装饰器用于记录函数的执行时间。
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} 运行时间: {end_time - start_time:.4f}秒")
        return result
    return wrapper