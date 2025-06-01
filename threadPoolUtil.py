from concurrent.futures import ThreadPoolExecutor


def get_transformer_thread_pool() -> ThreadPoolExecutor:
    thread_pool = ThreadPoolExecutor(
        max_workers=2,
        thread_name_prefix="transformerTask-",
    )
    return thread_pool
