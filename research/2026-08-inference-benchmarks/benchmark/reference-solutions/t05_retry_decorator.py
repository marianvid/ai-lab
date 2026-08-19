import functools
def retry(times, exceptions=(Exception,)):
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*a, **kw):
            last = None
            for _ in range(times):
                try:
                    return fn(*a, **kw)
                except exceptions as e:
                    last = e
            raise last
        return wrapper
    return deco
