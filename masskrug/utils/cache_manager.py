class CacheManager:
    def __init__(self):
        self.__time = None
        self.__cache = {}

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            cache = self.__cache.setdefault(self.time, {})
            if func.__name__ not in cache:
                cache[func.__name__] = func(*args, **kwargs)

            return cache[func.__name__]

        return wrapper

    @property
    def time(self):
        return self.__time

    @time.setter
    def time(self, v):
        self.invalidate()
        self.__time = v

    def invalidate(self):
        # TODO: partial invalidation should be possible by categorizing functions into subjects.
        if self.__time in self.__cache:
            self.__cache.pop(self.__time)
