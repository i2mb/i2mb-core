class MutableCycle:
    def __init__(self, iterable):
        self.__iterable = iterable
        self.__iter = iter(iterable)

    def __go_round(self):
        while True:
            try:
                yield next(self.__iter)

            except StopIteration:
                self.__iter = iter(self.__iterable)

    def __next__(self):
        return next(self.__go_round())

    def __iter__(self):
        return self.__go_round()
