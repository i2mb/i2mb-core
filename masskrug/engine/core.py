from functools import wraps
from masskrug.utils import cache_manager


class Engine:
    def __init__(self, models, num_steps=None, select=None):
        self.time = None
        self.models = models
        self.num_steps = num_steps
        if select is None:
            select = slice(None, None, None)

        self.model_selector = select

    def step(self):
        self.time = 0
        cache_manager.time = 0
        while True:
            res = [m.step(self.time) for m in self.models]
            yield [r for r in res[self.model_selector]]
            self.time += 1
            cache_manager.time = self.time

            if self.num_steps is not None and self.time == self.num_steps - 1:
                break

    def finalize(self):
        for m in self.models:
            m.final(self.time)
