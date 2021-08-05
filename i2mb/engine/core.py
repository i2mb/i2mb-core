from functools import wraps
from i2mb.utils import cache_manager


class Engine:
    def __init__(self, models, populations=None, num_steps=None, select=None, debug=False):
        self.debug = debug
        if debug:
            import time
            if hasattr(time, "process_time_ns"):
                self.current_time = time.process_time_ns
            else:
                self.current_time = time.process_time

            self.debug_timer = {}

        self.time = None
        self.models = models
        self.num_steps = num_steps
        if select is None:
            select = slice(None, None, None)

        self.model_selector = select
        self.populations = []
        if populations is not None:
            self.populations = populations

    def step(self):
        self.time = 0
        cache_manager.time = 0
        while True:
            for p in self.populations:
                p.set_current_time(self.time)

            res = []
            for m in self.models:
                if self.debug:
                    t0 = self.current_time()

                res.append(m.step(self.time))
                if self.debug:
                    tf = self.current_time()
                    self.debug_timer.setdefault(f"{m}", []).append((tf - t0) * 1e-9)

            yield [r for r in res[self.model_selector]]

            self.time += 1
            cache_manager.time = self.time

            if self.num_steps is not None and self.time == self.num_steps - 1:
                break

    def finalize(self):
        for m in self.models:
            m.final(self.time)
