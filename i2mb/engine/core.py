from i2mb.utils import cache_manager, global_time


class Engine:
    def __init__(self, models, populations=None, base_file_name="./", num_steps=None, select=None, debug=False):
        self.base_file_name = base_file_name
        self.debug = debug
        if debug:
            import time
            if hasattr(time, "process_time_ns"):
                self.current_time = time.process_time_ns
            else:
                self.current_time = time.process_time

            self.debug_timer = {}

        self.time = 0
        self.models = models
        self.num_steps = num_steps
        if select is None:
            select = slice(None, None, None)

        self.model_selector = select
        self.populations = []
        if populations is not None:
            self.populations = populations

    def step(self):
        cache_manager.time = 0
        while True:
            for p in self.populations:
                p.set_current_time(self.time)

            for m in self.models:
                if self.debug:
                    t0 = self.current_time()

                m.step(self.time)
                m.save_to_file(self.time)
                if self.debug:
                    tf = self.current_time()
                    self.debug_timer.setdefault(f"{m}", []).append((tf - t0) * 1e-9)

            yield None

            self.time += 1
            global_time.set_sim_time(self.time)
            cache_manager.time = self.time

            if self.num_steps is not None and self.time == self.num_steps - 1:
                break

    def finalize(self):
        for m in self.models:
            m.final(self.time)

    def post_init_modules(self):
        for m in self.models:
            m.post_init(base_file_name=self.base_file_name)
