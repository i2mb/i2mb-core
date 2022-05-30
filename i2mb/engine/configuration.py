from i2mb.utils import global_time


class Configuration(dict):
    def __init__(self, population_size=None, ticks_hour=None, config_file=None):
        super().__init__()

        # Setup global time
        if ticks_hour is None:
            ticks_hour = global_time.ticks_hour

        global_time.ticks_hour = ticks_hour
        self["ticks_hour"] = ticks_hour
        self["time_factor_ticks_day"] = global_time.ticks_day

        # Setup population size
        self["population_size"] = population_size

        # Load configuration
        if config_file is not None:
            self.config_file = config_file
            self.load_config(config_file)

    @property
    def ticks_hour(self):
        return global_time.ticks_hour

    @ticks_hour.setter
    def ticks_hour(self, value):
        # Setup global time
        global_time.ticks_hour = value
        self["ticks_hour"] = value
        self["time_factor_ticks_day"] = global_time.ticks_day

    @property
    def time_factor_ticks_day(self):
        return global_time.ticks_day

    @time_factor_ticks_day.setter
    def time_factor_ticks_day(self, value):
        global_time.ticks_day = value
        self["time_factor_ticks_day"] = global_time.ticks_day

    @property
    def population_size(self):
        return self["population_size"]

    @population_size.setter
    def population_size(self, value):
        self["population_size"] = value

    def load_config(self, config_file):
        self.config_file = config_file
        with open(config_file) as cfg_file:
            code = compile(cfg_file.read(), config_file, "exec")
            exec(code, {"config": self})
