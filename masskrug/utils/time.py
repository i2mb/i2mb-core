class SimulationTime:
    def __init__(self, tick_hour=8):
        self.ticks_hour = tick_hour
        self.ticks_week = 7 * self.time_scalar
        self.ticks_month = 30 * self.time_scalar

    @property
    def time_scalar(self):
        return 24 * self.ticks_hour

    def month(self, v):
        return v // self.ticks_month

    def day(self, v):
        """Day of the month"""
        return (v % self.ticks_month) // self.time_scalar

    def days(self, v):
        """Number of days elapsed since the beginning of the simulation."""
        return v // self.time_scalar

    def hour(self, v):
        return (v % self.time_scalar) // self.ticks_hour

    def minute(self, v):
        return ((v % self.time_scalar) % self.ticks_hour) * 60 // self.ticks_hour

    def week_start(self, v):
        return v // self.ticks_week * self.time_scalar

    def month_start(self, v):
        return v // self.ticks_month * self.time_scalar

    def to_current(self, delta, t):
        return t // self.time_scalar + delta

    def make_time(self, day=0, month=0, hour=0, minutes=0, week_day=0, delta=0):
        """Converts from date format to the number of ticks since the beginning of the simulation."""
        if week_day > 0 and (day > 0 or month > 0):
            raise RuntimeError("Both week_day and month or date provided, select one representation")

        day = self.time_scalar * day
        month = month * self.ticks_month
        hour = hour * self.ticks_hour
        minutes = minutes * self.ticks_hour // 60
        week_day = week_day * self.ticks_week

        return month + day + hour + minutes + week_day + delta
