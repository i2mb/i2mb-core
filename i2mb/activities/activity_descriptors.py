from i2mb.activities import atomic_activities
from i2mb.activities.base_activity_descriptor import ActivityDescriptor


class Sleep(ActivityDescriptor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.activity_class = atomic_activities.Sleep


class CommuteBus(ActivityDescriptor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.activity_class = atomic_activities.CommuteBus


class Work(ActivityDescriptor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.activity_class = atomic_activities.Work


class CoffeeBreak(ActivityDescriptor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.activity_class = atomic_activities.CoffeeBreak


class Eat(ActivityDescriptor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.activity_class = atomic_activities.Eat


class EatAtBar(ActivityDescriptor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.activity_class = atomic_activities.EatAtBar


class EatAtRestaurant(ActivityDescriptor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.activity_class = atomic_activities.EatAtRestaurant


class Rest(ActivityDescriptor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.activity_class = atomic_activities.Rest


class Toilet(ActivityDescriptor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.activity_class = atomic_activities.Toilet


class Sink(ActivityDescriptor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.activity_class = atomic_activities.Sink


class Shower(ActivityDescriptor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.activity_class = atomic_activities.Shower


class Cook(ActivityDescriptor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.activity_class = atomic_activities.Cook
