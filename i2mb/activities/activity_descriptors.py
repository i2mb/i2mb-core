from i2mb.activities import atomic_activities
from i2mb.activities.base_activity_descriptor import ActivityDescriptor


class Sleep(ActivityDescriptor):
    def __init__(self, location=None, device=None, pos_id=None):
        super().__init__(location, device, pos_id)
        self.activity_class = atomic_activities.Sleep


class Work(ActivityDescriptor):
    def __init__(self, location=None, device=None, pos_id=None):
        super().__init__(location, device, pos_id)
        self.activity_class = atomic_activities.Work


class Eat(ActivityDescriptor):
    def __init__(self, location=None, device=None, pos_id=None):
        super().__init__(location, device, pos_id)

        self.activity_class = atomic_activities.Eat


class Rest(ActivityDescriptor):
    def __init__(self, location=None, device=None, pos_id=None):
        super().__init__(location, device, pos_id)
        self.activity_class = atomic_activities.Rest


class Toilet(ActivityDescriptor):
    def __init__(self, location=None, device=None, pos_id=None):
        super().__init__(location, device, pos_id)
        self.activity_class = atomic_activities.Toilet


class Sink(ActivityDescriptor):
    def __init__(self, location=None, device=None, pos_id=None):
        super().__init__(location, device, pos_id)
        self.activity_class = atomic_activities.Sink


class Shower(ActivityDescriptor):
    def __init__(self, location=None, device=None, pos_id=None):
        super().__init__(location, device, pos_id)
        self.activity_class = atomic_activities.Shower


class Cook(ActivityDescriptor):
    def __init__(self, location=None, device=None, pos_id=None):
        super().__init__(location, device, pos_id)
        self.activity_class = atomic_activities.Cook
