from itertools import cycle

from i2mb.activities.base_activity import ActivityNone


class ActivityDescriptor:
    """Agents can perform activities while in a location. The ActivityDescriptor class is a base class which describes
    where in the current location an activity can be performed."""
    def __init__(self, location=None, device=None, pos_id=None):
        # Where the activity will take place
        self.location = location

        # Device in location used during activity, e.g., bed
        self.device = device

        # Activity availability
        self.available = True

        # Keep track of how much this particular activity was used.
        self.accumulated = 0

        # Keep track of how much this particular activity was used.
        self.activity_class = ActivityNone
        #
        if pos_id is None:
            pos_id = 0
        self.pos_id = pos_id

    def __repr__(self):
        return f"Activity  {type(self)}:(location:{self.location}, device:{self.device}, "\
                 f"accumulated:{self.accumulated})"


class CompoundActivityDescriptor:
    """Describes a sequence of activity descriptors. """
    def __init__(self, activity_descriptors):
        self.activity_descriptors = activity_descriptors
        self.__activity_cycle = cycle(self.activity_descriptors)

    def __next__(self):
        return next(self.__activity_cycle)