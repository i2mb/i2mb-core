from masskrug.worlds import CompositeWorld
import numpy as np


class ApartmentWorld(CompositeWorld):
    def __init__(self, apartment, working, **kwargs):
        super().__init__(**kwargs)
        self.apartment = apartment
        self.working = working

        n = len(self.population)
        self.is_sitting = np.zeros((n, 1), dtype=bool)
        self.current_sitting_duration = np.full((n, 1), np.inf)
        self.accumulated_sitting = np.zeros((n, 1))
        self.next_sitting_time = np.full((n, 1), -np.inf)
        self.sitting_position = np.zeros((n, 2))
        self.busy = np.zeros((n, 1), dtype=bool)

        self.stay = np.zeros((n, 1), dtype=bool)
        self.current_stay_duration = np.full((n, 1), -np.inf)
        self.accumulated_stay = np.zeros((n, 1))

        self.population.add_property("is_sitting", self.is_sitting)
        self.population.add_property("current_sitting_duration", self.current_sitting_duration)
        self.population.add_property("accumulated_sitting", self.accumulated_sitting)
        self.population.add_property("next_sitting_time", self.next_sitting_time)
        self.population.add_property("sitting_position", self.sitting_position)
        self.population.add_property("busy", self.busy)
        self.population.add_property("stay", self.stay)
        self.population.add_property("current_stay_duration", self.current_stay_duration)
        self.population.add_property("accumulated_stay", self.accumulated_stay)

    def set_entries(self):
        # apartment entries
        entries = [a.corridor.get_room_entries()[0] for a in self.population.home]
        ids = [a.corridor.get_room_entries()[1] for a in self.population.home]
        self.ids_flatten = [idx for sub in ids for idx in sub]
        self.entries_flatten = [e for sub in entries for e in sub]

        self.kitchen_entries = [self.entries_flatten[self.ids_flatten.index(id(a.kitchen))] for a in
                                self.population.home]
        self.kitchen_entries = np.array(self.kitchen_entries)

        self.dining_entries = [self.entries_flatten[self.ids_flatten.index(id(a.diningroom))] for a in
                               self.population.home]
        self.dining_entries = np.array(self.dining_entries)

        self.living_entries = [self.entries_flatten[self.ids_flatten.index(id(a.livingroom))] for a in
                               self.population.home]
        self.living_entries = np.array(self.living_entries)

        self.bath_entries = [self.entries_flatten[self.ids_flatten.index(id(a.bath))] for a in self.population.home]
        self.bath_entries = np.array(self.bath_entries)

        self.bed_entries = [self.entries_flatten[self.ids_flatten.index(id(x))] for x in self.population.bedroom]
        self.bed_entries = np.array(self.bed_entries)

    def move_to_corridor(self, ids, target=None):
        n = len(self.population)

        in_corridor = self.population.location == self.apartment.corridor
        ids = ids & ~in_corridor
        if ids.any():
            # at entry point of current room
            at_entry_point = np.zeros((n, 2), dtype=bool)
            at_entry_point[ids] = self.population.position[ids] == [i.entry_point for i in
                                                                    self.population.location[
                                                                        ids]]
            at_entry_point = np.array([all(i) for i in at_entry_point])

            # move to room entry point
            new_target = ~at_entry_point & ids
            if new_target.any():
                self.population.target[new_target] = np.array(
                    [i.entry_point for i in self.population.location[new_target]])

            # move particles from entry point to corridor
            switch = at_entry_point & ids
            if switch.any():
                moved = np.zeros((n,), dtype=bool)
                for c in self.population.home[switch]:
                    bool_idx = (self.population.home == c).ravel() & switch & ~moved
                    if bool_idx.any():
                        self.move_particles(bool_idx, c.corridor)
                        moved[bool_idx] = True
                    if (~moved).sum == 0:
                        break
                if target is not None:
                    self.population.target[switch] = target[switch]
                else:
                    # self.population.busy[switch] = False
                    self.population.target[switch] = np.nan

    def move_from_corridor(self, in_corridor):
        n = len(self.population)
        if in_corridor.any():
            at_target = self.population.position == self.population.target
            at_target = np.array([all(i) for i in at_target])

            # move to living_room
            living = self.population.target == self.living_entries
            living = np.array([all(i) for i in living])
            switch = at_target & living
            if switch.any():
                self.move_particles(switch, self.apartment.livingroom)
                self.population.target[switch] = np.nan

            # move to dining_room
            dining = self.population.target == self.dining_entries
            dining = np.array([all(i) for i in dining])
            switch = at_target & dining
            if switch.any():
                self.move_particles(switch, self.apartment.diningroom)
                self.apartment.diningroom.sit_particles(self.population.index[switch])
                self.population.is_eating[switch] = True
                self.population.is_preparing[switch] = False

            # move to kitchen
            kitchen = self.population.target == self.kitchen_entries
            kitchen = np.array([all(i) for i in kitchen])
            switch = at_target & kitchen
            if switch.any():
                self.move_particles(switch, self.apartment.kitchen)
                self.population.is_preparing[switch] = True

            # move to bedroom
            bedroom = self.population.target == self.bed_entries
            bedroom = np.array([all(i) for i in bedroom])
            switch = at_target & bedroom
            if switch.any():
                for i in np.argwhere(switch).ravel():
                    temp_bed = np.zeros(n, dtype=bool)
                    temp_bed[i] = True
                    self.move_particles(temp_bed, self.population.bedroom[i])
                    self.population.target[temp_bed] = np.nan

            # move to bath
            bath = self.population.target == self.bath_entries
            bath = np.array([all(i) for i in bath])
            switch = at_target & bath

            if switch.any() & ~self.apartment.bath.occupied:
                can_go = np.zeros(n, dtype=bool)
                can_go[np.where(switch)[0][0]] = True

                self.move_particles(can_go, self.apartment.bath)
                self.population.in_bathroom[can_go] = True
                self.population.target[can_go] = np.nan

    def step(self, t):

        if not self.population:
            return

        n = len(self.population)
        at_home = self.population.at_home.ravel()
        in_corridor = np.array(self.population.location == self.apartment.corridor)
        self.move_from_corridor(in_corridor)

        if hasattr(self.population, "working"):
            # Make people come home
            come_home = ~self.population.working.ravel() & ~at_home
            if come_home.any():
                self.move_particles(come_home, self.apartment.corridor)
                self.population.is_outside[come_home] = False
                self.population.busy[come_home] = False
                self.population.target[come_home] = np.nan
                self.population.motion_mask[come_home] = True
                self.population.at_home[come_home] = True

            # Send people to work
            send_to_work = (self.population.working & ~self.population.is_outside).ravel()
            if send_to_work.any():
                go_out = send_to_work & at_home & ~in_corridor
                self.move_to_corridor(go_out, np.full((n, 2), self.apartment.corridor.entry_point))
                new_target = send_to_work & in_corridor
                if new_target.any():
                    self.population.target[new_target] = self.apartment.corridor.entry_point

                at_entry_point = np.zeros((n, 2), dtype=bool)
                at_entry_point[new_target] = self.population.position[new_target] == self.apartment.corridor.entry_point
                at_entry_point = np.array([all(i) for i in at_entry_point])

                leave_apartment = at_entry_point

                if leave_apartment.any():
                    for i in np.argwhere(leave_apartment).ravel():
                        temp_leave = np.zeros(n, dtype=bool)
                        temp_leave[i] = True
                        self.move_particles(temp_leave, self.population.office[i])

                    self.population.is_outside[leave_apartment] = True
                    self.population.target[leave_apartment] = np.nan
                    self.population.at_home[leave_apartment] = False

        if hasattr(self.population, "bath"):
            in_bathroom = self.population.location == self.apartment.bath

            # Make people dirty
            get_dirty = ~self.population.bath.ravel() & self.population.in_bathroom.ravel() & at_home

            if get_dirty.any():
                self.move_to_corridor(get_dirty & in_bathroom)

                ready = get_dirty & in_corridor
                if ready.any():
                    self.population.busy[ready] = False
                    self.population.in_bathroom[ready] = False

            # Send people cleaning
            send_to_bath = (self.population.bath & ~self.population.in_bathroom).ravel()
            send_to_bath = send_to_bath & at_home
            if send_to_bath.any():
                in_bath = send_to_bath & in_bathroom
                if in_bath.any():
                    self.population.in_bathroom[in_bath] = True
                    self.population.target[in_bath] = np.nan
                self.move_to_corridor(send_to_bath & ~in_corridor & ~in_bathroom, self.bath_entries)
            in_cor = send_to_bath & in_corridor
            if in_cor.any():
                self.population.target[in_cor] = self.bath_entries[in_cor]

        if hasattr(self.population, "eat"):
            in_kitchen = self.population.location == self.apartment.kitchen
            in_dining = self.population.location == self.apartment.diningroom

            # let people start preparing
            start_prepare = (
                    self.population.eat & ~self.population.is_preparing & ~self.population.is_eating).ravel()
            start_prepare = start_prepare & at_home

            if start_prepare.any():
                sp = start_prepare & ~in_kitchen & ~in_corridor
                self.move_to_corridor(sp, self.kitchen_entries)
            in_cor = start_prepare & in_corridor
            if in_cor.any():
                self.population.target[in_cor] = self.kitchen_entries[in_cor]

            # let people start eating
            start_eat = (
                    self.population.eat & self.population.is_preparing & ~self.population.is_eating & self.population.ready_prepare).ravel()
            if start_eat.any():
                go = start_eat & ~in_corridor & ~in_dining
                self.move_to_corridor(go, self.dining_entries)

            # stop people eating
            stop_eat = (
                    ~self.population.eat & self.population.is_eating & ~self.population.is_preparing).ravel()
            # if stop_eat.any():
            self.move_to_corridor(stop_eat & ~in_corridor)
            if (stop_eat & in_corridor).any():
                self.population.is_eating[stop_eat & in_corridor] = False
                self.population.busy[stop_eat & in_corridor] = False

        if hasattr(self.population, "sleep"):
            # Wake people up
            in_bedroom = [self.population.location == self.population.bedroom]
            in_bedroom = np.array([d for sub in in_bedroom for d in sub])
            # Send people to sleep
            send_to_bed = (self.population.sleep & ~self.population.in_bed).ravel()
            send_to_bed = send_to_bed & at_home & ~in_corridor & ~in_bedroom

            if send_to_bed.any():
                self.move_to_corridor(send_to_bed, self.bed_entries)
            in_cor = send_to_bed & in_corridor
            if in_cor.any():
                self.population.target[in_cor] = self.bed_entries[in_cor]

        '''
        if hasattr(self.population, "in_quarantine"):
            if self.population.in_quarantine.any():
                targets = self.bed_entries
                at_home = self.population.at_home.ravel() & self.population.in_quarantine.ravel()
                if ~at_home.any():
                    self.move_particles(~at_home, self.apartment.corridor)
                    self.population.at_home[~at_home] = True
                    self.population.target[~at_home] = targets[~at_home]

                in_bedroom = self.population.location == self.population.bedroom
                in_corridor = self.population.location == self.apartment.corridor

                if at_home.any():
                    at_entry_point = np.zeros((n, 2), dtype=bool)
                    switch = at_home
                    at_entry_point[switch] = self.population.position[switch] == [i.entry_point for i in
                                                                                  self.population.location[switch]]
                    at_entry_point = np.array([all(i) for i in at_entry_point])
                    at_entry = at_entry_point & ~in_bedroom & ~in_corridor
                    if at_entry.any():
                        self.move_particles(at_entry, self.apartment.corridor)
                        self.population.target[at_entry] = targets[at_entry]

                    not_at_point = ~at_entry_point & ~in_corridor & ~in_bedroom & at_home
                    if not_at_point.any():
                        self.population.target[not_at_point] = np.array(
                            [i.entry_point for i in self.population.location[not_at_point]])

                    at_bedroom_entry = np.zeros((n, 2), dtype=bool)
                    at_bedroom_entry[at_home & in_corridor] = self.population.position[at_home & in_corridor] == \
                                                              targets[at_home & in_corridor]
                    at_bedroom_entry = np.array([all(i) for i in at_bedroom_entry])

                    if at_bedroom_entry.any():
                        for i in np.argwhere(at_bedroom_entry).ravel():
                            temp_bed = np.zeros(n, dtype=bool)
                            temp_bed[i] = True
                            self.move_particles(temp_bed, self.population.bedroom[i])

                in_bedroom_quarantine = at_home & in_bedroom
                if in_bedroom_quarantine.any:
                    if hasattr(self.population, "sleep"):
                        not_sleeping = ~self.population.sleep.ravel() & in_bedroom_quarantine
                        if not_sleeping.any():
                            self.population.target[not_sleeping] = np.nan
                            self.population.motion_mask[not_sleeping] = True
                    else:
                        self.population.target[in_bedroom_quarantine] = np.nan
        '''
        return
