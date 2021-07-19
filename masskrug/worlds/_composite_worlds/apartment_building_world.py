import random

from masskrug.worlds import CompositeWorld
import numpy as np


class ApartmentBuildingWorld(CompositeWorld):
    def __init__(self, buildings, working, **kwargs):
        super().__init__(**kwargs)
        self.building = buildings
        self.working = working

        n = len(self.population)

        self.is_sitting = np.zeros((n, 1), dtype=bool)
        self.current_sitting_duration = np.full((n, 1), np.inf)
        self.accumulated_sitting = np.zeros((n, 1))
        self.next_sitting_time = np.full((n, 1), -np.inf)
        self.sitting_position = np.zeros((n, 2))
        self.busy = np.zeros((n, 1), dtype=bool)

        self.stay = np.zeros((n, 1), dtype=bool)
        self.current_stay_duration = np.full((n, 1), np.inf)
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
        # public spaces and entries
        self.stairs = np.array([b.stairs for b in self.population.building])
        self.floor_numbers = np.array([a.floor_number for a in self.population.home], dtype=int)
        self.public_corridors = []
        for b in self.building:
            self.public_corridors += [np.array(b.corridor)[self.floor_numbers]]
        self.public_corridors = np.array(self.public_corridors).ravel()
        self.corridor_entries = []
        for c, b in zip(self.public_corridors, self.population.building):
            self.corridor_entries += [(b.stairs).enter_world(0, idx=[], locations=np.array([id(c)])).ravel()]
        self.corridor_entries = np.array(self.corridor_entries)
        self.apartment_entries = []
        for c, a in zip(self.public_corridors, self.population.home):
            self.apartment_entries += [c.enter_world(0, idx=[], locations=np.array([id(a.corridor)])).ravel()]
        self.apartment_entries = np.array(self.apartment_entries)

        # apartment entries
        entries = [a.corridor.get_room_entries()[0] for a in self.population.home]
        ids = [a.corridor.get_room_entries()[1] for a in self.population.home]
        self.ids_flatten = [idx for sub in ids for idx in sub]
        self.entries_flatten = [e for sub in entries for e in sub]

        self.kitchen_entries = [self.entries_flatten[self.ids_flatten.index(id(a.kitchen))] for a in
                                self.population.home]
        self.kitchen_entries = np.array(self.kitchen_entries)

        self.dining_entries = [self.entries_flatten[self.ids_flatten.index(id(a.dining_room))] for a in
                               self.population.home]
        self.dining_entries = np.array(self.dining_entries)

        self.living_entries = [self.entries_flatten[self.ids_flatten.index(id(a.living_room))] for a in
                               self.population.home]
        self.living_entries = np.array(self.living_entries)

        self.bath_entries = [self.entries_flatten[self.ids_flatten.index(id(a.bath))] for a in self.population.home]
        self.bath_entries = np.array(self.bath_entries)

        self.bed_entries = [self.entries_flatten[self.ids_flatten.index(id(x))] for x in self.population.bedroom]
        self.bed_entries = np.array(self.bed_entries)

        # rooms in apartment
        self.corridor = np.array([a.corridor for a in self.population.home])
        self.livingroom = np.array([a.living_room for a in self.population.home])
        self.bath = np.array([a.bath for a in self.population.home])
        self.diningroom = np.array([a.dining_room for a in self.population.home])
        self.kitchen = np.array([a.kitchen for a in self.population.home])

    '''
    def update_bedroom_entries(self):
        entries = [a.corridor.get_room_entries()[0] for a in self.population.home]
        ids = [a.corridor.get_room_entries()[1] for a in self.population.home]
        self.ids_flatten = [idx for sub in ids for idx in sub]
        self.entries_flatten = [e for sub in entries for e in sub]
        self.bed_entries = [self.entries_flatten[self.ids_flatten.index(id(x))] for x in self.population.bedroom]
        self.bed_entries = np.array(self.bed_entries)
    '''

    def move_to_corridor(self, ids, target=None):
        n = len(self.population)

        in_corridor = [self.population.location == self.corridor]
        in_corridor = np.array([c for sub in in_corridor for c in sub])
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
                for c in set(self.population.home[switch]):
                    bool_idx = (self.population.home == c).ravel() & switch
                    if bool_idx.any():
                        self.move_agents(bool_idx, c.corridor)
                if target is not None:
                    self.population.target[switch] = target[switch]
                else:
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
                for a in set(self.population.home[switch]):
                    bool_idx = (self.population.home == a).ravel() & switch
                    if bool_idx.any():
                        self.population.target[bool_idx] = np.nan
                        self.move_agents(bool_idx, a.living_room)

            # move to dining_room
            dining = self.population.target == self.dining_entries
            dining = np.array([all(i) for i in dining])
            switch = at_target & dining
            if switch.any():
                for a in set(self.population.home[switch]):
                    bool_idx = (self.population.home == a).ravel() & switch
                    if bool_idx.any():
                        self.move_agents(bool_idx, a.dining_room)
                        a.dining_room.sit_particles(self.population.index[bool_idx])
                self.population.is_eating[switch] = True
                self.population.is_preparing[switch] = False

            # move to kitchen
            kitchen = self.population.target == self.kitchen_entries
            kitchen = np.array([all(i) for i in kitchen])
            switch = at_target & kitchen
            if switch.any():
                for a in set(self.population.home[switch]):
                    bool_idx = (self.population.home == a).ravel() & switch
                    if bool_idx.any():
                        self.move_agents(bool_idx, a.kitchen)
                self.population.target[switch] = np.nan
                self.population.is_preparing[switch] = True

            # move to bedroom
            bedroom = self.population.target == self.bed_entries
            bedroom = np.array([all(i) for i in bedroom])
            switch = at_target & bedroom
            if switch.any():
                for i in np.argwhere(switch).ravel():
                    temp_bed = np.zeros(n, dtype=bool)
                    temp_bed[i] = True
                    self.move_agents(temp_bed, self.population.bedroom[i])

            # move to bath
            bath = self.population.target == self.bath_entries
            bath = np.array([all(i) for i in bath])
            switch = at_target & bath
            occupied = np.array([b.occupied for b in self.bath])
            move_bath = switch & ~occupied
            if move_bath.any():
                for a in set(self.population.home[move_bath]):
                    bool_idx = (self.population.home == a) & move_bath
                    if bool_idx.any():
                        can_go = np.zeros(n, dtype=bool)
                        can_go[np.where(bool_idx)[0][0]] = True

                        self.move_agents(can_go, a.bath)
                        self.population.in_bathroom[can_go] = True
                        a.bath.occupied = True
                        self.population.target[can_go] = np.nan

    def walk_to_apartment(self, idx):
        n = len(self.population)
        in_office = ~self.population.in_building.ravel() & idx
        if in_office.any():
            for b in self.building:
                in_building = (self.population.building == b) & in_office
                if in_building.any():
                    self.move_agents(in_building, b.stairs)
            self.population.is_outside[in_office] = False
            self.population.in_building[in_office] = True

            # 0 stairs, 1 lift
            chosen = np.array(random.choices([0, 1], weights=[44, 56], k=n), dtype=bool)
            always_lift = (self.floor_numbers > 6)
            always_stairs = (self.floor_numbers == 0)
            use_stairs = in_office & ((~chosen & ~always_lift) | always_stairs)
            use_lift = in_office & ((chosen & ~always_stairs) | always_lift)

            if use_lift.any():
                for b in self.building:
                    bool_idx = ((self.population.building == b) & use_lift)
                    if bool_idx.any():
                        self.population.target[bool_idx] = b.stairs.lift_points[0]

            if use_stairs.any():
                self.population.target[use_stairs] = self.corridor_entries[use_stairs]

        for b in self.building:
            in_lift = (self.population.location == b.lift) & (b == self.population.building)
            if in_lift.any():
                self.leave_lift(in_lift)

        wait_lift = np.zeros((n, 2), dtype=bool)
        wait_lift[idx] = self.population.target[idx] == [b.stairs.lift_points[0] for b in self.population.building[idx]]
        wait_lift = np.array([all(i) for i in wait_lift])
        if wait_lift.any():
            self.move_to_lift(wait_lift, self.floor_numbers)
        at_corridor_entry = self.population.position == [c for c in self.corridor_entries]
        at_corridor_entry = np.array([all(i) for i in at_corridor_entry])

        # switch to floor, move to apartment entry
        switch = at_corridor_entry & idx
        if switch.any():
            for i in np.argwhere(switch).ravel():
                temp_cor = np.zeros(n, dtype=bool)
                temp_cor[i] = True
                corridor = self.population.building[i].corridor[self.population.home[i].floor_number]
                self.move_agents(temp_cor, corridor)
            self.population.target[switch] = self.apartment_entries[switch]

        # switch to apartment
        at_apartment_entry = self.population.position == self.apartment_entries
        at_apartment_entry = np.array([all(i) for i in at_apartment_entry])
        switch = at_apartment_entry & idx
        if switch.any():
            for i in np.argwhere(switch).ravel():
                temp_app = np.zeros(n, dtype=bool)
                temp_app[i] = True
                self.move_agents(temp_app, self.population.home[i].corridor)
            self.population.target[switch] = np.nan
            self.population.at_home[switch] = True
            self.population.busy[switch] = False

    def walk_to_office(self, idx):
        at_corridor_entry = self.population.position == [c.entry_point for c in self.public_corridors]
        at_corridor_entry = np.array([all(i) for i in at_corridor_entry])
        n = len(self.population)
        switch = at_corridor_entry & idx
        if switch.any():
            for b in self.building:
                bool_idx = switch & (self.population.building == b)
                if bool_idx.any():
                    self.move_agents(bool_idx, b.stairs)
            # 0 stairs, 1 lift
            chosen = np.array(random.choices([0, 1], weights=[47, 53], k=n), dtype=bool)
            always_lift = (self.floor_numbers > 8)
            always_stairs = (self.floor_numbers == 0)
            use_stairs = switch & ((~chosen & ~always_lift) | always_stairs)
            use_lift = switch & ((chosen & ~always_stairs) | always_lift)

            if use_lift.any():
                for b in self.building:
                    call_lift = use_lift & (self.population.building == b)
                    if call_lift.any():
                        self.population.target[call_lift] = np.array(b.stairs.lift_points)[
                            self.floor_numbers[call_lift]]

            if use_stairs.any():
                self.population.target[use_stairs] = [s.entry_point for s in self.stairs[use_stairs]]

        for b in self.building:
            in_lift = (b == self.population.building) & (b.lift == self.population.location)
            if in_lift.any():
                self.leave_lift(in_lift)

        wait_lift = np.zeros((n, 2), dtype=bool)
        for b in self.building:
            bool_idx = idx & (b == self.population.building)
            wait_lift[bool_idx] = self.population.position[bool_idx] == [
                np.array(b.stairs.lift_points)[self.floor_numbers[bool_idx]]]
        wait_lift = np.array([all(i) for i in wait_lift])

        if wait_lift.any():
            self.move_to_lift(wait_lift, np.zeros(n))

        at_building_entry = self.population.position == np.array([s.entry_point for s in self.stairs])
        at_building_entry = np.array([all(i) for i in at_building_entry])

        # leave apartment building
        leave = at_building_entry & idx
        if leave.any():
            for i in np.argwhere(leave).ravel():
                temp_cor = np.zeros(n, dtype=bool)
                temp_cor[i] = True
                self.move_agents(temp_cor, self.working[i])
            self.population.target[leave] = np.nan
            self.population.is_outside[leave] = True
            self.population.in_building[leave] = False
            self.population.motion_mask[leave] = False

    def leave_lift(self, in_lift):
        n = len(self.population)
        outside = self.population.working.ravel()
        for b in self.building:
            home = b == self.population.building
            # move back to apartment
            at_floor = np.zeros(n, dtype=bool)
            if (in_lift & ~outside).any():
                at_floor = self.floor_numbers == b.lift.floor_number

            leave = at_floor & in_lift & home & ~outside
            if leave.any():
                for i in np.argwhere(leave).ravel():
                    temp_cor = np.zeros(n, dtype=bool)
                    temp_cor[i] = True
                    pos = list(np.copy(self.population.target[i]))
                    self.move_agents(temp_cor, self.population.building[i].stairs)
                    self.population.building[i].lift.available_seats.append(pos)
                self.population.target[leave] = self.corridor_entries[leave]
                self.population.position[leave] = self.corridor_entries[leave]

            # move outside
            at_floor = np.zeros(n, dtype=bool)
            if (in_lift & outside).any:
                at_floor = np.zeros(n) == b.lift.floor_number

            leave = at_floor & in_lift & home & outside
            if leave.any():
                for i in np.argwhere(leave).ravel():
                    temp_cor = np.zeros(n, dtype=bool)
                    temp_cor[i] = True
                    pos = list(np.copy(self.population.target[i]))
                    self.move_agents(temp_cor, self.population.office[i])
                    self.population.building[i].lift.available_seats.append(pos)

                self.population.target[leave] = np.nan
                self.population.is_outside[leave] = True
                self.population.in_building[leave] = False
                self.population.motion_mask[leave] = False

    def move_to_lift(self, wait_lift, destinations):
        n = len(self.population)
        starts = np.copy(self.floor_numbers)
        starts[destinations != 0] = 0
        lift_number = np.array([b.lift.floor_number for b in self.population.building])
        on_floor = (lift_number == starts) & wait_lift

        for b in self.building:
            b.lift.call_lift(starts[wait_lift])
            if on_floor.any():
                move = (self.population.building == b) & on_floor
                if move.any():
                    move_in_lift = min(len(b.lift.available_seats), move.sum())
                    for i, idx in enumerate(np.argwhere(move)):
                        bool_idx = np.zeros(n, dtype=bool)
                        if i >= move_in_lift:
                            break
                        bool_idx[idx] = True
                        self.move_agents(bool_idx, b.lift)
                        b.lift.call_lift(destinations[bool_idx])
                        pos = b.lift.available_seats.popleft()
                        self.population.target[bool_idx] = pos
                        self.population.position[bool_idx] = pos

    def step(self, t):
        if not self.population:
            return

        n = len(self.population)
        at_home = self.population.at_home.ravel()
        in_corridor = [self.population.location == self.corridor]
        in_corridor = np.array([c for sub in in_corridor for c in sub])

        self.move_from_corridor(in_corridor)
        in_corridor = [self.population.location == self.corridor]
        in_corridor = np.array([c for sub in in_corridor for c in sub])
        if hasattr(self.population, "working"):
            # Make people come home
            come_home = ~self.population.working.ravel() & ~self.population.at_home.ravel()
            if come_home.any():
                self.walk_to_apartment(come_home)

            # Send people to work
            send_to_work = (self.population.working & ~self.population.is_outside).ravel()
            if send_to_work.any():
                self.move_to_corridor(send_to_work & ~in_corridor & at_home,
                                      target=np.array([c.entry_point for c in self.corridor]))
                in_cor = send_to_work & in_corridor
                if in_cor.any():
                    self.population.target[in_cor] = [c.entry_point for c in self.corridor[in_cor]]

                at_room_entry = np.zeros((n, 2), dtype=bool)
                at_room_entry[at_home] = self.population.position[at_home] == [i.entry_point for i in
                                                                               self.corridor[at_home]]
                at_room_entry = np.array([all(i) for i in at_room_entry])

                # walk to stairs
                go_to_stairs = in_corridor & send_to_work & at_room_entry & at_home
                if go_to_stairs.any():
                    for i in np.argwhere(go_to_stairs).ravel():
                        temp_cor = np.zeros(n, dtype=bool)
                        temp_cor[i] = True
                        corridor = self.population.building[i].corridor[self.population.home[i].floor_number]
                        self.move_agents(temp_cor, corridor)
                        self.population.target[i] = corridor.entry_point
                    self.population.at_home[go_to_stairs] = False

                # go_outside
                go_out = send_to_work & ~at_home
                if go_out.any():
                    self.walk_to_office(go_out)

        if hasattr(self.population, "eat"):
            in_kitchen = [self.population.location == self.kitchen]
            in_kitchen = np.array([k for sub in in_kitchen for k in sub])
            in_dining = [self.population.location == self.diningroom]
            in_dining = np.array([d for sub in in_dining for d in sub])

            # let people start preparing
            start_prepare = (
                    self.population.eat & ~self.population.is_preparing & ~self.population.is_eating).ravel()
            start_prepare = start_prepare & at_home

            if start_prepare.any():
                sp = start_prepare & ~in_kitchen & ~in_corridor
                if sp.any():
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

        if hasattr(self.population, "bath"):
            in_bathroom = [self.population.location == self.bath]
            in_bathroom = np.array([d for sub in in_bathroom for d in sub])

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

        if hasattr(self.population, "isolated"):
            isolated = self.population.isolated.ravel()
            if isolated.any():
                self.population.position[isolated] = self.population.sleeping_pos[isolated]
                self.population.target[isolated] = self.population.sleeping_pos[isolated]
        return
