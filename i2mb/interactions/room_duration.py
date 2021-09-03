import numpy as np

from i2mb.utils import global_time

'''
module to create output data for evaluation -> durations and contact interactions
'''


class RoomDuration:
    def __init__(self, population):

        self.population = population
        n = len(population)
        # at home
        self.duration_livingroom = np.zeros((n,))
        self.duration_diningroom = np.zeros((n,))
        self.duration_kitchen = np.zeros((n,))
        self.duration_bedroom = np.zeros((n,))
        self.duration_bath = np.zeros((n,))
        self.duration_corridor = np.zeros((n,))

        # public spaces
        self.duration_lift = np.zeros((n,))
        self.duration_stairs = np.zeros((n,))
        self.duration_public_corridor = np.zeros((n,))

        self.duration_outside = np.zeros((n,))

        self.corridor = np.array([a.corridor for a in self.population.home])
        self.livingroom = np.array([a.living_room for a in self.population.home])
        self.bath = np.array([a.bathroom for a in self.population.home])
        self.diningroom = np.array([a.dining_room for a in self.population.home])
        self.kitchen = np.array([a.kitchen for a in self.population.home])

        self.floor_numbers = np.array([a.floor_number for a in self.population.home], dtype=int)
        self.public_corridors = []
        for b in set(self.population.building):
            self.public_corridors += [np.array(b.corridor)[self.floor_numbers]]
        self.public_corridors = np.array(self.public_corridors).ravel()
        self.stairs = np.array([b.stairs for b in self.population.building])
        self.lift = np.array([b.lift for b in self.population.building])

    def get_duration(self):
        s = "duration_livingroom: " + str(self.duration_livingroom) + "\n"
        s += "mean_livingroom: " + str(np.mean(self.duration_livingroom)) + "\n"
        s += "duration_diningroom: " + str(self.duration_diningroom) + "\n"
        s += "mean_diningroom: " + str(np.mean(self.duration_diningroom)) + "\n"
        s += ("duration_kitchen: " + str(self.duration_kitchen)) + "\n"
        s += ("mean_kitchen: " + str(np.mean(self.duration_kitchen))) + "\n"
        s += ("duration_bedroom: " + str(self.duration_bedroom)) + "\n"
        s += ("mean_bedroom: " + str(np.mean(self.duration_bedroom))) + "\n"
        s += ("duration_bath: " + str(self.duration_bath)) + "\n"
        s += ("mean_bath: " + str(np.mean(self.duration_bath))) + "\n"
        s += ("duration_apartment_corridor: " + str(self.duration_corridor)) + "\n"
        s += ("mean_apartment_corridor: " + str(np.mean(self.duration_corridor))) + "\n"
        s += ("_________public_______") + "\n"
        s += ("duration_stairs: " + str(self.duration_stairs)) + "\n"
        s += ("mean_stairs: " + str(np.mean(self.duration_stairs))) + "\n"
        s += ("duration_lift: " + str(self.duration_lift)) + "\n"
        s += ("mean_lift: " + str(np.mean(self.duration_lift))) + "\n"
        s += ("duration_public_corridor: " + str(self.duration_public_corridor)) + "\n"
        s += ("mean_public_corridor: " + str(np.mean(self.duration_public_corridor))) + "\n"
        s += ("_________outside_______") + "\n"
        s += ("duration_outside: " + str(self.duration_outside)) + "\n"
        s += ("mean_outside: " + str(np.mean(self.duration_outside))) + "\n"
        return s

    def step(self, t):
        if t % (global_time.time_scalar) == 0:
            n = len(self.population)
            # at home
            self.duration_livingroom = np.zeros((n,))
            self.duration_diningroom = np.zeros((n,))
            self.duration_kitchen = np.zeros((n,))
            self.duration_bedroom = np.zeros((n,))
            self.duration_bath = np.zeros((n,))
            self.duration_corridor = np.zeros((n,))

            # public spaces
            self.duration_lift = np.zeros((n,))
            self.duration_stairs = np.zeros((n,))
            self.duration_public_corridor = np.zeros((n,))

            self.duration_outside = np.zeros((n,))

        in_livingroom = [self.population.location == self.livingroom]
        in_livingroom = np.array([d for sub in in_livingroom for d in sub])
        if in_livingroom.any():
            self.duration_livingroom[in_livingroom] += 1

        in_diningroom = [self.population.location == self.diningroom]
        in_diningroom = np.array([d for sub in in_diningroom for d in sub])
        if in_diningroom.any():
            self.duration_diningroom[in_diningroom] += 1

        in_kitchen = [self.population.location == self.kitchen]
        in_kitchen = np.array([d for sub in in_kitchen for d in sub])
        if in_kitchen.any():
            self.duration_kitchen[in_kitchen] += 1

        in_bedroom = [self.population.location == self.population.bedroom]
        in_bedroom = np.array([d for sub in in_bedroom for d in sub])
        if in_bedroom.any():
            self.duration_bedroom[in_bedroom] += 1

        in_bath = [self.population.location == self.bath]
        in_bath = np.array([d for sub in in_bath for d in sub])
        if in_bath.any():
            self.duration_bath[in_bath] += 1

        in_corridor = [self.population.location == self.corridor]
        in_corridor = np.array([c for sub in in_corridor for c in sub])
        if in_corridor.any():
            self.duration_corridor[in_corridor] += 1

        in_lift = [self.population.location == self.lift]
        in_lift = np.array([d for sub in in_lift for d in sub])
        if in_lift.any():
            self.duration_lift[in_lift] += 1

        in_stairs = [self.population.location == self.stairs]
        in_stairs = np.array([d for sub in in_stairs for d in sub])
        if in_stairs.any():
            self.duration_stairs[in_stairs] += 1

        in_public_corridor = [self.population.location == self.public_corridors]
        in_public_corridor = np.array([d for sub in in_public_corridor for d in sub])
        if in_public_corridor.any():
            self.duration_public_corridor[in_public_corridor] += 1

        is_outside = self.population.is_outside.ravel()
        if is_outside.any():
            self.duration_outside[is_outside] += 1
