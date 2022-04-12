import networkx as nx

from i2mb.interactions.base_interaction import Interaction
from i2mb.utils.spatial_utils import contacts_within_radius


class ContactHistory(Interaction):
    def __init__(self, network: nx.DiGraph, radius, population):
        super().__init__()
        self.population = population
        self.radius = radius
        self.network = network

        self.track_history = {}
        self.track_history_seen_contacts = set()

        self.file = None

    def post_init(self, base_file_name=None):
        super().post_init(base_file_name=base_file_name)
        self.base_file_name = f"{self.base_file_name}_contact_history.csv"
        self.file = open(self.base_file_name, "w+")

    def step(self, t):
        self.track_history_seen_contacts.clear()

        # Keep track of last encounter
        contacts = contacts_within_radius(self.population, self.radius)
        for region_contacts in contacts:
            for r in region_contacts:
                contact_pair = tuple(r)
                self.track_history_seen_contacts.add(contact_pair)
                contact_type = "random"
                if contact_pair in self.network.edges:
                    contact_type = self.network.edges[contact_pair]["type"]

                if contact_pair in self.track_history:
                    self.track_history[contact_pair]["duration"] += 1
                else:
                    self.track_history[contact_pair] = dict(
                        type=contact_type,
                        contact_started=t,
                        duration=1,
                        location=type(self.population.location[contact_pair[0]]).__name__
                    )

    def save_to_file(self, t):
        contacts_not_seen = set(self.track_history) - self.track_history_seen_contacts
        for contact in contacts_not_seen:

            line = *contact, t, *[self.track_history[contact][k] for k in ["type", "contact_started", "duration",
                                                                           "location"]]
            line = [f"{str(l)}" for l in line]

            self.file.write(", ".join(line) + "\n")
            del self.track_history[contact]

    def __del__(self):
        self.file.close()
