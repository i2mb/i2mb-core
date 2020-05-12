class Contact:
    def __init__(self):
        # When was the last encounter
        self.__latest = None

        # Length of the last encounter
        self.__current = None

        # Length of longest encounter
        self.__longest = None

        self.__encounters = []

    @property
    def encounters(self):
        return self.__encounters

    @property
    def latest(self):
        return self.__latest

    @property
    def longest(self):
        return self.__longest

    @property
    def current(self):
        return self.__current

    def select(self, duration, keep_last=True):
        last = self.__encounters[-1]
        self.__encounters = [(t0, t1) for t0, t1 in self.__encounters[:-1] if t1 - t0 >= duration]
        if keep_last:
            self.__encounters.append(last)
            return

        if last[1] - last[0] >= duration:
            self.__encounters.append(last)
            return

        # Check if there are still valid encounters.
        if self.__encounters:
            return

        self.__latest = None
        self.__longest = None
        self.__current = None

    def add_encounter(self, t, use_last=False):
        if not self.__encounters:
            self.__encounters.append([t, t])
            self.__latest = t
            self.__current = 0
            self.__longest = 0
            return

        t_0, t_1 = self.__encounters[-1]
        if t - t_1 > 1:
            if use_last:
                self.__encounters[0] = [t, t]
            else:
                self.__encounters.append([t, t])

            self.__latest = t
            self.__current = 0
            return

        self.__encounters[-1][1] = t
        self.__current += 1
        if self.__current > self.__longest:
            self.__longest = self.__current

    def __repr__(self):
        return f"longest:{self.__longest}, latest:{self.__latest}, out of: {len(self.__encounters)}"


class ContactList:
    def __init__(self, track_time=None):
        self.track_time = track_time
        self.contacts = {}
        self.enabled = True

    def update(self, list_, t, duration=None, use_last=False):
        if not self.enabled:
            return

        for id_ in list_:
            contact = self.contacts.setdefault(id_, Contact())
            contact.add_encounter(t, use_last)
            if duration is not None:
                contact.select(duration)

        if self.track_time is not None:
            self.prune(t)

    def prune(self, timestamp):
        """Removes expired contacts. Contacts expire when the latest encounter occurred earlier than track_time."""
        if self.track_time is None:
            return

        self.contacts = {cid: contact for cid, contact in self.contacts.items()
                         if timestamp - contact.latest <= self.track_time}

    def enforce_duration(self, duration):
        if not self.enabled:
            return

        for c in self.contacts.values():
            c.select(duration, keep_last=False)

        self.contacts = {cid: contact for cid, contact in self.contacts.items()
                         if len(contact.encounters) > 0}

    def __repr__(self):
        return f"{self.contacts}"
