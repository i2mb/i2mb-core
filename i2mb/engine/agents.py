import numpy as np


def vectorized(fn):
    AgentList.particle_properties.append(fn.__name__)
    return fn


class ArrayView:
    def __init__(self, parent, p_index):
        self.__parent = parent
        self.__p_index = p_index
        self.__cv = parent[p_index]

    def __setitem__(self, key, value):
        key = self.__p_index[key]
        self.__parent[key] = value

    def __getitem__(self, item):
        return self.__cv.__getitem__(item)

    def __getattribute__(self, item):
        if item in [f"_ArrayView__{v}" for v in ["parent", "p_index", "cv"]]:
            return object.__getattribute__(self, item)

        return object.__getattribute__(self, "_ArrayView__cv").__getattribute__(item)

    def __invert__(self):
        return ~self.__cv

    def __eq__(self, other):
        return self.__cv == other


class AgentListView:
    def __init__(self, agents, index):
        self.__source = agents
        self.__index = index
        self.__len = len(index)

    @property
    def index(self):
        return self.__index

    @index.setter
    def index(self, v):
        self.__index = v
        self.__len = len(v)

    def __len__(self):
        return self.__len

    def __getitem__(self, item):
        return self.__source.getitem(self.index)[item]

    def __getattribute__(self, item):
        if item not in AgentList.particle_properties:
            return object.__getattribute__(self, item)

        attribute = self.__source.__getattribute__(item)
        a_view = ArrayView(attribute, self.index)
        return a_view

    def find_indexes(self, idx):
        return (self.__index.reshape((-1, 1)) == idx.ravel()).any(axis=1).ravel()

    def add(self, ids):
        old_ix = self.__index
        new_index = np.union1d(old_ix, ids)
        self.__index = new_index
        self.__len = len(new_index)

    def remove(self, ids):
        old_ix = self.__index
        new_ix = np.setdiff1d(old_ix, ids)
        self.__index = new_ix
        self.__len = len(new_ix)


class AgentList:
    particle_properties = []
    list_properties = []

    def __init__(self, agents=0):
        if not isinstance(agents, int):
            self.index = np.array(agents)
        else:
            self.index = np.array(range(agents))

        self.__agents = np.array([Agent(id_) for id_ in self.index])

        self.__view = False

        # Flag to speed up processing
        self.__updated = True
        self.__update_time = None
        self.__current_time = True

    def set_time(self, t):
        self.__current_time = t

    @property
    def updated(self):
        if self.__current_time == self.__update_time:
            return not self.__updated

        return self.__updated

    @updated.setter
    def updated(self, v):
        if self.__current_time == self.__update_time:
            # Prevent switching off the update flag if another module already changed it to True.
            self.__updated |= v
        else:
            self.__updated = v

    def __len__(self):
        return len(self.index)

    def __iter__(self):
        return iter(self.__agents)

    def getitem(self, item):
        return self.__agents[item]

    def __getitem__(self, item):
        """:return Agent item, or AgentList view with selected agents."""
        if isinstance(item, int) or isinstance(item, np.integer):
            return self.__agents[item]

        view = AgentListView(self, self.index[item])
        return view

    def __getattribute__(self, item):
        if item in AgentList.particle_properties and self.__view:
            return object.__getattribute__(self, item)[self.index]

        return object.__getattribute__(self, item)

    def add_property(self, prop, values, l_property=False):
        if self.__view:
            raise RuntimeWarning("Adding properties to AgentList views si not supported.")

        object.__setattr__(self, prop, values)

        if l_property:
            AgentList.list_properties.append(prop)
            return

        AgentList.particle_properties.append(prop)

        # If values is an numpy array, we want to keep a pointer.
        for p, v in zip(self.__agents, values):
            object.__setattr__(p, prop, v)

    # def find_indexes(self, idx):
    #     return (self.index.reshape((-1, 1)) == idx.ravel()).any(axis=1).ravel()


class Agent:
    def __init__(self, id_):
        self.__id = id_

    @property
    def id(self):
        return self.__id

