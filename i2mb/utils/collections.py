class ConstrainedDict(dict):
    def __init__(self, constraint, msg=None):
        super().__init__()
        if msg is None:
            msg = "Key {} already in constraint"
        self.msg = msg
        self.constraint = constraint

    def __setitem__(self, key, value):
        if key in self.constraint:
            raise KeyError(self.msg.format(key))

        super().__setitem__(key, value)
