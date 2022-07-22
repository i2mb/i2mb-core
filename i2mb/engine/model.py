class Model:
    def __init__(self):
        self.base_file_name = "./"

    def pre_step(self, t):
        """Prepare to take the step."""
        pass

    def step(self, t):
        pass

    def post_step(self, t):
        """Clean up after step"""
        pass

    def final(self, t):
        pass

    def post_init(self, base_file_name=None):
        if base_file_name is not None:
            self.base_file_name = base_file_name

    def save_to_file(self, t):
        pass
