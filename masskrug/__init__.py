from .engine.model import Model


class NullModel(Model):
    pass


NullModel = NullModel()


def __get_asset_path():
    import masskrug.assets as assets
    import os
    path = os.path.dirname(assets.__file__)
    return path


_assets_dir = __get_asset_path()
