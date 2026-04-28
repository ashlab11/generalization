import os


def repo_data_dir():
    return os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
