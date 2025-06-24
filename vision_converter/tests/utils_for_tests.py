import os


def normalize_path(path):
    return os.path.normcase(str(path).replace('\\\\?\\', ''))