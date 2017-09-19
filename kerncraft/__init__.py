__version__ = '0.5.0'


def get_header_path():
    import os
    return os.path.abspath(os.path.dirname(os.path.realpath(__file__))) + '/headers/'
