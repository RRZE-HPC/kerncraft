"""Kerncraft static analytical performance modeling framework and tool."""
__version__ = '0.5.9'


def get_header_path():
    """Return local folder path of header files."""
    import os
    return os.path.abspath(os.path.dirname(os.path.realpath(__file__))) + '/headers/'
