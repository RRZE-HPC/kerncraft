"""Kerncraft static analytical performance modeling framework and tool."""
__version__ = '0.8.2'

# To trigger travis deployment to pypi, do the following:
# 1. Increment __version___
# 2. commit to RRZE-HPC/kerncraft's master branch
# 3. wait for travis to complete successful (unless already tested)
# 4. tag commit with 'v{}'.format(__version__) (`git tag vX.Y.Z`)
# 5. push tag to github (`git push origin vX.Y.Z` or push all tags with `git push --tags`)


def get_header_path() -> str:
    """Return local folder path of header files."""
    import os
    return os.path.abspath(os.path.dirname(os.path.realpath(__file__))) + '/headers/'
