import re

VERSION = "0.1.2"


def is_release_version():
    return bool(re.match(r"^\d+\.\d+\.\d+$", VERSION))