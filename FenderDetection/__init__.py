"""Add used directories to path, clean and create temp dirs"""
import os
import os.path as p
import shutil
import site

from send2trash import send2trash

script_dir = os.path.dirname(os.path.realpath(__file__))
site.addsitedir(os.path.join(script_dir, "../HelperFunctions"))
# Create output directories relative to script dir
output_directories = [
    "../data/debug",
    "../data/tmp/",
]


def create_dirs(delete_before_creation=False, dataDir=None):
    """Create directories which are in static varialbe output_directories
    :param delete_before_creation: Remove directory tree, can be used for clean start"""
    if dataDir is not None:
        output_directories.append(p.join(dataDir, "broken"))
        output_directories.append(p.join(dataDir, "unbroken"))

    for d in sorted(output_directories, reverse=False):
        d = p.join(script_dir, d)
        if delete_before_creation and p.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)


create_dirs()
