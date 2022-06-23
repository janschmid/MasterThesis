"""Add script dirs to site..."""
import os
import site

script_dir = os.path.dirname(os.path.realpath(__file__))
site.addsitedir(os.path.join(script_dir, "../"))
site.addsitedir(os.path.join(script_dir, "../../HelperFunctions"))
site.addsitedir(os.path.join(script_dir, "../../early-stopping-pytorch"))
site.addsitedir(os.path.join(script_dir, "../../Classification-Report/src"))
