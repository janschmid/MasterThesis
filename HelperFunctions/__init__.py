"""For some reason, pre-commit hook requires some comment to prevent failing"""
import os
import site

script_dir = os.path.dirname(os.path.realpath(__file__))
site.addsitedir(os.path.join(script_dir, "../../LorenzTao/dataset_convert"))
