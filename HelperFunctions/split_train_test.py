"""Split dataset in test and training folders randomly"""
import argparse
import os
import shutil
from os import path, walk

from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument(
    "--datasetPath",
    help="Path to directory",
)
parser.add_argument("--outputPath", default="Fender/broken")
parser.add_argument("--test_size", help="test/eval dataset difference", default=0.2, type=float)
parser.add_argument("--toBagOfWords", default=True)
args = parser.parse_args()


def clean_or_create_directory(directory):
    """Function for a "fresh start" in a directory, clean all files in the directory (if there are),
    or create the directory and all intermediate ones"""
    if path.exists(directory):
        shutil.rmtree(directory)
        os.mkdir(directory)
    else:
        os.makedirs(directory)


if args.toBagOfWords:
    baseDir = os.path.dirname(args.outputPath)
    typeDir = os.path.basename(args.outputPath)
    trainOutDir = path.join(baseDir, "bagOfWords", "train", typeDir)
    testOutDir = path.join(baseDir, "bagOfWords", "test", typeDir)
else:
    trainOutDir = path.join(args.outputPath, "train")
    testOutDir = path.join(args.outputPath, "test")

clean_or_create_directory(trainOutDir)
clean_or_create_directory(testOutDir)


f = []
for (dirpath, dirnames, filenames) in walk(args.datasetPath):
    f.extend(filenames)
    break
x_train, x_eval = train_test_split(f, test_size=0.2)

for file in x_train:
    shutil.copyfile(path.join(args.datasetPath, file), path.join(trainOutDir, file.replace(" ", "_")))
for file in x_eval:
    shutil.copyfile(path.join(args.datasetPath, file), path.join(testOutDir, file.replace(" ", "_")))
print("doen")
