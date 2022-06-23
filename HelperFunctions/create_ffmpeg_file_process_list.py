"""Create list of .jpg files in same directory as files, same ordering then ffmpeg glob,
intendet use-case: Parse output (csv) list in deepstream, name output files of detections to automatically separate broken/unbroken samples"""
import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np


def save_list(file_path, sort_explorer_style=False):
    """Create list of .jpg files in same directory as files, same ordering then ffmpeg glob"""
    script_dir = os.path.dirname(os.path.realpath(__file__))
    fileList = []
    ffmpegFileList = []
    max=float('inf')
    current = 0
    if(sort_explorer_style):
        sorted_dir = os.listdir(file_path)
        print("Sort in explorer style")
    else:
        sorted_dir = sorted(os.listdir(file_path))
        print("sort in ffmpeg style")
    
    split_at = 10000
    i=0
    for file in sorted_dir:
        if Path(file).suffix == ".jpg":
            ffmpegFileList.append("file '{0}'".format(os.path.join(file_path, file)))
            fileList.append(file)
            current+=1
            if(current>max):
                break
            i+=1
            if(i%split_at==0):
            # ffmpegFileList.append("duration 1.0")
                np.savetxt("ordered_file_list_ffmpeg{0}.csv".format(int(i/split_at)), ffmpegFileList, delimiter=",", fmt="%s")
                np.savetxt("ordered_file_list{0}.csv".format(int(i/split_at)), fileList, delimiter=",", fmt="%s")
                ffmpegFileList=[]
                fileList=[]
    np.savetxt("ordered_file_list_ffmpeg.csv", ffmpegFileList, delimiter=",", fmt="%s")
    np.savetxt("ordered_file_list.csv", fileList, delimiter=",", fmt="%s")


if __name__ == "__main__":
    parser = ArgumentParser("Create list of .jpg files in same directory as files, same ordering then ffmpeg glob")
    parser.add_argument("path", type=str, help="Path to images root directory")
    parser.add_argument("-s", "--sort_explorer_style", help="Sort files in same order than file explorer", default=False, action='store_true')
    args = parser.parse_args()
    save_list(args.path, args.sort_explorer_style)
