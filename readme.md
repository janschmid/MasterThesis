# Overview
The aim of this work is to extend the autonomous quay inspection. This work is split into two parts: The object detection with DNN, for which Nvidia TAO is used.
The second part deals with the object classification, i.e. if an anomaly was detected or not.

## Get around in this repository
Due the fact that only a part of the work is open source, additional, closed source repositories are required to get the flow working. It might be possible to re-use parts of the work. The repository is structured as following:
- [FenderDetection](FenderDetection): Detect fenders at the quay area. 2 approachs are currently evaluated: Traditional image processing algorithms and DNN
- [HelperFunctions](HelperFunctions): Contains scripts to organize, split, separate the dataset and visualize the data.
- [Deepstream](deepstream): Run trained object detection model with deepstream 6, saves output of each frame inside folder. This should be only used for development purpose, not in production.