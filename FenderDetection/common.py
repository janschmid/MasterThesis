"""Contains all common function which are used for color_thresholder and/or deepspeed and/or deepstream"""
import os
import shutil
from cv2 import cv2
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def get_predictions(y, yhat):
    """Calculate TP, FP, FN, TN, return list of encodings.
    If called from pandas dataframe, dataframe needs to be sorted before
    :param y: true value
    :param yhat: pred value
    :return: encoding list"""
    pred_endocing = []
    if(max(yhat)==1):
        for i in range(len(yhat)):
            if y[i] == True and yhat[i] == True:
                # TP
                pred_endocing.append("True Positive - IS: broken, EXP: broken")
            elif y[i] == False and yhat[i] == True:
                pred_endocing.append("False Positive - IS: unbroken, EXP: broken")
                # FP
            elif y[i] == True and yhat[i] == False:
                # FN
                pred_endocing.append("False Negative - IS: broken, EXP: unbroken")
            else:
                pred_endocing.append("True Negative - IS: unbroken, EXP: unbroken")
                # TN
    else:
        for i in range(len(yhat)):
            if y[i] ==  yhat[i]:
                # TP
                pred_endocing.append("Correct prediction")
            else:
                pred_endocing.append("Wrong prediction")
    return pred_endocing

def cleanup_tensorboard(
    paths=[
        "tensorboard",
        "tensorboard_ds",
    ],
    min_steps=40,
    min_samples=50000,
):
    """Clean up tensorboard records on seleted folder, if "Loss/train" exists, min_steps are used, if
    Train/Samples/train_loss exists, min_samples is used as threshold for deletion
    """
    for path in paths:
        for root, dirs, files in os.walk(path, topdown=False):
            for file in files:
                file_path = os.path.join(root, file)
                event_acc = EventAccumulator(file_path)
                event_acc.Reload()
                remove = False
                if "13:08:49" in file_path:
                    print("wtf")
                if len(event_acc.Tags()["scalars"]) < 2:
                    remove = True
                elif "Loss/train" in event_acc.Tags()["scalars"]:
                    if len(event_acc.Scalars("Loss/train")) < min_steps:
                        remove = True
                elif "Train/Samples/train_loss" in event_acc.Tags()["scalars"]:
                    if len(event_acc.Scalars("Train/Samples/train_loss")) < min_samples:
                        remove = True

                if remove:
                    print(file_path)
                    # os.remove(file_path)
            for dir in dirs:
                if len(os.listdir(os.path.join(root, dir))) == 0:  # Check if the folder is empty
                    shutil.rmtree(os.path.join(root, dir))  # If so, delete it


if __name__ == "__main__":
    cleanup_tensorboard()
