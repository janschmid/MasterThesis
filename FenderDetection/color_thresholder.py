"""Check for broken fenders by color theshold, simple baseline comparison"""

import multiprocessing as mp
import os
import os.path as p
import pickle
import shutil
from argparse import ArgumentParser

import __init__
import common as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from custom_dataset import FenderDataset
from cv2 import cv2
from scipy.stats import norm
from separate_broken_samples import BrokenSamplesDetector
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from torchvision import transforms
from tqdm import tqdm
import time

script_dir = p.dirname(p.realpath(__file__))
data_dir = p.join(script_dir, "../data")
out_dir = p.join(data_dir, "debug")
rgb_values = []
hsv_values = []
pos = []


def show_images_store_selected_color_values(dataloader):
    """Visualize each image in folder, store hsv and rgb values where mouse clicked after each image to rgb_values.csv and hsv_values.csv"""
    last_pos = 0
    for input, lables in dataloader:
        img = input.numpy()
        cv2.imshow("img", img)
        cv2.setMouseCallback("img", _click_event)
        cv2.waitKey(0)
        last_pos = _extract_colors_from_image(img, pos, last_pos)
        cv2.destroyAllWindows()

        np.savetxt(p.join(data_dir, "hsv_values.csv"), hsv_values, delimiter=",", fmt="%f")


def _extract_colors_from_image(img, pos, last_pos):
    """Get value from image, pos and last_pos needs to be handled because callback is weired..."""
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    rgb_img = img
    height, width = img.shape[:2]
    print("width: {0}, height: {1}".format(width, height))
    for i in range(last_pos, len(pos)):
        x = pos[i][0]
        y = pos[i][1]
        print(x, " ", y)
        rgb_values.append((rgb_img[x, y]))
        hsv_values.append(hsv_img[x, y])
    return len(pos)


def remove_outliers_from_color_points(threshold):
    """z-score outlier removal, threshold defines z-score,
    saves results in rgb/hsv_values_outlier_removed.csv"""
    from scipy import stats

    df = pd.read_csv(p.join(data_dir, "hsv_values.csv"), header=None)
    old_min = (df[0].min(), df[1].min(), df[2].min())
    old_max = (df[0].max(), df[1].max(), df[2].max())
    # Format, for easy comparison
    df_without_outliers = df[(np.abs(stats.zscore(df)) < threshold).all(axis=1)].astype("object")
    if len(df_without_outliers) < 2:
        return None, None
    new_min = (df_without_outliers[0].min(), df_without_outliers[1].min(), df_without_outliers[2].min())
    new_max = (df_without_outliers[0].max(), df_without_outliers[1].max(), df_without_outliers[2].max())
    print("ouliers removed {2:.2f} MIN, old : {0}   new:{1}".format(old_min, new_min, threshold))
    print("ouliers removed {2:.2f} MAX, old : {0}   new:{1}".format(old_max, new_max, threshold))
    return new_min, new_max


def visualize_result(dataloader):
    """Visualize original image, rgb and hsv segementation for each image in folder, loads values from data directory"""
    hsv_df = pd.read_csv(p.join(out_dir, "hsv_values.csv"), header=None)
    hsv_df.columns = ["h", "s", "v"]

    hsv_min_range = np.array([hsv_df["h"].min(), hsv_df["s"].min(), hsv_df["v"].min()])
    hsv_max_range = np.array([hsv_df["h"].max(), hsv_df["s"].max(), hsv_df["v"].max()])
    for input, label in dataloader:
        for i in range(0, dataloader.batch_size):
            rgb_img = np.array(transforms.ToPILImage()(input[i]))
            cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
            hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
            segmentation_img_hsv = cv2.inRange(hsv_img, hsv_min_range, hsv_max_range)
            segmentation_img_hsv = cv2.cvtColor(segmentation_img_hsv, cv2.COLOR_GRAY2RGB)
            cv2.imshow("Image {0}".format(""), rgb_img)
            cv2.imshow("Segmentation {0}".format("HSV"), segmentation_img_hsv)
            cv2.waitKey(0)


def calculate_2d_confusion_matrix_stats(cm):
    """Calculate stats of confusion matrix, input: sklearn confustion matrix for 2D"""
    TN = cm[0][0]
    FN = cm[1][0]
    TP = cm[1][1]
    FP = cm[0][1]

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN)
    # Specificity or true negative rate
    TNR = TN / (TN + FP)
    # Precision or positive predictive value
    PPV = TP / (TP + FP)
    # Negative predictive value
    NPV = TN / (TN + FN)
    # Fall out or false positive rate
    FPR = FP / (FP + TN)
    # False negative rate
    FNR = FN / (TP + FN)
    # False discovery rate
    FDR = FP / (TP + FP)

    # Overall accuracy
    ACC = (TP + TN) / (TP + FP + FN + TN)
    return TPR, TNR, PPV, NPV, FPR, FNR, FDR, ACC


def rescale_fender_img(img, is_rotated=False):
    """Required for plotting when using np.hstack with rotated images.
    If the image is not rotated, it checks that the height is greater or equal to the width.
    If the image is rotated, it zero-padds the height to the width"""
    x = img.shape[0]
    y = img.shape[1]

    if not is_rotated:
        if x < y:
            # let's make a rectangle
            img_rescaled = cv2.copyMakeBorder(img.copy(), 0, y - x, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            return img_rescaled

    else:
        if x < y:
            img_rescaled = cv2.copyMakeBorder(img.copy(), y - x, 0, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            return img_rescaled
    return img


hist_result_list = mp.Manager().list()


def __calculate_histogram_result_table_async(rgb_img, name, hsv_min_range, hsv_max_range, detector, dry_run):
    """Call calculate_histogram_from_binary image, add results to global list hist_result_list
    :param rgb_img: path to rgb image
    :param name: name of image
    :param hsv_min_range: min threshold for hsv image for binary image
    :param hsv_max_range: max threshold for hsv image for binary image
    :param detector: intance of BrokenSamplesDetector
    :param dry_run: if set to true: stacked image is not saved at path, set to true for stat only, if result image is required, set to false
    """
    hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)

    segmentation_img_hsv = cv2.inRange(hsv_img, hsv_min_range, hsv_max_range)
    hist, hist_rot, mean_normal, mean_shifted = calculate_histogram_from_binary_image(segmentation_img_hsv)

    if not dry_run:

        segmentation_3c = cv2.cvtColor(segmentation_img_hsv, cv2.COLOR_GRAY2BGR)
        # stack multiple images for easier debugging...
        stacked = np.hstack(
            (
                rescale_fender_img(rgb_img),
                rescale_fender_img(segmentation_3c),
                rescale_fender_img(hist, True),
                rescale_fender_img(hist_rot),
            )
        )
        cv2.imwrite(p.join(out_dir, "broken" if detector.is_broken(name) else "unbroken", name), stacked)

    hist_result_list.append(
        (
            mean_normal,
            mean_shifted,
            1 if detector.is_broken(name) else 0,
            p.join(out_dir, "broken" if detector.is_broken(name) else "unbroken", name),
        )
    )


def calculate_histogram_result_table(dataloader, outlier_zscore=10, dry_run=False):
    """Call __calc_histogram_result_table_async for each entry in dataloader
    :param dataloader: dataloader of fender dataset
    :param outlier_zscore: remove hsv outliers based on zscore
    :param dry_run: set to true for stats only, if result should be visualized later, set to false. If true: "name" in hist_result_table is invalid!!!
    :return pd dataframe hist_result_table, columns=["mean_normal", "mean_shifted", "is_broken", "name"]
    """
    hsv_min_range, hsv_max_range = remove_outliers_from_color_points(outlier_zscore)
    if hsv_min_range is None or hsv_max_range is None:
        return None
    hist_result_list[:] = []

    detector = BrokenSamplesDetector()
    pool = mp.Pool(processes=mp.cpu_count())
    for input, label, path in dataloader:
        for i in range(0, dataloader.batch_size):
            rgb_img = cv2.cvtColor(np.array(transforms.ToPILImage()(input[i])), cv2.COLOR_BGR2RGB)

            name = os.path.basename(path[0])
            pool.apply_async(
                __calculate_histogram_result_table_async,
                args=(rgb_img, name, hsv_min_range, hsv_max_range, detector, dry_run),
            )
    pool.close()
    pool.join()
    hist_result_table = pd.DataFrame(list(hist_result_list))
    hist_result_table.columns = ["mean_normal", "mean_shifted", "is_broken", "name"]

    return hist_result_table


def visualize_stats(results_file_names="color_thresholder_results.csv"):
    """Save confusion matrix
    :param results_file_name: matrix with header \'threshold,f1_score,accuray,hit_rate,miss_rate,outlier_zscore\'"""
    if isinstance(results_file_names, str):
        results_file_names = [results_file_names]
    for result_file_name in results_file_names:
        confusion_matrix_path = p.join(data_dir, result_file_name)
        df = pd.read_csv(confusion_matrix_path, index_col="threshold")
        plt.ioff()
        df.plot()
        plt.savefig(p.join(data_dir, result_file_name.replace(".csv", ".jpg")))


def _click_event(event, x, y, flags, params):
    """Callback function onl click, store position in global variable pos"""
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        pos.append((y, x))


def color_histogram(img_path, outlier_zscore):
    """Calculate bw histogram from image
    :param img_path: path to image
    :param hsv_value_path: path to list of hsv values
    :param output_dir: result dir"""
    rgb_img = cv2.imread(img_path)
    hsv_min_range, hsv_max_range = remove_outliers_from_color_points(outlier_zscore)
    hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)

    segmentation_img_hsv = cv2.inRange(hsv_img, hsv_min_range, hsv_max_range)
    hist, hist_rot, mean_normal, mean_shifted = calculate_histogram_from_binary_image(segmentation_img_hsv)


def calculate_projection_histogram(binary):
    """Calculate projection matrix of a binary image. This is done by accumulating the non-masked pixels and plotting them on a histogram
    :param binary: binary image
    :return: projection histogram of binary image"""
    img_height, img_width = binary.shape
    proj = np.sum(binary, 1)
    m = binary.size
    result = np.zeros((proj.shape[0], img_width))
    if m != 0 and binary.max() > 0:
        for row in range(binary.shape[0]):
            line_end = int(proj[row] / binary.max()) - 1
            if line_end >= 0:
                cv2.line(result, (0, row), (line_end, row), (255, 255, 255), 1)
    return result


def subtract_mean_and_calculate_residual_mean(binary):
    """Calculate the mean of a projection histogram and subtract it from the projection histogram -> remove random noise
    :param binary: binary projection histogram
    :param return: projection histogram with removed mean, mean and standard deviation are plotted on the matrix, mean, std"""
    mean, std = norm.fit(binary)
    mean = cv2.countNonZero(binary) / binary.shape[0]
    for k in range(0, binary.shape[0]):
        binary[k, 0 : round(mean)] = 0

    mean, std = norm.fit(binary)
    binary = calculate_projection_histogram(binary)
    color_img = cv2.merge((binary, binary, binary))
    cv2.FONT_HERSHEY_PLAIN
    # cv2.putText(
    #     color_img,
    #     "mean: {0:.2f}, std: {1:.2f}".format(mean, std),
    #     bottom_left_corner_of_text,
    #     font,
    #     font_scale,
    #     font_color,
    #     thickness,
    #     line_type,
    # )
    return color_img, mean, std


def calculate_histogram_from_binary_image(binary):
    """Calculate binary color histogram from image and return 90 deg counterclockwise,
    :param binary: binary image,
    :return histogram, mean, std"""
    binary_rot = cv2.rotate(binary, cv2.ROTATE_90_CLOCKWISE)

    proj_normal = calculate_projection_histogram(binary)
    proj_shifted = calculate_projection_histogram(binary_rot)

    res_mean_bin_normal, mean_normal, std_normal = subtract_mean_and_calculate_residual_mean(proj_normal)
    res_mean_bin_shifted, mean_shifted, std_shifted = subtract_mean_and_calculate_residual_mean(proj_shifted)

    res_mean_bin_shifted = cv2.rotate(res_mean_bin_shifted, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return res_mean_bin_normal, res_mean_bin_shifted, mean_normal, mean_shifted


def classify(hist_result_table, classifier_model, fit_model=False):
    """Classify results in hist_result_table based on classifier
    :param hist_result_table: panddas dataframe, get by calling calculate_histogram_result_table
    :param classifier_model: sklearn classifier model, methods 'fit' and 'predict' need to be implemented
    :param fit_model: fit the model, if false model is only predicted -> need to be trained before
    "return: acc, f1score, hist_result_table, classifier_model"""
    hist_result_table = hist_result_table.sort_values(by=["mean_normal"], ascending=False)
    x = hist_result_table[["mean_normal", "mean_shifted"]].to_numpy()
    y = hist_result_table["is_broken"].to_numpy()
    if fit_model:
        classifier_model.fit(x, y)
    yhat = classifier_model.predict(x)

    hist_result_table["true_class"] = y
    hist_result_table["pred_class"] = yhat
    acc = accuracy_score(y, yhat)
    f1score = f1_score(y, yhat)
    return acc, f1score, hist_result_table, classifier_model


def visualize_2d_scatter(broken_csv_path, unbroken_csv_path, title, classifier_model):
    """Split broken and unbroken dataset with logitstic regression, visualize result on 2d scatter plot with plotly
    :param broken_csv_path: Path to broken images, this is generated by 'calculate_stats', file is in the data folder
    :param unbroken_csv_path: Path to unbroken images"""
    broken = pd.read_csv(broken_csv_path)
    broken = broken.sort_values(by=["mean_normal"], ascending=False)
    # broken.to_csv(broken_csv_path, index=False, Header=False)
    unbroken = pd.read_csv(unbroken_csv_path)
    unbroken = unbroken.sort_values(by=["mean_normal"], ascending=False)
    # #add column to konw state
    broken["broken"] = 1
    unbroken["broken"] = 0
    combined = broken.append(unbroken)
    combined.reset_index(drop=True, inplace=True)
    x = combined[["mean_normal", "mean_shifted"]].to_numpy()
    y = combined["broken"].to_numpy()
    classifier_model.fit(x, y)
    yhat = classifier_model.predict(x)

    combined["predictions"] = cm.get_predictions(y, yhat)

    import plotly_visualizer

    acc = accuracy_score(y, yhat)
    plotly_visualizer.setup(
        combined,
        "mean_normal",
        "mean_shifted",
        "predictions",
        "name",
        "{0} - Accuracy: {1:2f}".format(title, acc),
        p.dirname(unbroken_csv_path),
    )

    plotly_visualizer.app.run_server(debug=False)
    # plotly_visualizer.app.run_server(debug=True)

    print("Accuracy: {0:2f}".format(acc))


def visualize_classifiers_z_score_threshold(hist_result_table, visualize_table_friendly_name=None):
    """Create plotly_visualizer of hist_result_table
    :param hist_resutl_table: hist_result_table, get from calculate_histogram_result_table"""
    import plotly_visualizer

    conf_matrix = []
    scatter = []
    # for i in range(0, len(hist_result_table)):
    # for i in hist_result_table['z-score'].unique():
    scatter.append(
        plotly_visualizer.create_line_plot(
            hist_result_table, "train_acc", "z-score", "classifier", "Classifier Zscore Train acc"
        )
    )
    scatter.append(
        plotly_visualizer.create_line_plot(
            hist_result_table, "test_acc", "z-score", "classifier", "Classifier Zscore Test acc"
        )
    )
    scatter.append(
        plotly_visualizer.create_line_plot(
            hist_result_table, "f1_train", "z-score", "classifier", "Classifier Zscore Train F1 score"
        )
    )
    scatter.append(
        plotly_visualizer.create_line_plot(
            hist_result_table, "f1_test", "z-score", "classifier", "Classifier Zscore Test F1 score"
        )
    )
    for i in range(0, len(hist_result_table)):
        conf_matrix.append(
            plotly_visualizer.create_confustion_matrix(
                hist_result_table.iloc[i].test_table,
                hist_result_table.iloc[i].friendly_name,
                "true_class",
                "pred_class",
            )
        )
        #
    if visualize_table_friendly_name == None:
        best_result_table_train = hist_result_table.iloc[
            hist_result_table["test_acc"].argmax()
        ].train_table.sort_index()
        best_result_table_test = hist_result_table.iloc[hist_result_table["test_acc"].argmax()].test_table.sort_index()
    else:
        best_result_table_train = hist_result_table.iloc[
            hist_result_table[hist_result_table["friendly_name"] == visualize_table_friendly_name].index[0]
        ].train_table.sort_index()
        best_result_table_test = hist_result_table.iloc[
            hist_result_table[hist_result_table["friendly_name"] == visualize_table_friendly_name].index[0]
        ].test_table.sort_index()

    best_result_table_train["predicted"] = cm.get_predictions(
        best_result_table_train["true_class"], best_result_table_train["pred_class"]
    )
    best_result_table_test["predicted"] = cm.get_predictions(
        best_result_table_test["true_class"], best_result_table_test["pred_class"]
    )

    confusion_matrix(np.array(best_result_table_test["true_class"]), np.array(best_result_table_test["pred_class"]))

    best_result_test = plotly_visualizer.create_scatter_plot(
        best_result_table_test,
        "mean_normal",
        "mean_shifted",
        "predicted",
        "name",
        hist_result_table.iloc[hist_result_table["test_acc"].argmax()]["friendly_name"] + "test",
    )
    best_result_train = plotly_visualizer.create_scatter_plot(
        best_result_table_train,
        "mean_normal",
        "mean_shifted",
        "predicted",
        "name",
        hist_result_table.iloc[hist_result_table["train_acc"].argmax()]["friendly_name"] + "train",
        1,
    )
    plotly_visualizer.plot_html_figures([best_result_train, best_result_test] + scatter + conf_matrix)
    plotly_visualizer.app.run_server(host="127.0.0.1", port="8080", debug=False)

def measure_classifier_runtime(dataloader, classifier, outlier_zscore=10, load_simulation_data=True):
    hsv_min_range, hsv_max_range = remove_outliers_from_color_points(outlier_zscore)
    start = time.time()
    if(load_simulation_data):
        for i in range(0,len(dataloader)):
            data = torch.zeros(3,480,360)
            rgb_img = cv2.cvtColor(np.array(transforms.ToPILImage()(data)), cv2.COLOR_BGR2RGB)
            hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
            segmentation_img_hsv = cv2.inRange(hsv_img, hsv_min_range, hsv_max_range)
            hist, hist_rot, mean_normal, mean_shifted = calculate_histogram_from_binary_image(segmentation_img_hsv)
            yhat = classifier.predict([[mean_normal, mean_shifted]])
    else:
        for input, label, path in dataloader:
            for i in range(0, dataloader.batch_size):
                rgb_img = cv2.cvtColor(np.array(transforms.ToPILImage()(input[i])), cv2.COLOR_BGR2RGB)
                hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
                segmentation_img_hsv = cv2.inRange(hsv_img, hsv_min_range, hsv_max_range)
                hist, hist_rot, mean_normal, mean_shifted = calculate_histogram_from_binary_image(segmentation_img_hsv)
                yhat = classifier.predict([[mean_normal, mean_shifted]])
    end=time.time()
    return (end-start), len(dataloader)

if __name__ == "__main__":
    execution_choices = ["train", "verify", "stats", "hist", "2d_scatter"]
    parser = ArgumentParser("Fault detection based on color theshold")
    parser.add_argument(
        "--images_dir",
        "-i",
        help="Path to images",
    )
    parser.add_argument(
        "-e",
        "--execution_type",
        help="Train: Select colors on image, Verify: Visualize each image, Stats: No visualization, run thorugh all images and print results,\
        choices={0}".format(
            execution_choices
        ),
        required=True,
        nargs="+",
    )
    parser.add_argument("-c", "--clean", help="Clean output directories before start", action="store_true")
    parser.add_argument("-o", "--outputDir", help="Output dir to save files during run", required=True)
    args = parser.parse_args()
    for arg in args.execution_type:
        if arg not in execution_choices:
            print('arg not know: "{1}"\nallowed: {0}'.format(execution_choices, arg))
            # exit(-1)
    out_dir = p.join(data_dir, args.outputDir)
    __init__.create_dirs(args.clean, p.join(data_dir, args.outputDir))
    image_datasets = {x: FenderDataset(args.images_dir + x, transforms.PILToTensor()) for x in ["train", "test"]}
    dataloaders_dict = {
        x: torch.utils.data.DataLoader(image_datasets[x], shuffle=True, num_workers=4) for x in ["train", "test"]
    }

    if "train" in args.execution_type:
        show_images_store_selected_color_values(dataloaders_dict["train"])

    if "verify" in args.execution_type:
        visualize_result(dataloaders_dict["test"])

    if "stats" in args.execution_type:
        paths = []
        results = []
        if True:
            for i in tqdm(np.arange(1, 2.5, 0.2)):
                hist_result_table_test = calculate_histogram_result_table(dataloaders_dict["test"], i, True)
                hist_result_table_train = calculate_histogram_result_table(dataloaders_dict["train"], i, True)
                if hist_result_table_train is None or hist_result_table_test is None:
                    continue
                results.append((i, hist_result_table_train, hist_result_table_test))

            with open("hist_result_tables.pickle", "wb") as handle:
                pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # hist_result_table_test = calculate_histogram_result_table(dataloaders_dict['test'], 2.2, False)
        # hist_result_table_train = calculate_histogram_result_table(dataloaders_dict['train'], 1.8, False)
        if False:
            with open("hist_result_tables.pickle", "rb") as handle:
                hist_result_tables = pickle.load(handle)
                results = []
                for hist_result_table in hist_result_tables:
                    classifiers = [
                        # (
                        #     "KNN_3_zscore_{0}".format(hist_result_table[0]),
                        #     KNeighborsClassifier(n_neighbors=3, n_jobs=-1),
                        # ),
                        # (
                        #     "KNN_5_zscore_{0}".format(hist_result_table[0]),
                        #     KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
                        # ),
                        # (
                        #     "KNN_7_zscore_{0}".format(hist_result_table[0]),
                        #     KNeighborsClassifier(n_neighbors=7, n_jobs=-1),
                        # ),
                        (
                            "KNN_3_distance_zscore_{0}".format(hist_result_table[0]),
                            KNeighborsClassifier(n_neighbors=3, weights="distance", n_jobs=-1),
                        ),
                        (
                            "KNN_5_distance_zscore_{0}".format(hist_result_table[0]),
                            KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=-1),
                        ),
                        (
                            "KNN_7_distance_zscore_{0}".format(hist_result_table[0]),
                            KNeighborsClassifier(n_neighbors=7, weights="distance", n_jobs=-1),
                        ),
                        (
                            "Logistic_Regression_zscore_{0}".format(hist_result_table[0]),
                            LogisticRegression(class_weight="balanced"),
                        ),
                    ]
                    for classifier in classifiers:
                        acc_train, f1_train, knn_hist_result_table_train, model = classify(
                            hist_result_table[1], classifier[1], fit_model=True
                        )
                        acc_test, f1_test, knn_hist_result_table_test, model = classify(
                            hist_result_table[2], model, fit_model=False
                        )
                        results.append(
                            (
                                classifier[0],
                                classifier[0].split("_zscore")[0],
                                hist_result_table[0],
                                acc_train,
                                acc_test,
                                f1_train,
                                f1_test,
                                knn_hist_result_table_train,
                                knn_hist_result_table_test,
                                model,
                            )
                        )
                visualize_classifiers_z_score_threshold(
                    pd.DataFrame(
                        results,
                        columns=[
                            "friendly_name",
                            "classifier",
                            "z-score",
                            "train_acc",
                            "test_acc",
                            "f1_train",
                            "f1_test",
                            "train_table",
                            "test_table",
                            "model",
                        ],
                    ),
                    # "KNN_7_distance_zscore_2.2",
                )
                # hist_result_table_test = calculate_histogram_result_table(dataloaders_dict['test'], 2.2, False)
        # with open('debug.pickle', 'rb') as handle:
        #     results = pickle.load(handle)
        #     visualize_classifiers_z_score_threshold(results)
        # with open('hist_result_tables.pickle', 'rb') as handle:
        #     hist_result_table_train = pickle.load(handle)
        #
    if True:
        with open("hist_result_tables.pickle", "rb") as handle:
            hist_result_tables = pickle.load(handle)
            hist_result_table_train=hist_result_tables[5][1]
            hist_result_table_test=hist_result_tables[5][2]
            classifier = KNeighborsClassifier(n_neighbors=5, n_jobs=1, weights="distance")
            acc_train, f1_train, knn_hist_result_table_train, model = classify(
                hist_result_table_train, classifier, fit_model=True
                )
            acc_test, f1_test, knn_hist_result_table_test, model = classify(
                hist_result_table_test, classifier, fit_model=False
                )
            classification_report = classification_report(knn_hist_result_table_test['true_class'], knn_hist_result_table_test['pred_class'], labels=[0,1])
            print(classification_report)
            print("Start classification measurement...")
            runtime, num_images= measure_classifier_runtime(dataloaders_dict["test"], model, 2.0)
            print("Runtime test set: {0}, classified images: {1}".format(runtime, num_images))

    if "hist" in args.execution_type:
        base_path = "/media/jan/Data/ubuntu_data_dir/git/output/cropped_images/fender/images_broken_test"
        if os.path.exists(base_path + "_histogram"):
            shutil.rmtree(base_path + "_histogram")
        for img_path in os.listdir(base_path):
            color_histogram(
                p.join(base_path, img_path),
                "hsv_values_outlier_removed_zscore_2.csv",
                base_path + "_histogram",
            )
    if "2d_scatter" in args.execution_type:
        visualize_2d_scatter(
            p.join(out_dir, "debug_broken.csv"),
            p.join(out_dir, "debug_unbroken.csv"),
            args.outputDir + "K-",
            KNeighborsClassifier(n_neighbors=5),
        )
