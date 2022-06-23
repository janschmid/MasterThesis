#!/usr/bin/env python3
"""Multiple examples used as template functionality:
- Can be run with or without tracker
- Save detections to out_crops
- Can be modified to stream or save output (.mp4)
- To run it on images, convert images to video before with:
    ffmpeg framerate 1 -pattern_type glob -i '*.jpg' -colorspace 1 -codec h264 output_name.h264
- Run ffmpeg with cuda accell
    /usr/local/bin/ffmpeg -hwaccel cuda -framerate 1 -pattern_type glob -i '*.jpg' -colorspace 1 -codec h264 output_name.h264
"""
################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

import argparse
import configparser
import sys
from cairo import SCRIPT_MODE_BINARY

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstRtspServer", "1.0")
import math
import os
import os.path
import shutil
import sys
from ctypes import *
from os import path
import __init__
import numpy as np
import pyds
from bus_call import bus_call
from cv2 import FORMATTER_FMT_DEFAULT, cv2
from FPS import GETFPS
from gi.repository import GObject, Gst, GstRtspServer
from is_aarch_64 import is_aarch64
from kitti_format_handler import get_labels_from_kitti

fps_streams = {}
frame_count = {}
saved_count = {}
global PGIE_CLASS_ID_PERSON
# PGIE_CLASS_ID_BACKGROUND = 0
# PGIE_CLASS_ID_FENDER = 1
# PGIE_CLASS_ID_LADDER = 2
PGIE_CLASS_ID_BACKGROUND = 50
PGIE_CLASS_ID_FENDER = 0
PGIE_CLASS_ID_LADDER = 1
past_tracking_meta = [0]

MAX_DISPLAY_LEN = 64

MUXER_OUTPUT_WIDTH = 720
MUXER_OUTPUT_HEIGHT = 576
MUXER_BATCH_TIMEOUT_USEC = 400000
TILED_OUTPUT_WIDTH = 720
TILED_OUTPUT_HEIGHT = 576
STREAM_HEIGHT = 1080
STREAM_WIDTH = 1920
GST_CAPS_FEATURES_NVMM = "memory:NVMM"
# pgie_classes_str = ["Background", "Fender", "Ladder"]
pgie_classes_str = ["Fender", "Ladder"]

MIN_CONFIDENCE = 0.3
MAX_CONFIDENCE = 0.4

def detect_if_img_is_valid(img):
    """Bug fix, sometimes images are black, we don't want them..."""
    grayScale = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    average = grayScale.mean(axis=0).mean(axis=0)
    if average > 1:
        return True
    else:
        print("Skipped export, image is not valid...")
        return False


def tiler_sink_pad_buffer_probe(pad, info, u_data):
    """WIP, save cropped and full images to output, show detection in stream"""
    frame_number = 0
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return
    # Retrieve batch metadata from the gst_buffer
    # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
    # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        uniqueId = None
        try:
            # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
            # The casting is done by pyds.NvDsFrameMeta.cast()
            # The casting also keeps ownership of the underlying memory
            # in the C code, so the Python garbage collector will leave
            # it alone.
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break
        fps = fps_streams["stream{0}".format(frame_meta.pad_index)].get_fps()
        frame_number = frame_meta.frame_num
        if(frame_number<0):
            if(frame_number%10==0):
                print("skip executing: {0}".format(frame_number))
            return Gst.PadProbeReturn.OK
        wtf_index=int(np.load(wtf_index_dir))
        if(frame_number%20==0):
            print("processing frame: {0}, wtf_frame: {1}".format(frame_number, frame_number+wtf_index))
        l_obj = frame_meta.obj_meta_list
        frame_meta.num_obj_meta
        obj_counter = {
            PGIE_CLASS_ID_BACKGROUND: 0,
            PGIE_CLASS_ID_FENDER: 0,
            PGIE_CLASS_ID_LADDER: 0,
        }

        # Get full frame to draw bounding boxes for each detection
        n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
        n_full_frame = np.copy(n_frame)
        n_full_frame = cv2.cvtColor(n_full_frame, cv2.COLOR_RGBA2BGRA)
        count = 0
        name = None
        while l_obj is not None:
            try:
                # Casting l_obj.data to pyds.NvDsObjectMeta
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                uniqueId = obj_meta.object_id
            except StopIteration:
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Stop Iteration 0...")
                break
            obj_counter[obj_meta.class_id] += 1

            #            osd_rect_params =  pyds.NvOSD_RectParams.cast(obj_meta.rect_params)
            # Draw black patch to cover faces (class_id = 2), can change to other colors
            obj_meta.text_params.display_text = "{0}, conf: {1:.3f}".format(
                pgie_classes_str[obj_meta.class_id], obj_meta.confidence
            )
            obj_meta.confidence

            n_frame_cropped = np.copy(n_frame)
            # Save full frame and cropped frames
            n_full_frame = draw_bounding_boxes(n_full_frame, obj_meta, obj_meta.confidence)

            n_frame_cropped = cv2.cvtColor(n_frame_cropped, cv2.COLOR_RGBA2BGRA)

            inside_center, n_frame_cropped = crop_object(n_frame_cropped, obj_meta)
            
            if inside_center and obj_meta.confidence > 0.5 and detect_if_img_is_valid(n_frame_cropped):
                name = None
                max_wtf_index=wtf_index+1
                for i in range(wtf_index, max_wtf_index):
                    name, iou = get_origin_lable_name_with_iou(obj_meta, frame_number+i)
                    if name is not None and iou >0.5 :
                        if(i!=wtf_index):
                            np.save(wtf_index_dir, i)
                            print("WTF index is: increased from {0} to {1} with iou {2}".format(wtf_index, i, iou))
                        break
                img_path = "{0}/stream_{1}/{2}/{3}".format(
                    folder_name,
                    0,
                        pgie_classes_str[obj_meta.class_id],
                    name + ".jpg"
                    if name is not None
                    else file_name_list[frame_number].replace(".jpg", "_count{0}.jpg".format(count)),
                )
                count += 1
                cv2.imwrite(img_path, n_frame_cropped)

            try:
                l_obj = l_obj.next
            except StopIteration:
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Stop Iteration 1...")
                break
        past_tracking_meta[0] = past_tracking

        if past_tracking_meta[0] == 1:
            l_user = batch_meta.batch_user_meta_list
            while l_user is not None:
                try:
                    # Note that l_user.data needs a cast to pyds.NvDsUserMeta
                    # The casting is done by pyds.NvDsUserMeta.cast()
                    # The casting also keeps ownership of the underlying memory
                    # in the C code, so the Python garbage collector will leave
                    # it alone
                    user_meta = pyds.NvDsUserMeta.cast(l_user.data)
                except StopIteration:
                    break
                if user_meta and user_meta.base_meta.meta_type == pyds.NvDsMetaType.NVDS_TRACKER_PAST_FRAME_META:
                    try:
                        # Note that user_meta.user_meta_data needs a cast to pyds.NvDsPastFrameObjBatch
                        # The casting is done by pyds.NvDsPastFrameObjBatch.cast()
                        # The casting also keeps ownership of the underlying memory
                        # in the C code, so the Python garbage collector will leave
                        # it alone
                        pyds.NvDsPastFrameObjBatch.cast(user_meta.user_meta_data)
                    except StopIteration:
                        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Stop Iteration 2...")
                        break
                    # for trackobj in pyds.NvDsPastFrameObjBatch.list(pPastFrameObjBatch):
                    #     print("streamId=",trackobj.streamID)
                    #     print("surfaceStreamID=",trackobj.surfaceStreamID)
                    #     for pastframeobj in pyds.NvDsPastFrameObjStream.list(trackobj):
                    #         print("numobj=",pastframeobj.numObj)
                    #         print("uniqueId=",pastframeobj.uniqueId)
                    #         uniqueId = pastframeobj.uniqueId
                    #         print("classId=",pastframeobj.classId)
                    #         print("objLabel=",pastframeobj.objLabel)
                    #         for objlist in pyds.NvDsPastFrameObjList.list(pastframeobj):
                    #             print('frameNum:', objlist.frameNum)
                    #             print('tBbox.left:', objlist.tBbox.left)
                    #             print('tBbox.width:', objlist.tBbox.width)
                    #             print('tBbox.top:', objlist.tBbox.top)
                    #             print('tBbox.right:', objlist.tBbox.height)
                    #             print('confidence:', objlist.confidence)
                    #             print('age:', objlist.age)
                try:
                    l_user = l_user.next
                except StopIteration:
                    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Stop Iteration 3...")
                    break

        # Save file
        if detect_if_img_is_valid(n_full_frame):
            img_path = "{0}/stream_{1}/{2}/{3}".format(
                folder_name,
                0,
                "full_frames",
                uniqueId if file_name_list is None else file_name_list[frame_number],
            )
            img_path_original_label = "{0}/stream_{1}/{2}/{3}".format(
                folder_name,
                0,
                "full_frames_orig_label",
                uniqueId if file_name_list is None else file_name_list[frame_number],
            )
            cv2.imwrite(img_path, n_full_frame)
            draw_original_label_on_full_frame(n_full_frame, frame_number)
            cv2.imwrite(img_path_original_label, n_full_frame)
        try:
            l_frame = l_frame.next
        except StopIteration:
            break
    return Gst.PadProbeReturn.OK

def draw_original_label_on_full_frame(frame, frame_number):
    original_labels, l_width, l_height = get_labels_from_kitti(
        os.path.join(label_dir, file_name_list[frame_number].replace(".jpg", ".txt")), True
    )
    for label in original_labels:
        xScale = STREAM_WIDTH/l_width
        yScale = STREAM_HEIGHT/l_height
        xmin = int(float(label.xmin)*xScale)
        xmax = int(float(label.xmax)*xScale)
        ymin = int(float(label.ymin)*yScale)
        ymax = int(float(label.ymax)*yScale)
        color = (0, 255, 0, 0)
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, thickness=2)
        cv2.putText(frame, label.name, (xmin, ymin - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, color)


def get_origin_lable_name_with_iou(obj_meta, frame_number):
    """We would like to have the same names with the detecitions than the manual labeled for ease of debug.
    Therefore we match the detection with each label, the one with the greatest IOU is used, return none if IOU<0.5
    :param return: cropped_image_name if IOU<0.5 else None"""
    original_labels, l_width, l_height = get_labels_from_kitti(
        os.path.join(label_dir, file_name_list[frame_number].replace(".jpg", ".txt")), True
    )
    max_iou = 0
    rect_params = obj_meta.rect_params
    top = int(rect_params.top)
    left = int(rect_params.left)
    width = int(rect_params.width)
    height = int(rect_params.height)
    y_pred = {
        "x1": left / STREAM_WIDTH,
        "x2": (left + width) / STREAM_WIDTH,
        "y1": top / STREAM_HEIGHT,
        "y2": (top + height) / STREAM_HEIGHT,
    }
    # print("pred: {0}".format(y_pred))
    for label in original_labels:
        y_true = {
            "x1": label.xmin / l_width,
            "x2": label.xmax / l_width,
            "y1": label.ymin / l_height,
            "y2": label.ymax / l_height,
        }
        # print("true: {0}".format(y_true))
        # print("y_pred: {0}".format(y_pred))
        # print("y_true: {0}".format(y_true))
        iou = calculate_iou(y_true, y_pred)
        if iou > max_iou:
            max_iou = iou
        if iou > 0.5:
            cropped_img_name = label.text
            # print("It's a MATCH, {1}; {0}".format(file_name_list[frame_number].replace(".jpg", ".txt"), frame_number))
            return cropped_img_name, iou
    print(
        "Failed to match image: {3}; {2}, IOU: {0:.2f}, {1}".format(
            max_iou, pgie_classes_str[obj_meta.class_id], file_name_list[frame_number].replace(".jpg", ".txt"),
            frame_number
        )
    )
    return None, 0


def calculate_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1["x1"] < bb1["x2"]
    assert bb1["y1"] < bb1["y2"]
    assert bb2["x1"] < bb2["x2"]
    assert bb2["y1"] < bb2["y2"]

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1["x1"], bb2["x1"])
    y_top = max(bb1["y1"], bb2["y1"])
    x_right = min(bb1["x2"], bb2["x2"])
    y_bottom = min(bb1["y2"], bb2["y2"])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1["x2"] - bb1["x1"]) * (bb1["y2"] - bb1["y1"])
    bb2_area = (bb2["x2"] - bb2["x1"]) * (bb2["y2"] - bb2["y1"])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def draw_bounding_boxes(image, obj_meta, confidence):
    """Draw bounding boxes, used for tracker"""
    confidence = "{0:.2f}".format(confidence)
    rect_params = obj_meta.rect_params
    top = int(rect_params.top)
    left = int(rect_params.left)
    width = int(rect_params.width)
    height = int(rect_params.height)
    obj_name = pgie_classes_str[obj_meta.class_id]
    # image = cv2.rectangle(image, (left, top), (left + width, top + height), (0, 0, 255, 0), 2, cv2.LINE_4)
    color = (0, 0, 255, 0)
    thickness=2
    w_percents = int(width * 0.05) if width > 100 else int(width * 0.1)
    h_percents = int(height * 0.05) if height > 100 else int(height * 0.1)
    linetop_c1 = (left + w_percents, top)
    linetop_c2 = (left + width - w_percents, top)
    image = cv2.line(image, linetop_c1, linetop_c2, color, thickness)
    linebot_c1 = (left + w_percents, top + height)
    linebot_c2 = (left + width - w_percents, top + height)
    image = cv2.line(image, linebot_c1, linebot_c2, color, thickness)
    lineleft_c1 = (left, top + h_percents)
    lineleft_c2 = (left, top + height - h_percents)
    image = cv2.line(image, lineleft_c1, lineleft_c2, color, thickness)
    lineright_c1 = (left + width, top + h_percents)
    lineright_c2 = (left + width, top + height - h_percents)
    image = cv2.line(image, lineright_c1, lineright_c2, color, thickness)
    # Note that on some systems cv2.putText erroneously draws horizontal lines across the image
    image = cv2.putText(
        image,
        obj_name + ",C=" + str(confidence),
        (left - 10, top - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255, 0),
        2,
    )
    return image


def crop_object(image, obj_meta, centered_percentage=0.8):
    """Crop image based on detection window, returns if inside center percentage (Bool) and cropped image"""
    rect_params = obj_meta.rect_params
    top = int(rect_params.top)
    left = int(rect_params.left)
    width = int(rect_params.width)
    height = int(rect_params.height)
    pgie_classes_str[obj_meta.class_id]

    # Check if detectin is in the center or cornder
    img_height, img_width, img_channels = image.shape
    left_limit_percentage = (1 - centered_percentage) / 2
    right_limit_percentage = 1 - left_limit_percentage

    crop_img = image[top : top + height, left : left + width]
    if left > (img_width * left_limit_percentage) and right_limit_percentage < (img_width * right_limit_percentage):
        return True, crop_img
    else:
        return False, crop_img


def cb_newpad(decodebin, decoder_src_pad, data):
    """Copied from deepstream_imagedata-multistream_redaction.py"""
    print("In cb_newpad\n")
    caps = decoder_src_pad.get_current_caps()
    gststruct = caps.get_structure(0)
    gstname = gststruct.get_name()
    source_bin = data
    features = caps.get_features(0)

    # Need to check if the pad created by the decodebin is for video and not
    # audio.
    if gstname.find("video") != -1:
        # Link the decodebin pad only if decodebin has picked nvidia
        # decoder plugin nvdec_*. We do this by checking if the pad caps contain
        # NVMM memory features.
        if features.contains("memory:NVMM"):
            # Get the source bin ghost pad
            bin_ghost_pad = source_bin.get_static_pad("src")
            if not bin_ghost_pad.set_target(decoder_src_pad):
                sys.stderr.write("Failed to link decoder src pad to source bin ghost pad\n")
        else:
            sys.stderr.write(" Error: Decodebin did not pick nvidia decoder plugin.\n")


def decodebin_child_added(child_proxy, Object, name, user_data):
    """Copied from deepstream_imagedata-multistream_redaction.py"""
    print("Decodebin child added:", name, "\n")
    if name.find("decodebin") != -1:
        Object.connect("child-added", decodebin_child_added, user_data)
    # if is_aarch64() and name.find("nvv4l2decoder") != -1:
    #    print("Seting bufapi_version\n")
    #    Object.set_property("bufapi-version", True)


def create_source_bin(index, uri):
    """Copied from deepstream_imagedata-multistream_redaction.py"""
    print("Creating source bin")

    # Create a source GstBin to abstract this bin's content from the rest of the
    # pipeline
    bin_name = "source-bin-%02d" % index
    print(bin_name)
    nbin = Gst.Bin.new(bin_name)
    if not nbin:
        sys.stderr.write(" Unable to create source bin \n")

    # Source element for reading from the uri.
    # We will use decodebin and let it figure out the container format of the
    # stream and the codec and plug the appropriate demux and decode plugins.
    uri_decode_bin = Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
    if not uri_decode_bin:
        sys.stderr.write(" Unable to create uri decode bin \n")
    # We set the input uri to the source element
    uri_decode_bin.set_property("uri", uri)
    # Connect to the "pad-added" signal of the decodebin which generates a
    # callback once a new pad for raw data has beed created by the decodebin
    uri_decode_bin.connect("pad-added", cb_newpad, nbin)
    uri_decode_bin.connect("child-added", decodebin_child_added, nbin)

    # We need to create a ghost pad for the source bin which will act as a proxy
    # for the video decoder src pad. The ghost pad will not have a target right
    # now. Once the decode bin creates the video decoder and generates the
    # cb_newpad callback, we will set the ghost pad target to the video decoder
    # src pad.
    Gst.Bin.add(nbin, uri_decode_bin)
    bin_pad = nbin.add_pad(Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC))
    if not bin_pad:
        sys.stderr.write(" Failed to add ghost pad in source bin \n")
        return None
    return nbin


def main(uri_inputs, codec, bitrate, delete_out_folder, glob_label_dir):
    """Mainly copied from deepstream_imagedata-multistream_redaction.py,
    add optional tracker support"""
    # Check input arguments
    number_sources = len(uri_inputs)
    for i in range(0, number_sources):
        fps_streams["stream{0}".format(i)] = GETFPS(i)
    global folder_name
    folder_name = "out_crops/images"
    global label_dir
    label_dir = glob_label_dir
    global wtf_index_dir
    wtf_index_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "wtf.npy")
    np.save(wtf_index_dir, 0)

    
    if delete_out_folder and os.path.exists(folder_name):
        shutil.rmtree(folder_name)
        print("Deleted output folder: {0}".format(folder_name))

    if path.exists(folder_name):
        sys.stderr.write("The output folder %s already exists. Please remove it first.\n" % folder_name)
        sys.exit(1)

    os.makedirs(folder_name)
    print("Frames will be saved in ", folder_name)
    # Standard GStreamer initialization
    GObject.threads_init()
    Gst.init(None)

    # Create gstreamer elements */
    # Create Pipeline element that will form a connection of other elements
    print("Creating Pipeline \n ")
    pipeline = Gst.Pipeline()

    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline \n")
    print("Creating streamux \n ")

    # Create nvstreammux instance to form batches from one or more sources.
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write(" Unable to create NvStreamMux \n")

    pipeline.add(streammux)
    os.mkdir(folder_name + "/stream_" + str(0))
    os.mkdir(folder_name + "/stream_" + str(0) + "/full_frames")
    os.mkdir(folder_name + "/stream_" + str(0) + "/full_frames_orig_label")
    for clas in pgie_classes_str:
        os.mkdir(folder_name + "/stream_" + str(0) + "/" + clas)
    for i in range(number_sources):
        frame_count["stream_" + str(i)] = 0
        saved_count["stream_" + str(i)] = 0
        print("Creating source_bin ", i, " \n ")
        uri_name = uri_inputs[i]
        if uri_name.find("rtsp://") == 0:
            pass
        source_bin = create_source_bin(i, uri_name)
        if not source_bin:
            sys.stderr.write("Unable to create source bin \n")
        pipeline.add(source_bin)
        padname = "sink_%u" % i
        sinkpad = streammux.get_request_pad(padname)
        if not sinkpad:
            sys.stderr.write("Unable to create sink pad bin \n")
        srcpad = source_bin.get_static_pad("src")
        if not srcpad:
            sys.stderr.write("Unable to create src pad bin \n")
        srcpad.link(sinkpad)
    print("Creating Pgie \n ")
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    if not pgie:
        sys.stderr.write(" Unable to create pgie \n")
    if tracker_enabled:
        tracker = Gst.ElementFactory.make("nvtracker", "tracker")
        if not tracker:
            sys.stderr.write(" Unable to create tracker \n")
    # Add nvvidconv1 and filter1 to convert the frames to RGBA
    # which is easier to work with in Python.
    print("Creating nvvidconv1 \n ")
    nvvidconv1 = Gst.ElementFactory.make("nvvideoconvert", "convertor1")
    if not nvvidconv1:
        sys.stderr.write(" Unable to create nvvidconv1 \n")
    print("Creating filter1 \n ")
    caps1 = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
    filter1 = Gst.ElementFactory.make("capsfilter", "filter1")
    if not filter1:
        sys.stderr.write(" Unable to get the caps filter1 \n")
    filter1.set_property("caps", caps1)
    print("Creating tiler \n ")
    tiler = Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
    if not tiler:
        sys.stderr.write(" Unable to create tiler \n")
    print("Creating nvvidconv \n ")
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    if not nvvidconv:
        sys.stderr.write(" Unable to create nvvidconv \n")
    print("Creating nvosd \n ")
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    if not nvosd:
        sys.stderr.write(" Unable to create nvosd \n")
    nvvidconv_postosd = Gst.ElementFactory.make("nvvideoconvert", "convertor_postosd")
    if not nvvidconv_postosd:
        sys.stderr.write(" Unable to create nvvidconv_postosd \n")

    # Create a caps filter
    caps = Gst.ElementFactory.make("capsfilter", "filter")
    caps.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=I420"))

    # Make the encoder
    if codec == "H264":
        encoder = Gst.ElementFactory.make("nvv4l2h264enc", "encoder")
        print("Creating H264 Encoder")
    elif codec == "H265":
        encoder = Gst.ElementFactory.make("nvv4l2h265enc", "encoder")
        print("Creating H265 Encoder")
    if not encoder:
        sys.stderr.write(" Unable to create encoder")
    encoder.set_property("bitrate", bitrate)
    if is_aarch64():
        encoder.set_property("preset-level", 1)
        encoder.set_property("insert-sps-pps", 1)
        encoder.set_property("bufapi-version", 1)

    # Make the payload-encode video into RTP packets
    if codec == "H264":
        rtppay = Gst.ElementFactory.make("rtph264pay", "rtppay")
        print("Creating H264 rtppay")
    elif codec == "H265":
        rtppay = Gst.ElementFactory.make("rtph265pay", "rtppay")
        print("Creating H265 rtppay")
    if not rtppay:
        sys.stderr.write(" Unable to create rtppay")

    # Make the UDP sink
    updsink_port_num = 5400
    sink = Gst.ElementFactory.make("udpsink", "udpsink")
    if not sink:
        sys.stderr.write(" Unable to create udpsink")

    sink.set_property("host", "224.224.255.255")
    sink.set_property("port", updsink_port_num)
    sink.set_property("async", False)
    sink.set_property("sync", 1)
    # added for mp4-out
    codecparse = Gst.ElementFactory.make("h264parse", "h264_parse")
    if not codecparse:
        sys.stderr.write(" Unable to create codecparse \n")

    file_mux = Gst.ElementFactory.make("mp4mux", "mux")
    if not file_mux:
        sys.stderr.write(" Unable to create mux \n")

    file_sink = Gst.ElementFactory.make("filesink", "filesink")
    if not file_sink:
        sys.stderr.write(" Unable to create filesink \n")
    file_sink.set_property("location", output_path)
    #################

    print("Playing file {} ".format(uri_inputs))

    streammux.set_property("width", STREAM_WIDTH)
    streammux.set_property("height", STREAM_HEIGHT)
    streammux.set_property("batch-size", number_sources)
    streammux.set_property("batched-push-timeout", 4000000)
    streammux.set_property("live-source", 0)
    pgie.set_property("config-file-path", "config/pgie_yolov4_tlt_config.txt")
    pgie_batch_size = pgie.get_property("batch-size")
    if pgie_batch_size != number_sources:
        print(
            "WARNING: Overriding infer-config batch-size",
            pgie_batch_size,
            " with number of sources ",
            number_sources,
            " \n",
        )
        pgie.set_property("batch-size", number_sources)
    tiler_rows = int(math.sqrt(number_sources))
    tiler_columns = int(math.ceil((1.0 * number_sources) / tiler_rows))
    tiler.set_property("rows", tiler_rows)
    tiler.set_property("columns", tiler_columns)
    tiler.set_property("width", TILED_OUTPUT_WIDTH)
    tiler.set_property("height", TILED_OUTPUT_HEIGHT)

    if not is_aarch64():
        # Use CUDA unified memory in the pipeline so frames
        # can be easily accessed on CPU in Python.
        mem_type = int(pyds.NVBUF_MEM_CUDA_UNIFIED)
        streammux.set_property("nvbuf-memory-type", mem_type)
        nvvidconv.set_property("nvbuf-memory-type", mem_type)
        nvvidconv1.set_property("nvbuf-memory-type", mem_type)
        tiler.set_property("nvbuf-memory-type", mem_type)

    # Set properties of tracker
    config = configparser.ConfigParser()
    if tracker_enabled:
        config.read("config/dstest2_tracker_config.txt")
        config.sections()

        for key in config["tracker"]:
            if key == "tracker-width":
                tracker_width = config.getint("tracker", key)
                tracker.set_property("tracker-width", tracker_width)
            if key == "tracker-height":
                tracker_height = config.getint("tracker", key)
                tracker.set_property("tracker-height", tracker_height)
            if key == "gpu-id":
                tracker_gpu_id = config.getint("tracker", key)
                tracker.set_property("gpu_id", tracker_gpu_id)
            if key == "ll-lib-file":
                tracker_ll_lib_file = config.get("tracker", key)
                tracker.set_property("ll-lib-file", tracker_ll_lib_file)
            if key == "ll-config-file":
                tracker_ll_config_file = config.get("tracker", key)
                tracker.set_property("ll-config-file", tracker_ll_config_file)
            if key == "enable-batch-process":
                tracker_enable_batch_process = config.getint("tracker", key)
                tracker.set_property("enable_batch_process", tracker_enable_batch_process)
            if key == "enable-past-frame":
                tracker_enable_past_frame = config.getint("tracker", key)
                tracker.set_property("enable_past_frame", tracker_enable_past_frame)

    print("Adding elements to Pipeline \n")
    pipeline.add(pgie)
    if tracker_enabled:
        pipeline.add(tracker)
    pipeline.add(tiler)
    pipeline.add(nvvidconv)
    pipeline.add(filter1)
    pipeline.add(nvvidconv1)
    pipeline.add(nvosd)
    pipeline.add(nvvidconv_postosd)
    pipeline.add(caps)
    pipeline.add(encoder)
    pipeline.add(rtppay)
    # added for mp4 out
    pipeline.add(codecparse)
    pipeline.add(file_mux)
    pipeline.add(file_sink)
    ###############
    # pipeline.add(sink)

    print("Linking elements in the Pipeline \n")
    streammux.link(pgie)
    if tracker_enabled:
        pgie.link(tracker)
        tracker.link(nvvidconv1)
    else:
        pgie.link(nvvidconv1)
    nvvidconv1.link(filter1)
    filter1.link(tiler)
    tiler.link(nvvidconv)
    nvvidconv.link(nvosd)
    nvosd.link(nvvidconv_postosd)
    nvvidconv_postosd.link(caps)
    caps.link(encoder)
    # for mp4
    encoder.link(codecparse)
    codecparse.link(file_mux)
    file_mux.link(file_sink)
    #
    # encoder.link(rtppay)
    # rtppay.link(sink)

    # create an event loop and feed gstreamer bus mesages to it
    loop = GObject.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    # Start streaming
    rtsp_port_num = 8554

    server = GstRtspServer.RTSPServer.new()
    server.props.service = "%d" % rtsp_port_num
    server.attach(None)

    factory = GstRtspServer.RTSPMediaFactory.new()
    factory.set_launch(
        '( udpsrc name=pay0 port=%d buffer-size=524288 caps="application/x-rtp, media=video, clock-rate=90000, encoding-name=(string)%s, payload=96 " )'
        % (updsink_port_num, codec)
    )
    factory.set_shared(True)
    server.get_mount_points().add_factory("/ds-test", factory)

    print("\n *** DeepStream: Launched RTSP Streaming at rtsp://localhost:%d/ds-test ***\n\n" % rtsp_port_num)

    tiler_sink_pad = tiler.get_static_pad("sink")
    if not tiler_sink_pad:
        sys.stderr.write(" Unable to get sink pad \n")
    else:
        tiler_sink_pad.add_probe(Gst.PadProbeType.BUFFER, tiler_sink_pad_buffer_probe, 0)

    print("Starting pipeline \n")
    # start play back and listed to events
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass
    # cleanup
    print("Exiting app\n")
    pipeline.set_state(Gst.State.NULL)


def parse_args():
    """Parse arguments from argparser
    :param return: parsed args"""
    parser = argparse.ArgumentParser(description="RTSP Output Sample Application Help ")

    parser.add_argument(
        "-i",
        "--uri_inputs",
        metavar="N",
        type=str,
        nargs="+",
        help="Path to inputs URI e.g. rtsp:// ...  or file:// seperated by space",
    )

    parser.add_argument(
        "-c", "--codec", default="H264", help="RTSP Streaming Codec H264/H265 , default=H264", choices=["H264", "H265"]
    )
    parser.add_argument("-b", "--bitrate", default=2728936, help="Set the encoding bitrate ", type=int)
    parser.add_argument(
        "-o", "--output", default="/dli/task/my_apps/images_broken.mp4", help="Set the output file path "
    )

    parser.add_argument("-m", "--meta", default=0, help="set past tracking meta ", type=int)
    parser.add_argument("-d", help="delete output folder before processing", action="store_true")
    parser.add_argument("-t", "--enable_tracker")
    parser.add_argument(
        "-n",
        "--file_name_list",
        help="Csv lists of filenames, has to match the ffmpeg video, used for pretty debug file naming",
    )
    parser.add_argument("-l", "--label_dir", help="Labeled file list", required=True)
    # Check input arguments
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()

    print("URI Inputs: " + str(args.uri_inputs))
    global output_path
    global past_tracking
    global tracker_enabled
    global file_name_list
    file_name_list = []
    if args.file_name_list is not None:
        file_name_list = np.genfromtxt(args.file_name_list, delimiter=",", dtype=str)
    tracker_enabled = args.enable_tracker
    if tracker_enabled:
        print("Tracker enabled")
    else:
        print("Tracker disabled")
    output_path = args.output
    past_tracking = args.meta

    return args.uri_inputs, args.codec, args.bitrate, args.d, args.label_dir


if __name__ == "__main__":
    uri_inputs, out_codec, out_bitrate, delete_out_folder, label_dir = parse_args()
    sys.exit(main(uri_inputs, out_codec, out_bitrate, delete_out_folder, label_dir))


# Failed to match image: portai20211215111931_IMG_0001_copy_22.txt, IOU: 0.39, Ladder
# Failed to match image: portai20211215111931_IMG_0024_copy_10.txt, IOU: 0.00, Fender
# Failed to match image: portai20211215111931_IMG_0031_copy_11.txt, IOU: 0.00, Fender
# Failed to match image: portai20211215111931_IMG_0036_copy_10.txt, IOU: 0.00, Fender
# Failed to match image: portai20211215111931_IMG_0041_copy_10.txt, IOU: 0.00, Fender
# Failed to match image: portai20211215111931_IMG_0042_copy_10.txt, IOU: 0.00, Fender
# Failed to match image: portai20211215111931_IMG_0050_copy_2.txt, IOU: 0.50, Ladder
# Failed to match image: portai20211215111931_IMG_0051_copy_2.txt, IOU: 0.44, Ladder
# Failed to match image: portai20211215111931_IMG_0067_copy_9.txt, IOU: 0.30, Fender
# Failed to match image: portai20211215111931_IMG_0109_copy_3.txt, IOU: 0.00, Fender
# Failed to match image: portai20211215111931_IMG_0115_copy_3.txt, IOU: 0.00, Fender
# Failed to match image: portai20211215111931_IMG_0117_copy_11.txt, IOU: 0.00, Ladder
# Failed to match image: portai20211215111931_IMG_0147_copy_8.txt, IOU: 0.00, Fender
# Failed to match image: portai20211215111931_IMG_0174_copy_16.txt, IOU: 0.00, Fender
# Failed to match image: portai20211215111931_IMG_0180_2.txt, IOU: 0.00, Fender
# Failed to match image: portai20211215111931_IMG_0180_copy_16.txt, IOU: 0.00, Fender
# Failed to match image: portai20211215111931_IMG_0231_copy.txt, IOU: 0.00, Ladder
# Failed to match image: portai20211215111931_IMG_0266_copy_8.txt, IOU: 0.47, Ladder
# Failed to match image: portai20211215111931_IMG_0280_copy_4.txt, IOU: 0.00, Ladder
# Failed to match image: portai20211215111931_IMG_0285_copy_10.txt, IOU: 0.00, Fender
# Failed to match image: portai20211215111931_IMG_0287_copy_8.txt, IOU: 0.00, Fender
# Failed to match image: portai20211215111931_IMG_0350_copy_11.txt, IOU: 0.00, Fender
# Failed to match image: portai20211215111931_IMG_0351_copy_12.txt, IOU: 0.00, Fender
# Failed to match image: portai20211215111931_IMG_0364.txt, IOU: 0.47, Fender
# Failed to match image: portai20211215111931_IMG_0365.txt, IOU: 0.49, Fender
# Failed to match image: portai20211215111931_IMG_0373.txt, IOU: 0.45, Fender
# Failed to match image: portai20211215111931_IMG_0391_copy_4.txt, IOU: 0.50, Fender
# Failed to match image: portai20211215111931_IMG_0395.txt, IOU: 0.37, Fender
# Failed to match image: portai20211215111931_IMG_0395_copy_4.txt, IOU: 0.00, Fender
# Failed to match image: portai20211215111931_IMG_0395_copy_4.txt, IOU: 0.00, Fender
# Failed to match image: portai20211215111931_IMG_0409.txt, IOU: 0.46, Fender
# Failed to match image: portai20211215111931_IMG_0410_copy_3.txt, IOU: 0.42, Fender
# Failed to match image: portai20211215111931_IMG_0412_copy_3.txt, IOU: 0.45, Fender
# Failed to match image: portai20211215111931_IMG_0412_copy_3.txt, IOU: 0.42, Fender
# Failed to match image: portai20211215111931_IMG_0413.txt, IOU: 0.50, Fender
# Failed to match image: portai20211215111931_IMG_0417_copy_8.txt, IOU: 0.45, Fender
# Failed to match image: portai20211215111931_IMG_0422_copy_3.txt, IOU: 0.50, Fender
# Failed to match image: portai20211215111931_IMG_0423_copy_3.txt, IOU: 0.46, Fender
# Failed to match image: portai20211215111931_IMG_0424_copy_3.txt, IOU: 0.46, Fender
# Failed to match image: portai20211215111931_IMG_0427_copy_3.txt, IOU: 0.43, Fender
# Failed to match image: portai20211215111931_IMG_0430_copy_3.txt, IOU: 0.45, Fender
# Failed to match image: portai20211215111931_IMG_0435_copy_3.txt, IOU: 0.50, Fender
# Failed to match image: portai20211215111931_IMG_0447_copy_7.txt, IOU: 0.41, Ladder
# Failed to match image: portai20211215111931_IMG_0494_copy.txt, IOU: 0.49, Ladder
# Failed to match image: portai20211215111931_IMG_0531_copy.txt, IOU: 0.38, Ladder
# Failed to match image: portai20211215111931_IMG_0536_copy_5.txt, IOU: 0.00, Ladder
# Failed to match image: portai20211215111931_IMG_0537_copy_5.txt, IOU: 0.00, Ladder
# Failed to match image: portai20211215111931_IMG_0538_copy_5.txt, IOU: 0.00, Ladder
# Failed to match image: portai20211215111931_IMG_0543_copy_5.txt, IOU: 0.00, Fender
# Failed to match image: portai20211215111931_IMG_0577.txt, IOU: 0.00, Ladder
# Failed to match image: portai20211215111931_IMG_0584_copy.txt, IOU: 0.00, Fender
# Failed to match image: portai20211215111931_IMG_0584_copy.txt, IOU: 0.01, Fender
# Failed to match image: portai20211215111931_IMG_0592.txt, IOU: 0.50, Fender
# Failed to match image: portai20211215111931_IMG_0596.txt, IOU: 0.50, Fender
# Failed to match image: portai20211215111931_IMG_0604.txt, IOU: 0.45, Fender
# Failed to match image: portai20211215111931_IMG_0605.txt, IOU: 0.49, Fender
# Failed to match image: portai20211215111931_IMG_0607.txt, IOU: 0.49, Fender
# Failed to match image: portai20211215111931_IMG_0607.txt, IOU: 0.47, Fender
# Failed to match image: portai20211215111931_IMG_0614.txt, IOU: 0.49, Fender
# Failed to match image: portai20211215111931_IMG_0615.txt, IOU: 0.47, Fender
# Failed to match image: portai20211215111931_IMG_0619.txt, IOU: 0.41, Fender
# Failed to match image: portai20211215111931_IMG_0619_copy_2.txt, IOU: 0.49, Ladder
# Failed to match image: portai20211215111931_IMG_0622.txt, IOU: 0.48, Fender
# Failed to match image: portai20211215111931_IMG_0622_copy_2.txt, IOU: 0.46, Ladder
# Failed to match image: portai20211215111931_IMG_0625.txt, IOU: 0.50, Fender
# Failed to match image: portai20211215111931_IMG_0644_copy.txt, IOU: 0.00, Fender
# Failed to match image: portai20211215111931_IMG_0648_copy_3.txt, IOU: 0.00, Ladder
# Failed to match image: portai20211215111931_IMG_0649_copy.txt, IOU: 0.00, Fender
# Failed to match image: portai20211215111931_IMG_0660_copy.txt, IOU: 0.45, Fender
# Failed to match image: portai20211215111931_IMG_0668_copy.txt, IOU: 0.48, Fender
