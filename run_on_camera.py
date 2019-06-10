from __future__ import division
import sys
import os

import csv
import time
import numpy as np
import cv2
import tensorflow as tf

from PIL import Image
import src.siamese as siam
from src.tracker import tracker
from src.parse_arguments import parse_arguments
from src.region_to_bbox import region_to_bbox
from src.visualization import show_frame, show_crops, show_scores

import matplotlib.pyplot as plt

roi_list = []
refPt = []


def _update_target_position(pos_x, pos_y, score, final_score_sz, tot_stride, search_sz, response_up, x_sz):
    # find location of score maximizer
    p = np.asarray(np.unravel_index(np.argmax(score), np.shape(score)))
    # displacement from the center in search area final representation ...
    center = float(final_score_sz - 1) / 2
    disp_in_area = p - center
    # displacement from the center in instance crop
    disp_in_xcrop = disp_in_area * float(tot_stride) / response_up
    # displacement from the center in instance crop (in frame coordinates)
    disp_in_frame = disp_in_xcrop * x_sz / search_sz
    # *position* within frame in frame coordinates
    pos_y, pos_x = pos_y + disp_in_frame[0], pos_x + disp_in_frame[1]
    return pos_x, pos_y


def convert_roi(x, y):
    # x, y are left top and right lower corner
    w = y[0] - x[0]
    h = y[1] - x[1]
    cx = x[0] + w / 2.0
    cy = x[1] + h / 2.0

    return cx, cy, w, h


def draw_roi(event, x, y, flags, param):
    # grab references to the global variables
    global refPt
    global roi_list

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]

    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        refPt.append((x, y))

        # draw a rectangle around the region of interest
        clone = param[0]
        cv2.rectangle(clone, refPt[0], refPt[1], (0, 255, 0), 2)
        roi_list.append(refPt)
        cv2.imshow(param[1], clone)


def get_roi(img, w_text="N/A"):
    imgcl = img.copy()
    global roi_list
    cv2.namedWindow("Set Regions for " + w_text)
    cv2.setMouseCallback("Set Regions for " + w_text, draw_roi, [imgcl, "Set Regions for " + w_text])

    roi_list = []
    while True:
        # display the image and wait for a keypress
        cv2.imshow("Set Regions for " + w_text, imgcl)
        key = cv2.waitKey(1) & 0xFF

        # if the 'r' key is pressed, reset the cropping region
        if key == ord("n"):
            roi_list.append(refPt)
            refPt = []
            print(roi_list)

        # if the 'c' key is pressed, break from the loop
        if key == ord("c"):
            cv2.destroyAllWindows()
            return roi_list


# def main():
#     # avoid printing TF debugging information
#     os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#     # TODO: allow parameters from command line or leave everything in json files?
#     hp, evaluation, run, env, design = parse_arguments()
#     # Set size for use with tf.image.resize_images with align_corners=True.
#     # For example,
#     #   [1 4 7] =>   [1 2 3 4 5 6 7]    (length 3*(3-1)+1)
#     # instead of
#     # [1 4 7] => [1 1 2 3 4 5 6 7 7]  (length 3*3)
#     final_score_sz = hp.response_up * (design.score_sz - 1) + 1
#     # build TF graph once for all
#     filename, image, templates_z, scores = siam.build_tracking_graph(final_score_sz, design, env)

#     gt, frame_name_list, _, _ = _init_video(env, evaluation, evaluation.video)
#     pos_x, pos_y, target_w, target_h = region_to_bbox(gt[evaluation.start_frame])
#     bboxes, speed = tracker(hp, run, design, frame_name_list, pos_x, pos_y, target_w, target_h, final_score_sz,
#                             filename, image, templates_z, scores, evaluation.start_frame)
#     _, precision, precision_auc, iou = _compile_results(gt, bboxes, evaluation.dist_threshold)
#     print evaluation.video + \
#           ' -- Precision ' + "(%d px)" % evaluation.dist_threshold + ': ' + "%.2f" % precision +\
#           ' -- Precision AUC: ' + "%.2f" % precision_auc + \
#           ' -- IOU: ' + "%.2f" % iou + \
#           ' -- Speed: ' + "%.2f" % speed + ' --'
#     print


def main_camera():
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        exit()



    bboxes = np.zeros((10, 4))

    # avoid printing TF debugging information
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # TODO: allow parameters from command line or leave everything in json files?
    hp, evaluation, run, env, design = parse_arguments()
    # Set size for use with tf.image.resize_images with align_corners=True.
    # For example,
    #   [1 4 7] =>   [1 2 3 4 5 6 7]    (length 3*(3-1)+1)
    # instead of
    # [1 4 7] => [1 1 2 3 4 5 6 7 7]  (length 3*3)
    final_score_sz = hp.response_up * (design.score_sz - 1) + 1
    # build TF graph once for all
    image, templates_z, scores = siam.build_tracking_graph_cam(final_score_sz, design, env)

    ret, frame = cam.read()
    print(frame.dtype)
    roi = get_roi(frame)
    pos_x, pos_y, target_w, target_h = convert_roi(roi[0][0], roi[0][1])
    # pos_x, pos_y, target_w, target_h = region_to_bbox(gt[evaluation.start_frame])

    scale_factors = hp.scale_step ** np.linspace(-np.ceil(hp.scale_num / 2), np.ceil(hp.scale_num / 2), hp.scale_num)
    # cosine window to penalize large displacements    
    hann_1d = np.expand_dims(np.hanning(final_score_sz), axis=0)
    penalty = np.transpose(hann_1d) * hann_1d
    penalty = penalty / np.sum(penalty)

    context = design.context * (target_w + target_h)
    z_sz = np.sqrt(np.prod((target_w + context) * (target_h + context)))
    x_sz = float(design.search_sz) / design.exemplar_sz * z_sz

    # thresholds to saturate patches shrinking/growing
    min_z = hp.scale_min * z_sz
    max_z = hp.scale_max * z_sz
    min_x = hp.scale_min * x_sz
    max_x = hp.scale_max * x_sz

    run_opts = {}

    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        tf.global_variables_initializer().run()
        # Coordinate the loading of image files.
        # coord = tf.train.Coordinator()
        # threads = tf.train.start_queue_runners(coord=coord)

        # save first frame position (from ground-truth)
        bboxes[0, :] = pos_x - target_w / 2, pos_y - target_h / 2, target_w, target_h

        # TODO: convert roi[0] to the silly siam format
        image_, templates_z_ = sess.run([image, templates_z], feed_dict={
            siam.pos_x_ph: pos_x,
            siam.pos_y_ph: pos_y,
            siam.z_sz_ph: z_sz,
            image: frame})
        new_templates_z_ = templates_z_

        t_start = time.time()

        # Get an image from the queue
        while True:
            scaled_exemplar = z_sz * scale_factors
            scaled_search_area = x_sz * scale_factors
            scaled_target_w = target_w * scale_factors
            scaled_target_h = target_h * scale_factors
            ret, frame = cam.read()
            image_, scores_ = sess.run(
                [image, scores],
                feed_dict={
                    siam.pos_x_ph: pos_x,
                    siam.pos_y_ph: pos_y,
                    siam.x_sz0_ph: scaled_search_area[0],
                    siam.x_sz1_ph: scaled_search_area[1],
                    siam.x_sz2_ph: scaled_search_area[2],
                    templates_z: np.squeeze(templates_z_),
                    image: frame,
                }, **run_opts)
            scores_ = np.squeeze(scores_)
            # penalize change of scale
            scores_[0, :, :] = hp.scale_penalty * scores_[0, :, :]
            scores_[2, :, :] = hp.scale_penalty * scores_[2, :, :]
            # find scale with highest peak (after penalty)
            new_scale_id = np.argmax(np.amax(scores_, axis=(1, 2)))
            # update scaled sizes
            x_sz = (1 - hp.scale_lr) * x_sz + hp.scale_lr * scaled_search_area[new_scale_id]
            target_w = (1 - hp.scale_lr) * target_w + hp.scale_lr * scaled_target_w[new_scale_id]
            target_h = (1 - hp.scale_lr) * target_h + hp.scale_lr * scaled_target_h[new_scale_id]
            # select response with new_scale_id
            score_ = scores_[new_scale_id, :, :]
            score_ = score_ - np.min(score_)
            score_ = score_ / np.sum(score_)
            # apply displacement penalty
            score_ = (1 - hp.window_influence) * score_ + hp.window_influence * penalty
            pos_x, pos_y = _update_target_position(pos_x, pos_y, score_, final_score_sz, design.tot_stride,
                                                   design.search_sz, hp.response_up, x_sz)
            # convert <cx,cy,w,h> to <x,y,w,h> and save output
            out = pos_x - target_w / 2, pos_y - target_h / 2, target_w, target_h
            # out = pos_x-target_w/2, pos_y-target_h/2, target_w, target_h
            # update the target representation with a rolling average

            if hp.z_lr > 0:
                new_templates_z_ = sess.run([templates_z], feed_dict={
                    siam.pos_x_ph: pos_x,
                    siam.pos_y_ph: pos_y,
                    siam.z_sz_ph: z_sz,
                    image: image_
                })

                templates_z_ = (1 - hp.z_lr) * np.asarray(templates_z_) + hp.z_lr * np.asarray(new_templates_z_)

            # update template patch size
            z_sz = (1 - hp.scale_lr) * z_sz + hp.scale_lr * scaled_exemplar[new_scale_id]

            if run.visualization:
                show_frame(image_, out, 1)

        t_elapsed = time.time() - t_start
        speed = num_frames / t_elapsed

        # Finish off the filename queue coordinator.
        # coord.request_stop()
        # coord.join(threads) 

        # from tensorflow.python.client import timeline
        # trace = timeline.Timeline(step_stats=run_metadata.step_stats)
        # trace_file = open('timeline-search.ctf.json', 'w')
        # trace_file.write(trace.generate_chrome_trace_format())

    plt.close('all')


if __name__ == '__main__':
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        print("test")
    sys.exit(main_camera())
