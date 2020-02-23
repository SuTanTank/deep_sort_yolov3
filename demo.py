#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import os
import sys
import warnings

import cv2
import numpy as np
from PIL import Image

from deep_sort import nn_matching
from deep_sort import preprocessing
from deep_sort import visualization
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from yolo import YOLO

warnings.filterwarnings('ignore')


def main(video_file, yolo=None, det_txt=None):
    video_path, video_filename = os.path.split(video_file)
    video_name, video_ext = os.path.splitext(video_filename)

    # Definition of the parameters
    max_cosine_distance = 0.2
    nn_budget = 100
    nms_max_overlap_car = 0.9
    nms_max_overlap_person = 0.99

    # deep_sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename,batch_size=32)
    cat_id = {'person': 1, 'vehicle': 2}
    
    metric_person = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker_person = Tracker(metric_person, max_age=100, cat=cat_id['person'])
    metric_car = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker_car = Tracker(metric_car, max_age=100, cat=cat_id['vehicle'])

    display = True

    det_list = []

    if det_txt:
        det = np.loadtxt(det_txt, delimiter=',')
        frame_indices = det[:, 0].astype(np.int)
        min_frame_idx = frame_indices.astype(np.int).min()
        max_frame_idx = frame_indices.astype(np.int).max()
        for frame_idx in range(min_frame_idx, max_frame_idx + 1):
            rows = det[frame_indices == frame_idx]
            person = rows[rows[:, 1] == 1][:, 2:7]
            vehicle = rows[rows[:, 1] == 2][:, 2:7]
            # person = [x[2:6] for x in rows if x[1] == 1]
            # vehicle = [x[2:6] for x in rows if x[1] == 2]
            det_list.append({'person': person, 'vehicle': vehicle})

    def load_detections(frame_index, cat='person'):
        ret = det_list[frame_index][cat]
        return ret[:, :4].tolist(), ret[:, -1].tolist()

    def frame_callback(vis, frame_index):
        ret, frame = video_capture.read()
        if not ret:
            return
        # t1 = time.time()

        image = Image.fromarray(frame)

        # person
        if yolo:
            boxes_person, scores_person = yolo.detect_image(image, 'person')
        else:
            boxes_person, scores_person = load_detections(frame_index, 'person')
        # print("box_num",len(boxs))
        features_person = encoder(frame, boxes_person)

        detections_person = [Detection(bbox, score, feature) for bbox, score, feature in zip(boxes_person, scores_person, features_person)]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections_person])
        scores = np.array([d.confidence for d in detections_person])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap_person, scores)
        detections_person = [detections_person[i] for i in indices]

        # vehicle
        if yolo:
            boxes_car, scores_car = yolo.detect_image(image, 'vehicle')
        else:
            boxes_car, scores_car = load_detections(frame_index, 'vehicle')
        # print("box_num",len(boxs))
        features_car = encoder(frame, boxes_car)

        detections_car = [Detection(bbox, score, feature) for bbox, score, feature in zip(boxes_car, scores_car, features_car)]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections_car])
        scores = np.array([d.confidence for d in detections_car])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap_car, scores)
        detections_car = [detections_car[i] for i in indices]

        # Call the tracker
        tracker_person.fov = np.shape(frame)[:2]

        tracker_person.predict()
        tracker_person.update(detections_person, encoder, frame)

        tracker_car.fov = np.shape(frame)[:2]

        tracker_car.predict()
        tracker_car.update(detections_car, encoder, frame)

        if display:
            vis.set_image(frame.copy())
            vis.draw_trackers(tracker_person.tracks, cat=1)
            vis.draw_trackers(tracker_car.tracks, cat=2)
            vis.draw_detections(detections_person)
            vis.draw_detections(detections_car)
            out.write(vis.viewer.image)
    
    video_capture = cv2.VideoCapture(video_file)

    # Define the codec and create VideoWriter object
    w = int(video_capture.get(3))
    h = int(video_capture.get(4))
    image_size = (h, w)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    # fourcc = -1
    out = cv2.VideoWriter(video_name + '-output-person-car.avi', fourcc, 15, (w, h))
        
    # fps = 0.0
    seq_info = {
        "sequence_name" : video_name,
        "image_size": image_size,
        "min_frame_idx": int(video_capture.get(1)),
        "max_frame_idx": int(video_capture.get(7))
    }
    if display:
        visualizer = visualization.Visualization(seq_info, update_ms=5)    
    else:
        visualizer = visualization.NoVisualization(seq_info)
    visualizer.run(frame_callback)

    video_capture.release()
    if display:
        out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    print(sys.argv)
    if len(sys.argv) == 2:
        main(sys.argv[1], yolo=YOLO())
    elif len(sys.argv) == 3:
        main(sys.argv[1], det_txt=sys.argv[2])
    else:
        print("usage: python demo.py INPUTVIDEO [detection.txt]")
