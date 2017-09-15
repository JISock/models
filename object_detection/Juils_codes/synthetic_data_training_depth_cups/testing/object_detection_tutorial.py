import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
from scipy.misc import imread, imsave
import pickle
import ruamel.yaml as yaml

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

sys.path.append("../../..")

from utils import label_map_util

from utils import visualization_utils as vis_util


# # Model preparation

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = '../training/faster_rcnn_inception_resnet_v2_atrous_synthetic/frozen_inference_graph.pb'
# PATH_TO_CKPT = '/home/juil/workspace/tensorflow_object_detection/object_detection/models/model/train/inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = '../../../data/synthetic_label_map_cup.pbtxt'

print(PATH_TO_LABELS)

NUM_CLASSES = 1

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# ## Helper code

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


# # Detection

# PATH_TO_TEST_IMAGES_DIR = '/home/juil/Downloads/training_scene_generator_20170901/sixd_toolkit-master/output/render'
# PATH_TO_TEST_IMAGES_DIR = '/home/juil/workspace/6DOF-datasets/doumanoglou/test/01'
# # img_path = PATH_TO_TEST_IMAGES_DIR+'/coffee_{:03d}/depth/{:04d}.png'
# img_path = PATH_TO_TEST_IMAGES_DIR+'/depth/{:04d}.png'
# gt_path = PATH_TO_TEST_IMAGES_DIR+'/gt.yml'
#
# Detection_results_and_GT = list()
# idx = 0
# with detection_graph.as_default():
#     with tf.Session(graph=detection_graph) as sess:
#         with open(gt_path, 'r') as f:
#             gt = yaml.load(f, Loader=yaml.CLoader)
#         for img_idx in range(1, 50):
#             print(idx)
#             # image = Image.open(img_path.format(img_idx))
#             img = imread(img_path.format(img_idx))
#             height = img.shape[0]
#             width = img.shape[1]
#             output_im = np.zeros((height, width, 3))
#             output_im[:, :, 0] = img
#             output_im[:, :, 1] = img
#             output_im[:, :, 2] = img
#             imsave('temp.png', output_im)
#             image = Image.open('temp.png')
#             image_np = load_image_into_numpy_array(image)
#             rgb_image_np = load_image_into_numpy_array(Image.open('/home/juil/workspace/6DOF-datasets/doumanoglou/test/01/rgb/{:04d}.png'
# .format(img_idx)))
#             rgb_image_np2 = load_image_into_numpy_array(
#                 Image.open('/home/juil/workspace/6DOF-datasets/doumanoglou/test/01/rgb/{:04d}.png'
#                            .format(img_idx)))
#             # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
#             image_np_expanded = np.expand_dims(image_np, axis=0)
#             image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
#             # Each box represents a part of the image where a particular object was detected.
#             boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
#             # Each score represent how level of confidence for each of the objects.
#             # Score is shown on the result image, together with the class label.
#             scores = detection_graph.get_tensor_by_name('detection_scores:0')
#             classes = detection_graph.get_tensor_by_name('detection_classes:0')
#             num_detections = detection_graph.get_tensor_by_name('num_detections:0')
#             # Actual detection.
#             (boxes, scores, classes, num_detections) = sess.run(
#                 [boxes, scores, classes, num_detections],
#                 feed_dict={image_tensor: image_np_expanded})
#             vis_util.visualize_boxes_and_labels_on_image_array(
#                 image_np,
#                 np.squeeze(boxes),
#                 np.squeeze(classes).astype(np.int32),
#                 np.squeeze(scores),
#                 category_index,
#                 use_normalized_coordinates=True,
#                 line_thickness=4,
#                 max_boxes_to_draw=100,
#                 min_score_thresh=.1
#             )
#             im_width, im_height = image_np.shape[0:2]
#             Scaled_boxes = np.zeros([int(num_detections[0]), 4])
#             Scaled_boxes[:, 0] = boxes[0, 0:int(num_detections[0]), 0] * im_width
#             Scaled_boxes[:, 1] = boxes[0, 0:int(num_detections[0]), 1] * im_height
#             Scaled_boxes[:, 2] = boxes[0, 0:int(num_detections[0]), 2] * im_width
#             Scaled_boxes[:, 3] = boxes[0, 0:int(num_detections[0]), 3] * im_height
#
#             # for i in range(0, 2):
#             #     Scaled_boxes[:, i] = boxes[0, 0:int(num_detections[0]), i] * im_width
#             # for i in range(2, 4):
#             #     Scaled_boxes[:, i] = boxes[0, 0:int(num_detections[0]), i] * im_height
#             Scaled_scores = np.zeros([int(num_detections[0]), ])
#             Scaled_scores[:] = scores[0][0:int(num_detections[0])]
#             # ---------------------------------------------------------------------------groundtruth
#             GroundTruth = np.zeros([len(gt[img_idx]), 4])
#             for c, obj in enumerate(gt[img_idx], 0):
#                 GroundTruth[c, 0] = float(obj['obj_bb'][1])
#                 GroundTruth[c, 1] = float(obj['obj_bb'][0])
#                 GroundTruth[c, 2] = float(obj['obj_bb'][1] + obj['obj_bb'][3])
#                 GroundTruth[c, 3] = float(obj['obj_bb'][0] + obj['obj_bb'][2])
#
#             image_np2 = load_image_into_numpy_array(image)
#             boxes2 = boxes
#             boxes2.fill(0)
#             scores2 = scores
#             scores2.fill(0)
#             classes2 = classes
#             for c, obj in enumerate(gt[img_idx], 0):
#                 # boxes[0, c, 0:2] = np.divide(map(float,obj['obj_bb'][0:2]),im_width)
#                 # boxes[0, c, 2:4] = np.divide(map(float,obj['obj_bb'][0:2])+map(float,obj['obj_bb'][4:-3:-1]),im_height)
#                 # ymin
#                 boxes2[0, c, 0] = np.divide(float(obj['obj_bb'][1]), im_width)
#                 # xmin
#                 boxes2[0, c, 1] = np.divide(float(obj['obj_bb'][0]), im_height)
#                 # ymax
#                 boxes2[0, c, 2] = np.divide(float(obj['obj_bb'][1] + obj['obj_bb'][3]), im_width)
#                 # xmax
#                 boxes2[0, c, 3] = np.divide(float(obj['obj_bb'][0] + obj['obj_bb'][2]), im_height)
#                 scores2[0][c] = 1.0
#             vis_util.visualize_boxes_and_labels_on_image_array(
#                 image_np2,
#                 np.squeeze(boxes2),
#                 np.squeeze(classes2).astype(np.int32),
#                 np.squeeze(scores2),
#                 category_index,
#                 use_normalized_coordinates=True,
#                 line_thickness=4,
#                 max_boxes_to_draw=100,
#                 min_score_thresh=.4
#             )
#
#             Detection_results_and_GT.append({'Number_of_detection': num_detections, 'detected_boxes': Scaled_boxes,
#                                              'detected_scores': Scaled_scores, 'GroundTruth': GroundTruth})
#
#             plt.imsave(fname='/home/juil/Downloads/synthetic_data_analysis/analysis/detection_result/real/rgb_detection_real_{}.png'.format(idx),arr=rgb_image_np)
#             plt.imsave(fname='/home/juil/Downloads/synthetic_data_analysis/analysis/detection_result/synthetic/rgb_gt_real_{}.png'.format(idx), arr=rgb_image_np2)
#             idx += 1
#
#
# with open('Detection_results_and_GT_real_data_46983.pkl', 'wb') as handle:
#     pickle.dump(Detection_results_and_GT, handle)
#
#
#




# PATH_TO_TEST_IMAGES_DIR = '/home/juil/Downloads/training_scene_generator_20170901/sixd_toolkit-master/output/render'
# # PATH_TO_TEST_IMAGES_DIR = '/home/juil/workspace/6DOF-datasets/doumanoglou/test/01'
# img_path = PATH_TO_TEST_IMAGES_DIR+'/coffee_{:03d}/depth/{:04d}.png'
# # img_path = PATH_TO_TEST_IMAGES_DIR + '/depth/{:04d}.png'
# gt_path = PATH_TO_TEST_IMAGES_DIR + '/coffee_{:03d}/gt.yml'
# idx=0
# Detection_results_and_GT = list()
# with detection_graph.as_default():
#     with tf.Session(graph=detection_graph) as sess:
#         for scene in range(4000,4003):
#             with open(gt_path.format(scene), 'r') as f:
#                 gt = yaml.load(f, Loader=yaml.CLoader)
#             for img_idx in range(1,72):
#                 print(idx)
#                 img =  imread(img_path.format(scene,img_idx))
#                 height = img.shape[0]
#                 width = img.shape[1]
#                 output_im = np.zeros((height, width, 3))
#                 output_im[:, :, 0] = img
#                 output_im[:, :, 1] = img
#                 output_im[:, :, 2] = img
#                 imsave('temp.png', output_im)
#                 image = Image.open('temp.png')
#                 image_np = load_image_into_numpy_array(image)
#                 rgb_image_np = load_image_into_numpy_array(Image.open(PATH_TO_TEST_IMAGES_DIR+'/coffee_{:03d}/rgb/{:04d}.png'.format(scene,img_idx)))
#                 rgb_image_np2 = load_image_into_numpy_array(Image.open(PATH_TO_TEST_IMAGES_DIR+'/coffee_{:03d}/rgb/{:04d}.png'.format(scene,img_idx)))
#                 # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
#                 image_np_expanded = np.expand_dims(image_np, axis=0)
#                 image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
#                 # Each box represents a part of the image where a particular object was detected.
#                 boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
#                 # Each score represent how level of confidence for each of the objects.
#                 # Score is shown on the result image, together with the class label.
#                 scores = detection_graph.get_tensor_by_name('detection_scores:0')
#                 classes = detection_graph.get_tensor_by_name('detection_classes:0')
#                 num_detections = detection_graph.get_tensor_by_name('num_detections:0')
#                 # Actual detection.
#                 (boxes, scores, classes, num_detections) = sess.run(
#                     [boxes, scores, classes, num_detections],
#                     feed_dict={image_tensor: image_np_expanded})
#                 vis_util.visualize_boxes_and_labels_on_image_array(
#                     image_np,
#                     np.squeeze(boxes),
#                     np.squeeze(classes).astype(np.int32),
#                     np.squeeze(scores),
#                     category_index,
#                     use_normalized_coordinates=True,
#                     line_thickness=4,
#                     max_boxes_to_draw = 100,
#                     min_score_thresh=.1
#                 )
#                 im_width, im_height = image_np.shape[0:2]
#                 Scaled_boxes = np.zeros([int(num_detections[0]), 4])
#                 Scaled_boxes[:, 0] = boxes[0, 0:int(num_detections[0]), 0] * im_width
#                 Scaled_boxes[:, 1] = boxes[0, 0:int(num_detections[0]), 1] * im_height
#                 Scaled_boxes[:, 2] = boxes[0, 0:int(num_detections[0]), 2] * im_width
#                 Scaled_boxes[:, 3] = boxes[0, 0:int(num_detections[0]), 3] * im_height
#                 # for i in range(0,2):
#                 #     Scaled_boxes[:,i] = boxes[0,0:int(num_detections[0]),i]*im_width
#                 # for i in range(2,4):
#                 #     Scaled_boxes[:,i] = boxes[0,0:int(num_detections[0]),i]*im_height
#                 Scaled_scores = np.zeros([int(num_detections[0]),])
#                 Scaled_scores[:] = scores[0][0:int(num_detections[0])]
#                 # ---------------------------------------------------------------------------groundtruth
#                 GroundTruth = np.zeros([len(gt[img_idx]), 4])
#                 for c, obj in enumerate(gt[img_idx], 0):
#                     GroundTruth[c, 0] = float(obj['obj_bb'][0])
#                     GroundTruth[c, 1] = float(obj['obj_bb'][1])
#                     GroundTruth[c, 2] = float(obj['obj_bb'][0] + obj['obj_bb'][2])
#                     GroundTruth[c, 3] = float(obj['obj_bb'][1] + obj['obj_bb'][3])
#
#                 image_np2 = load_image_into_numpy_array(image)
#                 boxes2 = boxes
#                 boxes2.fill(0)
#                 scores2 = scores
#                 scores2.fill(0)
#                 classes2 = classes
#                 for c,obj in enumerate(gt[img_idx],0):
#                     # boxes[0, c, 0:2] = np.divide(map(float,obj['obj_bb'][0:2]),im_width)
#                     # boxes[0, c, 2:4] = np.divide(map(float,obj['obj_bb'][0:2])+map(float,obj['obj_bb'][4:-3:-1]),im_height)
#                     # ymin
#                     boxes2[0, c, 0] = np.divide(float(obj['obj_bb'][0]), im_height)
#                     # xmin
#                     boxes2[0, c, 1] = np.divide(float(obj['obj_bb'][1]),im_width)
#                     # ymax
#                     boxes2[0, c, 2] = np.divide(float(obj['obj_bb'][0] + obj['obj_bb'][2]), im_height)
#                     # xmax
#                     boxes2[0, c, 3] = np.divide(float(obj['obj_bb'][1] + obj['obj_bb'][3]), im_width)
#                     scores2[0][c] = 1.0
#                 vis_util.visualize_boxes_and_labels_on_image_array(
#                     image_np2,
#                     np.squeeze(boxes2),
#                     np.squeeze(classes2).astype(np.int32),
#                     np.squeeze(scores2),
#                     category_index,
#                     use_normalized_coordinates=True,
#                     line_thickness=4,
#                     max_boxes_to_draw=100,
#                     min_score_thresh=.1
#                 )
#                 # plt.imsave(fname='/home/juil/Downloads/synthetic_data_analysis/analysis/detection_result/synthetic_detection/detection_synthetic_{}.png'.format(idx), arr=image_np)
#                 # plt.imsave(fname='/home/juil/Downloads/synthetic_data_analysis/analysis/detection_result/synthetic_gt/gt_synthetic_{}.png'.format(idx), arr=image_np2)
#                 Detection_results_and_GT.append({'Number_of_detection':num_detections,'detected_boxes': Scaled_boxes,'detected_scores': Scaled_scores,'GroundTruth': GroundTruth})
#                 idx+=1
#
# with open('results_synthetic_dataset.pkl', 'wb') as handle:
#     pickle.dump(Detection_results_and_GT, handle)


# PATH_TO_TEST_IMAGES_DIR = '/home/juil/Downloads/training_scene_generator_20170901/sixd_toolkit-master/output/render'
PATH_TO_TEST_IMAGES_DIR = '/home/juil/workspace/6DOF-datasets/Tejani/test/02'
# img_path = PATH_TO_TEST_IMAGES_DIR+'/coffee_{:03d}/depth/{:04d}.png'
img_path = PATH_TO_TEST_IMAGES_DIR+'/depth/{:04d}.png'
gt_path = PATH_TO_TEST_IMAGES_DIR+'/gt.yml'

Detection_results_and_GT = list()
idx = 0
with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        with open(gt_path, 'r') as f:
            gt = yaml.load(f, Loader=yaml.CLoader)
        for img_idx in range(1, 1000):
            print(idx)
            # image = Image.open(img_path.format(img_idx))
            img = imread(img_path.format(img_idx))
            height = img.shape[0]
            width = img.shape[1]
            output_im = np.zeros((height, width, 3))
            output_im[:, :, 0] = img
            output_im[:, :, 1] = img
            output_im[:, :, 2] = img
            imsave('temp.png', output_im)
            image = Image.open('temp.png')
            rgb_image_np = load_image_into_numpy_array(Image.open(PATH_TO_TEST_IMAGES_DIR +'/rgb/{:04d}.png'.format(img_idx)))
            image_np = load_image_into_numpy_array(image)
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            # Actual detection.
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            vis_util.visualize_boxes_and_labels_on_image_array(
                # image_np,
                rgb_image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=4,
                max_boxes_to_draw=100,
                min_score_thresh=.9
            )
            im_width, im_height = image_np.shape[0:2]
            Scaled_boxes = np.zeros([int(num_detections[0]), 4])
            Scaled_boxes[:, 0] = boxes[0, 0:int(num_detections[0]), 0] * im_width
            Scaled_boxes[:, 1] = boxes[0, 0:int(num_detections[0]), 1] * im_height
            Scaled_boxes[:, 2] = boxes[0, 0:int(num_detections[0]), 2] * im_width
            Scaled_boxes[:, 3] = boxes[0, 0:int(num_detections[0]), 3] * im_height

            # for i in range(0, 2):
            #     Scaled_boxes[:, i] = boxes[0, 0:int(num_detections[0]), i] * im_width
            # for i in range(2, 4):
            #     Scaled_boxes[:, i] = boxes[0, 0:int(num_detections[0]), i] * im_height
            Scaled_scores = np.zeros([int(num_detections[0]), ])
            Scaled_scores[:] = scores[0][0:int(num_detections[0])]
            # ---------------------------------------------------------------------------groundtruth
            GroundTruth = np.zeros([len(gt[img_idx]), 4])
            for c, obj in enumerate(gt[img_idx], 0):
                GroundTruth[c, 0] = float(obj['obj_bb'][1])
                GroundTruth[c, 1] = float(obj['obj_bb'][0])
                GroundTruth[c, 2] = float(obj['obj_bb'][1] + obj['obj_bb'][3])
                GroundTruth[c, 3] = float(obj['obj_bb'][0] + obj['obj_bb'][2])

            image_np2 = load_image_into_numpy_array(image)
            boxes2 = boxes
            boxes2.fill(0)
            scores2 = scores
            scores2.fill(0)
            classes2 = classes
            for c, obj in enumerate(gt[img_idx], 0):
                # boxes[0, c, 0:2] = np.divide(map(float,obj['obj_bb'][0:2]),im_width)
                # boxes[0, c, 2:4] = np.divide(map(float,obj['obj_bb'][0:2])+map(float,obj['obj_bb'][4:-3:-1]),im_height)
                # ymin
                boxes2[0, c, 0] = np.divide(float(obj['obj_bb'][1]), im_width)
                # xmin
                boxes2[0, c, 1] = np.divide(float(obj['obj_bb'][0]), im_height)
                # ymax
                boxes2[0, c, 2] = np.divide(float(obj['obj_bb'][1] + obj['obj_bb'][3]), im_width)
                # xmax
                boxes2[0, c, 3] = np.divide(float(obj['obj_bb'][0] + obj['obj_bb'][2]), im_height)
                scores2[0][c] = 1.0
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np2,
                np.squeeze(boxes2),
                np.squeeze(classes2).astype(np.int32),
                np.squeeze(scores2),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=4,
                max_boxes_to_draw=100,
                min_score_thresh=.5
            )

            Detection_results_and_GT.append({'Number_of_detection': num_detections, 'detected_boxes': Scaled_boxes,
                                             'detected_scores': Scaled_scores, 'GroundTruth': GroundTruth})
            idx += 1
            plt.imsave(
                fname='/home/juil/Downloads/synthetic_data_analysis/analysis/detection_result/tejani_detection/detection_{}.png'.format(idx), arr=rgb_image_np)
with open('Detection_results_and_GT_real_data_tejani.pkl', 'wb') as handle:
    pickle.dump(Detection_results_and_GT, handle)


1





