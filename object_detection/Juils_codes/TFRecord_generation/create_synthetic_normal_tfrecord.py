import hashlib
import io
import logging
import os
import PIL.Image
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import inout
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

########################################################################################################################
# def dict_to_tf_example(data,
#                        label_map_dict,
#                        image_subdirectory,
#                        ignore_difficult_instances=False):
def dict_to_tf_example(img_path,visible_obj,scene_gt,label_map_dict):

    ################################ juil_ have to change this part
    with tf.gfile.GFile(img_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    img = PIL.Image.open(encoded_jpg_io)
    rgb_image = img.convert('RGB')
    rgb_image.save('temp.jpg')
    with tf.gfile.GFile('temp.jpg', 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_jpg).hexdigest()

    # width = int(data['size']['width'])
    # height = int(data['size']['height'])
    width = int(image.size[0])
    height= int(image.size[1])


    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    truncated = []
    poses = []
    difficult_obj = []
    for obj in range(0,len(scene_gt)):
        if (scene_gt[obj]['visib_percentage']<0.3):
            difficult = True
        else:
            difficult = False
        if (difficult is False):

            difficult_obj.append(int(difficult))

            # xmin.append(float(obj['bndbox']['xmin']) / width)
            # ymin.append(float(obj['bndbox']['ymin']) / height)
            # xmax.append(float(obj['bndbox']['xmax']) / width)
            # ymax.append(float(obj['bndbox']['ymax']) / height)

            xmin.append(float(scene_gt[obj]['obj_bb'][1]) / width)
            ymin.append(float(scene_gt[obj]['obj_bb'][0]) / height)
            xmax.append(float(scene_gt[obj]['obj_bb'][1]+scene_gt[obj]['obj_bb'][3]) / width)
            ymax.append(float(scene_gt[obj]['obj_bb'][0]+scene_gt[obj]['obj_bb'][2]) / height)

            classes.append(scene_gt[obj]['obj_id'])
            classes_text.append(label_map_dict.keys()[label_map_dict.values().index(scene_gt[obj]['obj_id'])].encode('utf8'))

            truncated.append(int(0))
            poses.append('unspecified'.encode('utf8'))

            # visualize
            #print(scene_gt[obj]['obj_id'])
            #fig,ax = plt.subplots(1)
            #ax.imshow(image)
            #rect = patches.Rectangle((scene_gt[obj]['obj_bb'][1],scene_gt[obj]['obj_bb'][0]),scene_gt[obj]['obj_bb'][3],scene_gt[obj]['obj_bb'][2],linewidth=1,edgecolor='r',facecolor='none')
            #ax.add_patch(rect)
            #plt.show()
            #1



    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(
            img_path.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(
            img_path.encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
        'image/object/truncated': dataset_util.int64_list_feature(truncated),
        'image/object/view': dataset_util.bytes_list_feature(poses),
    }))
    return example
########################################################################################################################





# center_obj_id,scene_id, im_id = 1,1,0




flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw PASCAL VOC dataset.')
flags.DEFINE_string('set', 'train', 'Convert training set, validation set or '
                    'merged set.')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS
SETS = ['train', 'val', 'trainval', 'test']

# def main(_):
FLAGS.set = 'train'
if FLAGS.set not in SETS:
    raise ValueError('set must be in : {}'.format(SETS))
FLAGS.output_path = 'synthetic_train_2to3_surface.record'
FLAGS.data_dir = '/home/juil/workspace/synthetic_scene_generation/output/render'

writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
label_map_dict = label_map_util.get_label_map_dict('/home/juil/workspace/tensorflow_object_detection/object_detection/data/synthetic_label_map.pbtxt')



for center_obj_id in range(2,4):
    for scene_id in range(1,100):
        for im_id in range(0,72):
            print('working on center_obj_id : {:03d}, scene_id : {:03d}, im_id : {:03d}'.format(center_obj_id,scene_id,im_id))
            im_id_str = '{0:04d}'.format(im_id)
            base_path = '/home/juil/workspace/training_scene_generator/sixd_toolkit-master/output/render/{:03d}_{:03d}'.format(center_obj_id,scene_id)
            img_path = base_path + '/rgb/' + im_id_str + '.png'
            surface_path = base_path + '/normal_map/' + im_id_str + '.png'
            depth_path = base_path + '/depth' + im_id_str + '.png'
            # visibmask_path = base_path+'/visib'+img_num_str+'.png'
            # invisibmask_path = base_path+'/invisib'+img_num_str+'.png'
            scene_info = inout.load_info(
                '/home/juil/workspace/training_scene_generator/sixd_toolkit-master/output/render/{:03d}_{:03d}/info.yml'.format(
                    center_obj_id, scene_id))
            scene_gt = inout.load_gt(
                '/home/juil/workspace/training_scene_generator/sixd_toolkit-master/output/render/{:03d}_{:03d}/gt.yml'.format(
                    center_obj_id, scene_id))
            visible_obj = scene_info[im_id]['visible_obj']
            tf_example = dict_to_tf_example(surface_path, visible_obj, scene_gt[im_id], label_map_dict)
            writer.write(tf_example.SerializeToString())

writer.close()


