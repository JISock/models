import input_output
import os
from PIL import Image
import scipy.misc as scimisc
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import math

def convert_depth_map_to_surface_map(im):
    width = im.shape[1]
    height = im.shape[0]
    output_im = np.zeros((height,width,3))
    # output_im = np.zeros((height, width, 3))
    gx, gy = np.gradient(im)

    # for x in range(0,height-1):
    #     for y in range(0,width-1):
    #         dzdx = im[x+1][y] - im[x-1][y]/2.0
    #         dzdy = im[x][y+1] - im[x][y-1]/2.0
    #         normx = -dzdx
    #         normy = -dzdy
    #         normz = 1.0
    #         norm = math.sqrt(normx**2+normy**2+normz**2)
    #         # normalize
    #         output_im[x][y][0] = normx/norm
    #         output_im[x][y][1] = normy/norm
    #         output_im[x][y][2] = normz/norm
    normx = -gx
    normy = -gy
    normz = 1.0*np.zeros((height,width))
    norm  = np.sqrt(np.power(normx,2)+np.power(normy,2)+np.power(normz,2))
    output_im[:,:,0] = np.divide(normx , norm)
    output_im[:,:,1] = np.divide(normy , norm)
    output_im[:,:,2] = np.divide(normz , norm)
    no_nan_output_im = np.nan_to_num(output_im)
    return no_nan_output_im



root_path = '/home/juil/workspace/training_scene_generator/sixd_toolkit-master/output/render'
level1 = input_output.fn_list_all_dir(root_path)
for folder_name in level1:
    print(root_path+'/'+folder_name)
    current_dir = root_path+'/'+folder_name
    surface_normal_folder = root_path+'/'+folder_name + '/normal_map'
    if os.path.exists(surface_normal_folder):
        already_flag = 1
        os.rmdir(surface_normal_folder)
        already_flag = 0
    else:
        already_flag = 0

    if not already_flag:
        os.makedirs(surface_normal_folder)
        depth_folder = current_dir + '/depth'
        img_list = input_output.fn_list_all_img_in_a_dir(depth_folder)
        for img_name in img_list:
            # im = Image.open(depth_folder+ '/' +img_name)
            depth_im = scimisc.imread(depth_folder + '/' + img_name, 'I')
            surface_map = convert_depth_map_to_surface_map(depth_im)

            # im_surface = Image.fromarray(surface_map)
            # im_surface.save(surface_normal_folder+'/'+img_name)
            scimisc.imsave(surface_normal_folder+'/'+img_name,surface_map)
            print(surface_normal_folder+'/'+img_name)
