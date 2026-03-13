import os
import numpy as np
import torch
import torchvision
import time
from PIL import Image
import uuid
import glob
import itertools

global_res_save_counter = itertools.count()

#create save_path
def create_experiment_folder(check_base_dir, image_base_dir):
    os.makedirs(check_base_dir, exist_ok=True)
    os.makedirs(image_base_dir, exist_ok=True)

    check_experiment_folders = [int(folder) for folder in os.listdir(check_base_dir) if folder.isdigit()]
    next_check_num = max(check_experiment_folders, default=-1) + 1
    check_new_folder = os.path.join(check_base_dir, str(next_check_num))
    os.makedirs(check_new_folder, exist_ok=True)

    img_experiment_folders = [int(folder) for folder in os.listdir(image_base_dir) if folder.isdigit()]
    next_img_num = max(img_experiment_folders, default=-1) + 1
    img_new_folder = os.path.join(image_base_dir, str(next_img_num))
    os.makedirs(img_new_folder, exist_ok=True)
    os.makedirs(f"{img_new_folder}/train", exist_ok=True)
    os.makedirs(f"{img_new_folder}/val", exist_ok=True)

    return check_new_folder, img_new_folder

# image save
def save_img(out_images, in_images, gt, masks, g_image,  path, epoch, mode, per, batch_szie, edgemap=None, name = None):
    os.makedirs(path, exist_ok=True)
    if mode == "save_input":
        file_name = name[0] if name is not None else "input"
        torchvision.utils.save_image(in_images.squeeze(0), f'{path}/{file_name}.png')
    if mode == "test":
        file_name = name[0] if name is not None else "test"
        torchvision.utils.save_image(out_images.squeeze(0), f'{path}/{file_name}.png')
    elif mode == "train":
        grid_generated = torchvision.utils.make_grid(out_images, nrow=batch_szie)
        grid_input = torchvision.utils.make_grid(in_images, nrow=batch_szie)
        gird_g_image = torchvision.utils.make_grid(g_image, nrow=batch_szie)
        grid_gt = torchvision.utils.make_grid(gt, nrow=batch_szie)
        grid_masks = torchvision.utils.make_grid(masks, nrow=batch_szie)
        if edgemap != None:
            grid_edgemap = torchvision.utils.make_grid(edgemap, nrow=batch_szie)
            combined_grid = torch.cat((grid_masks, grid_input, grid_edgemap, gird_g_image, grid_generated, grid_gt), dim=1)
        else:
            combined_grid = torch.cat((grid_masks, grid_input, gird_g_image, grid_generated, grid_gt), dim=1)
        
        torchvision.utils.save_image(combined_grid, f'{path}/train/Epoch_{epoch+1}_{int(per)}per.png')
        
    elif mode == "val":
        grid_generated = torchvision.utils.make_grid(out_images, nrow=batch_szie)
        grid_input = torchvision.utils.make_grid(in_images, nrow=batch_szie)
        gird_g_image = torchvision.utils.make_grid(g_image, nrow=batch_szie)
        grid_gt = torchvision.utils.make_grid(gt, nrow=batch_szie)
        grid_masks = torchvision.utils.make_grid(masks, nrow=batch_szie)
        if edgemap != None:
            grid_edgemap = torchvision.utils.make_grid(edgemap, nrow=batch_szie)
            combined_grid = torch.cat((grid_masks, grid_input, grid_edgemap, gird_g_image, grid_generated, grid_gt), dim=1)
        else:
            combined_grid = torch.cat((grid_masks, grid_input, gird_g_image, grid_generated, grid_gt), dim=1)
            
        torchvision.utils.save_image(combined_grid, f'{path}/val/Epoch_{epoch+1}_{int(per)}per.png')


