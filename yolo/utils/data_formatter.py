import os 
import numpy as np
import pandas as pd
import cv2
import argparse
import shutil
import sys
import random
import matplotlib.pyplot as plt
import tensorflow as tf

from tqdm import tqdm
import time


image_dir = ''
mask_dir = ''
phase = 'testing'

def plot_num_classes(info_file):
    if not os.path.exists(info_file):
        raise ValueError("info_file not exist")
    else:
        num_classes_per_image = np.load(info_file)
        plt.figure(figsize=(5, 5))
        class_array = ['nonTIL_stromal', 'sTIL', 'tumor_any', 'other']
        plt.bar(class_array, num_classes_per_image)
        plt.title('Number of classes per image', fontsize=14, fontweight='bold')
        plt.xlabel('Class', fontsize=14, fontweight='bold')
        plt.ylabel('Number of classes', fontsize=14, fontweight='bold')
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.savefig(os.path.join(os.path.dirname(info_file), 'num_classes_per_image.png'), dpi=300)



def make_folds(image_dir, mask_dir, save_dir, folds,seed=42):
    if folds <=0:
        raise ValueError("folds must be greater than 0")
    elif not os.path.exists(image_dir):
        raise ValueError("image_dir not exist")
    elif len(os.listdir(image_dir)) == 0:
        raise ValueError("image_dir is empty")
    else:
        images = os.listdir(image_dir)
        masks = os.listdir(mask_dir)
        print(f"Number of images: {len(images)}")

        random.seed(seed)
        random.shuffle(images)
        random.seed(seed)
        random.shuffle(masks)

        test_len = 0.2*len(images)
        print(f"Test length: {test_len}")
        test_imgs = images[:int(test_len)]
        test_masks = masks[:int(test_len)]

        for image in tqdm (test_imgs, desc="Creating Test...", ascii=False, ncols=75):
            im_save_path = os.path.join(save_dir,'test', 'images')
            mask_save_path = os.path.join(save_dir,'test', 'labels')
            if not os.path.exists(im_save_path):
                os.makedirs(im_save_path)
            if not os.path.exists(mask_save_path):
                os.makedirs(mask_save_path)
            time.sleep(0.01)
            image_path = os.path.join(image_dir, image)
            mask_path = os.path.join(mask_dir, image.split('.png')[0] + '.txt')
            shutil.copy(image_path, im_save_path)
            shutil.copy(mask_path, mask_save_path)

        images = images[int(test_len):]
        masks = masks[int(test_len):]

        print(f"Number of images: {len(images)}")
        len_of_each_fold = len(images) // folds

        random.seed(seed)
        random.shuffle(images)

        for i in tqdm (range(folds), desc="Creating Folds...", ascii=False, ncols=75):  
            print(f"Creating fold {i+1}")
            fold_dir = os.path.join(save_dir, f"fold_{i+1}")
            im_save_dir = os.path.join(fold_dir)
            mask_save_dir = os.path.join(fold_dir)
            if not os.path.exists(im_save_dir):
                os.makedirs(im_save_dir)
            if not os.path.exists(mask_save_dir):
                os.makedirs(mask_save_dir)
            try:
                images.remove('.DS_Store')
            except:
                pass
            train_images = images[:i*len_of_each_fold] + images[(i+1)*len_of_each_fold:]
            val_images = images[i*len_of_each_fold:(i+1)*len_of_each_fold]

            for image in tqdm (train_images, desc="Creating Train...", ascii=False, ncols=75):
                im_save_path = os.path.join(im_save_dir,'train', 'images')
                mask_save_path = os.path.join(mask_save_dir,'train', 'labels')
                if not os.path.exists(im_save_path):
                    os.makedirs(im_save_path)
                if not os.path.exists(mask_save_path):
                    os.makedirs(mask_save_path)
                time.sleep(0.01)
                image_path = os.path.join(image_dir, image)
                mask_path = os.path.join(mask_dir, image.split('.png')[0] + '.txt')
                shutil.copy(image_path, im_save_path)
                shutil.copy(mask_path, mask_save_path)
            for image in tqdm (val_images, desc="Creating Val...", ascii=False, ncols=75):
                im_save_path = os.path.join(im_save_dir,'val', 'images')
                mask_save_path = os.path.join(mask_save_dir,'val', 'labels')
                if not os.path.exists(im_save_path):
                    os.makedirs(im_save_path)
                if not os.path.exists(mask_save_path):
                    os.makedirs(mask_save_path)
                time.sleep(0.01)
                image_path = os.path.join(image_dir, image)
                mask_path = os.path.join(mask_dir, image.split('.png')[0] + '.txt')
                shutil.copy(image_path, im_save_path)
                shutil.copy(mask_path, mask_save_path)



def image_info(image_dir, mask_dir, save_dir, phase):
    if not os.path.exists(image_dir):
        raise ValueError("image_dir not exist")
    elif len(os.listdir(image_dir)) == 0:
        raise ValueError("image_dir is empty")
    else:
        save_dir = os.path.join(save_dir, "master")
        im_save_dir = os.path.join(save_dir, 'images')
        mask_save_dir = os.path.join(save_dir, 'labels')
        if not os.path.exists(im_save_dir):
            os.makedirs(im_save_dir)
        if not os.path.exists(mask_save_dir):
            os.makedirs(mask_save_dir)
        class_array = ['nonTIL_stromal', 'sTIL', 'tumor_any', 'other']
        num_classes_per_image = np.zeros(len(class_array))
        for i in tqdm (range(len(os.listdir(image_dir))), desc="Creating Master...", ascii=False, ncols=75):
            time.sleep(0.01)
            image_name = os.listdir(image_dir)[i]
            image_path = os.path.join(image_dir, image_name)
            mask_path = os.path.join(mask_dir, image_name.split('.png')[0] + '.csv')
            image = cv2.imread(image_path)
            im_height, im_width, _ = image.shape
            try:
                mask = pd.read_csv(mask_path, header=0)
            except:
                print(f"{mask_path.split('/')[-1]} not exist")
                continue
            # print(image.shape)
            # print(mask.keys())
            for index, row in mask.iterrows():
                x_min = row['xmin']
                y_min = row['ymin']
                x_max = row['xmax']
                y_max = row['ymax']
                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2
                width = x_max - x_min
                height = y_max - y_min
                norm_x_center = x_center / im_width
                norm_y_center = y_center / im_height
                norm_width = width / im_width
                norm_height = height / im_height
                class_name = row['super_classification']
                if class_name == 'AMBIGUOUS' or class_name == 'other_nucleus':
                    class_name = 'other'
                try:
                    class_id = class_array.index(class_name)
                except ValueError:
                    raise ValueError(f"{class_name} not in class_array")
                num_classes_per_image[class_id] += 1
                
                # print(x_min, y_min, x_max, y_max, class_name, class_id)
                yolo_format = f"{class_id} {norm_x_center} {norm_y_center} {norm_width} {norm_height}"
                # print(yolo_format)

                with open(os.path.join(mask_save_dir, image_name.split('.png')[0] + '.txt'), 'a') as f:
                    f.write(yolo_format + '\n')
                shutil.copy(image_path, im_save_dir)
            if phase == 'testing':
                result = "testing complete"
                return result
        np.save(os.path.join(save_dir, 'num_classes_per_image.npy'), num_classes_per_image)
        print("Completed")

def image_info_single(image_dir, mask_dir, save_dir, phase):
    if not os.path.exists(image_dir):
        raise ValueError(f"{image_dir} not exist")
    elif len(os.listdir(image_dir)) == 0:
        raise ValueError(f"{image_dir} is empty")
    else:
        save_dir = os.path.join(save_dir, "master")
        im_save_dir = os.path.join(save_dir, 'images')
        mask_save_dir = os.path.join(save_dir, 'labels')
        if not os.path.exists(im_save_dir):
            os.makedirs(im_save_dir)
        if not os.path.exists(mask_save_dir):
            os.makedirs(mask_save_dir)
        class_array = ['nonTIL_stromal', 'sTIL', 'tumor_any', 'other']
        num_classes_per_image = np.zeros(len(class_array))
        for i in tqdm (range(len(os.listdir(image_dir))), desc="Creating Master...", ascii=False, ncols=75):
            time.sleep(0.01)
            image_name = os.listdir(image_dir)[i]
            image_path = os.path.join(image_dir, image_name)
            mask_path = os.path.join(mask_dir, image_name.split('.png')[0] + '.csv')
            image = cv2.imread(image_path)
            im_height, im_width, _ = image.shape
            try:
                mask = pd.read_csv(mask_path, header=0)
            except:
                print(f"{mask_path.split('/')[-1]} not exist")
                continue
            # print(image.shape)
            # print(mask.keys())
            for index, row in mask.iterrows():
                x_min = row['xmin']
                y_min = row['ymin']
                x_max = row['xmax']
                y_max = row['ymax']
                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2
                width = x_max - x_min
                height = y_max - y_min
                norm_x_center = x_center / im_width
                norm_y_center = y_center / im_height
                norm_width = width / im_width
                norm_height = height / im_height
                class_name = row['super_classification']
                if class_name == 'AMBIGUOUS' or class_name == 'other_nucleus':
                    class_name = 'other'
                try:
                    class_id = class_array.index(class_name)
                except ValueError:
                    raise ValueError(f"{class_name} not in class_array")
                num_classes_per_image[class_id] += 1
                
                # print(x_min, y_min, x_max, y_max, class_name, class_id)
                yolo_format = f"0 {norm_x_center} {norm_y_center} {norm_width} {norm_height}"
                # print(yolo_format)

                with open(os.path.join(mask_save_dir, image_name.split('.png')[0] + '.txt'), 'a') as f:
                    f.write(yolo_format + '\n')
                shutil.copy(image_path, im_save_dir)
            if phase == 'testing':
                result = "testing complete"
                return result
        np.save(os.path.join(save_dir, 'num_classes_per_image.npy'), num_classes_per_image)
        print("Completed")


def image_info_three_class(image_dir, mask_dir, save_dir, phase, model_path):
    if not os.path.exists(image_dir):
        raise ValueError("image_dir not exist")
    elif len(os.listdir(image_dir)) == 0:
        raise ValueError("image_dir is empty")
    else:
        model = tf.keras.models.load_model(model_path)
        save_dir = os.path.join(save_dir, "master")
        im_save_dir = os.path.join(save_dir, 'images')
        mask_save_dir = os.path.join(save_dir, 'labels')
        if not os.path.exists(im_save_dir):
            os.makedirs(im_save_dir)
        if not os.path.exists(mask_save_dir):
            os.makedirs(mask_save_dir)
        class_array = ['nonTIL_stromal', 'sTIL', 'tumor_any', 'other']
        num_classes_per_image = np.zeros(len(class_array))
        for i in tqdm (range(len(os.listdir(image_dir))), desc="Creating Master...", ascii=False, ncols=75):
            time.sleep(0.01)
            image_name = os.listdir(image_dir)[i]
            image_path = os.path.join(image_dir, image_name)
            mask_path = os.path.join(mask_dir, image_name.split('.png')[0] + '.csv')
            image = cv2.imread(image_path)
            im_height, im_width, _ = image.shape
            try:
                mask = pd.read_csv(mask_path, header=0)
            except:
                print(f"{mask_path.split('/')[-1]} not exist")
                continue
            # print(image.shape)
            # print(mask.keys())
            for index, row in mask.iterrows():
                x_min = row['xmin']
                y_min = row['ymin']
                x_max = row['xmax']
                y_max = row['ymax']
                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2
                width = x_max - x_min
                height = y_max - y_min
                norm_x_center = x_center / im_width
                norm_y_center = y_center / im_height
                norm_width = width / im_width
                norm_height = height / im_height
                class_name = row['super_classification']
                if class_name == 'AMBIGUOUS' or class_name == 'other_nucleus':
                    x1 = row['xmin']
                    x2 = row['xmax']
                    y1 = row['ymin']
                    y2 = row['ymax']
                    # making a square box from the rectangle
                    if (x2-x1) > (y2-y1):
                        y2 = y1 + (x2-x1)
                    else:
                        x2 = x1 + (y2-y1)
                    test_img = image[y1:y2, x1:x2]
                    test_img = cv2.resize(test_img, (300, 300))
                    pred = model.predict(np.array([test_img]))
                    pred = np.argmax(pred)
                    class_id = pred
                    class_name = class_array[class_id]
                    # print(pred)
                else:
                    try:
                        class_id = class_array.index(class_name)
                        class_name = class_array[class_id]
                    except ValueError:
                        raise ValueError(f"{class_name} not in class_array")
                
                # print(x_min, y_min, x_max, y_max, class_name, class_id)
                yolo_format = f"{class_id} {norm_x_center} {norm_y_center} {norm_width} {norm_height}"
                # print(yolo_format)

                with open(os.path.join(mask_save_dir, image_name.split('.png')[0] + '.txt'), 'a') as f:
                    f.write(yolo_format + '\n')
                shutil.copy(image_path, im_save_dir)
            if phase == 'testing':
                result = "testing complete"
                return result
        np.save(os.path.join(save_dir, 'num_classes_per_image.npy'), num_classes_per_image)
        print("Completed")


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Data Formatter')
    argparser.add_argument('-i', '--image_dir', required=True, help='image directory')
    argparser.add_argument('-m', '--mask_dir', required=True, help='mask directory')
    argparser.add_argument('-s', '--save_dir', required=True, help='save directory')
    argparser.add_argument('-f', '--folds', required=True, help='number of folds')
    argparser.add_argument('-p', '--phase', required=True, help='phase')
    argparser.add_argument('-v', '--version', required=True, help='version')
    argparser.add_argument('-mp', '--model_path', required=False, help='version')
    argparser.add_argument('--seed', required=False, help='Seed for reproducibility')

    args = argparser.parse_args()

    # image_info(args.image_dir, args.mask_dir, args.save_dir, args.phase,"version")
    if args.version == 'single':
        image_info_single(args.image_dir, args.mask_dir, args.save_dir, args.phase)
    elif args.version == 'three_class':
        image_info_three_class(args.image_dir, args.mask_dir, args.save_dir, args.phase, args.model_path)
    else:
        image_info(args.image_dir, args.mask_dir, args.save_dir, args.phase)
    make_folds(os.path.join (args.save_dir, 'master', 'images'), os.path.join (args.save_dir, 'master', 'labels'), args.save_dir, int(args.folds), int(args.seed))
    # plot_num_classes(os.path.join(args.save_dir,'master', 'num_classes_per_image.npy'))