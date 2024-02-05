import os 
import numpy as np
import pandas as pd
import cv2
import argparse
import matplotlib.pyplot as plt
import tensorflow as tf

from tqdm import tqdm
import time
import json


image_dir = ''
mask_dir = ''
phase = 'testing'

master_list = []
master_img = {
    'file_name': "",
    'height': 0,
    'width': 0,
    'image_id': "",
    'annotations': []
}

master_ann = {
    'bbox': [],
    'category_id': 0,
    'bbox_mode': 0,
}

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



def make_folds(npy_path, save_dir, folds,seed=42):
    if not os.path.exists(npy_path):
        raise ValueError("npy_path not exist")
    else:
        npy = np.load(npy_path, allow_pickle=True)
        np.random.seed(seed)
        np.random.shuffle(npy)
        print(f"Total number of images: {len(npy)}")

        # split into 0.2:0.8
        test_size = int(0.2 * len(npy))
        test = npy[:test_size]

        test = np.array(test)
        np.save(os.path.join(save_dir, 'test.npy'), test)

        train = npy[test_size:]
        print(f"Number of images in test set: {len(test)}")
        print(f"Number of images in train set: {len(train)}")
        number_per_fold = len(train) // folds
        for i in tqdm (range(folds), desc="Creating Folds...", ascii=False, ncols=75):  
            fold_dir = os.path.join(save_dir, f"fold_{i+1}")
            if not os.path.exists(fold_dir):
                os.makedirs(fold_dir)
            fold_val = train[int(i*number_per_fold):int((i+1)*number_per_fold)]
            fold_train = np.concatenate((train[:int(i*number_per_fold)], train[int((i+1)*number_per_fold):]), axis=0)
            # print(f"Number of images in fold {i+1} test set: {len(fold_val)}")
            # print(f"Number of images in fold {i+1} train set: {len(fold_train)}")
            np.save(os.path.join(fold_dir, 'val.npy'), fold_val)
            np.save(os.path.join(fold_dir, 'train.npy'), fold_train)

                
            if phase == 'testing' and i == 10:
                result = "testing complete"
                return result
    



def image_info(image_dir, mask_dir, save_dir, phase):
    if not os.path.exists(image_dir):
        raise ValueError(f"{image_dir} not exist")
    elif len(os.listdir(image_dir)) == 0:
        raise ValueError(f"{image_dir} is empty")
    else:
        master_list = []
        save_dir = os.path.join(save_dir, "master")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
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
                # print(f"{mask_path.split('/')[-1]} not exist")
                continue
            # print(image.shape)
            # print(mask.keys())
            loc_img = master_img.copy()
            loc_img['file_name'] = os.path.join("/storage/bic/Aakash/aakash-rao-capstone-project/datasets/master/NuCLSEvalSet/rgb", image_name)
            loc_img['height'] = im_height
            loc_img['width'] = im_width
            loc_img['image_id'] = image_name.split('.png')[0]
            loc_img['annotations'] = []
            for index, row in mask.iterrows():
                loc_ann = master_ann.copy()
                x_min = row['xmin']
                y_min = row['ymin']
                x_max = row['xmax']
                y_max = row['ymax']
                class_name = row['super_classification'] if row['super_classification'] in class_array else 'other'
                class_id = class_array.index(row['super_classification']) if row['super_classification'] in class_array else class_array.index('other')
                num_classes_per_image[class_id] += 1
                # print(f"Class: {class_name}, Class ID: {class_id}")
                loc_ann['bbox'] = [x_min, y_min, x_max, y_max]
                loc_ann['category_id'] = class_id
                loc_ann['bbox_mode'] = 0
                loc_img['annotations'].append(loc_ann.copy())
            master_list.append(loc_img.copy())
            # print(f"Image {i+1} completed")

                
            if phase == 'testing' and i == 10:
                result = "testing complete"
                # printing result in a pretty way
                print("\n")
                print("Result:")
                print("=======")
                for image in master_list:
                    print(image)
                    print("\n")
                    print("---------------------------------------------------------------------------------------------------------------------------------------")
                    print("\n")
                return result
            
        
        if phase != 'testing':
            master_list = np.array(master_list)
            np.save(os.path.join(save_dir, 'master.npy'), master_list)
            np.save(os.path.join(save_dir, 'num_classes_per_image.npy'), num_classes_per_image)
            print("Completed")

def image_info_three_class(image_dir, mask_dir, save_dir, phase, model_path):
    if not os.path.exists(image_dir):
        raise ValueError(f"{image_dir} not exist")
    elif len(os.listdir(image_dir)) == 0:
        raise ValueError(f"{image_dir} is empty")
    else:
        master_list = []
        save_dir = os.path.join(save_dir, "master")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        class_array = ['nonTIL_stromal', 'sTIL', 'tumor_any', 'other']
        num_classes_per_image = np.zeros(len(class_array))
        model = tf.keras.models.load_model(model_path)
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
                # print(f"{mask_path.split('/')[-1]} not exist")
                continue
            # print(image.shape)
            # print(mask.keys())
            loc_img = master_img.copy()
            loc_img['file_name'] = os.path.join("/storage/bic/Aakash/aakash-rao-capstone-project/datasets/master/NuCLSEvalSet/rgb", image_name)
            loc_img['height'] = im_height
            loc_img['width'] = im_width
            loc_img['image_id'] = image_name.split('.png')[0]
            loc_img['annotations'] = []
            for index, row in mask.iterrows():
                loc_ann = master_ann.copy()
                x_min = row['xmin']
                y_min = row['ymin']
                x_max = row['xmax']
                y_max = row['ymax']
                class_name = row['super_classification'] if row['super_classification'] in class_array else 'other'
                class_id = class_array.index(row['super_classification']) if row['super_classification'] in class_array else class_array.index('other')
                if class_name == 'other':
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
                    # print(pred)
                num_classes_per_image[class_id] += 1
                # print(f"Class: {class_name}, Class ID: {class_id}")
                loc_ann['bbox'] = [x_min, y_min, x_max, y_max]
                loc_ann['category_id'] = class_id
                loc_ann['bbox_mode'] = 0
                loc_img['annotations'].append(loc_ann.copy())
            master_list.append(loc_img.copy())
            # print(f"Image {i+1} completed")

                
            if phase == 'testing' and i == 10:
                result = "testing complete"
                # printing result in a pretty way
                print("\n")
                print("Result:")
                print("=======")
                for image in master_list:
                    print(image)
                    print("\n")
                    print("---------------------------------------------------------------------------------------------------------------------------------------")
                    print("\n")
                return result
            
        
        if phase != 'testing':
            master_list = np.array(master_list)
            np.save(os.path.join(save_dir, 'master.npy'), master_list)
            np.save(os.path.join(save_dir, 'num_classes_per_image.npy'), num_classes_per_image)
            print("Completed")

def image_info_single(image_dir, mask_dir, save_dir, phase):
    if not os.path.exists(image_dir):
        raise ValueError(f"{image_dir} not exist")
    elif len(os.listdir(image_dir)) == 0:
        raise ValueError(f"{image_dir} is empty")
    else:
        master_list = []
        save_dir = os.path.join(save_dir, "master")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
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
                # print(f"{mask_path.split('/')[-1]} not exist")
                continue
            # print(image.shape)
            # print(mask.keys())
            loc_img = master_img.copy()
            loc_img['file_name'] = os.path.join("/storage/bic/Aakash/aakash-rao-capstone-project/datasets/master/NuCLSEvalSet/rgb", image_name)
            loc_img['height'] = im_height
            loc_img['width'] = im_width
            loc_img['image_id'] = image_name.split('.png')[0]
            loc_img['annotations'] = []
            for index, row in mask.iterrows():
                loc_ann = master_ann.copy()
                x_min = row['xmin']
                y_min = row['ymin']
                x_max = row['xmax']
                y_max = row['ymax']
                class_name = 'Cell'
                class_id = 0
                num_classes_per_image[class_id] += 1
                # print(f"Class: {class_name}, Class ID: {class_id}")
                loc_ann['bbox'] = [x_min, y_min, x_max, y_max]
                loc_ann['category_id'] = class_id
                loc_ann['bbox_mode'] = 0
                loc_img['annotations'].append(loc_ann.copy())
            master_list.append(loc_img.copy())
            # print(f"Image {i+1} completed")

                
            if phase == 'testing' and i == 10:
                result = "testing complete"
                # printing result in a pretty way
                print("\n")
                print("Result:")
                print("=======")
                for image in master_list:
                    print(image)
                    print("\n")
                    print("---------------------------------------------------------------------------------------------------------------------------------------")
                    print("\n")
                return result
            
        
        if phase != 'testing':
            master_list = np.array(master_list)
            np.save(os.path.join(save_dir, 'master.npy'), master_list)
            # # saving as json
            # object = json.dumps(master_list.tolist(), indent = 4)
            # with open(os.path.join(save_dir, 'master.json'), "w+") as outfile:
            #     outfile.write(object)
            # np.save(os.path.join(save_dir, 'num_classes_per_image.npy'), num_classes_per_image)
            # print("Completed")





if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Data Formatter')
    argparser.add_argument('-i', '--image_dir', required=True, help='image directory')
    argparser.add_argument('-m', '--mask_dir', required=True, help='mask directory')
    argparser.add_argument('-s', '--save_dir', required=True, help='save directory')
    argparser.add_argument('-f', '--folds', required=True, help='number of folds')
    argparser.add_argument('-v', '--version', required=False, default='', help='version of the data')
    argparser.add_argument('-p', '--phase', required=True, help='phase')
    argparser.add_argument('-mp', '--model_path', required=False, help='model path')
    argparser.add_argument('--seed', required=False, help='Seed for reproducibility')

    args = argparser.parse_args()

    print("Creating Master...")
    if args.version == 'single':
        image_info_single(args.image_dir, args.mask_dir, args.save_dir, args.phase)
    elif args.version == 'three_class':
        image_info_three_class(args.image_dir, args.mask_dir, args.save_dir, args.phase, args.model_path)
    else:
        image_info(args.image_dir, args.mask_dir, args.save_dir, args.phase)
    print("Creating Folds...")
    make_folds(os.path.join(args.save_dir, 'master', 'master.npy'), args.save_dir, int(args.folds), int(args.seed))
    # print("Creating Plot...")
    # plot_num_classes(os.path.join(args.save_dir,'master', 'num_classes_per_image.npy'))