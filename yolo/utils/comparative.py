import os
import pandas as pd
import argparse
import matplotlib.pyplot as plt


cols = ['epoch', 'train_box_loss', 'train_obj_loss', 'train_cls_loss', 'precision', 'recall', 'mAP_50', 'mAP_5095', 'val_box_loss', 'val_obj_loss', 'val_cls_loss']
titles = ['epoch', 'Train-time Box Loss', 'Train-time Object Loss', 'Train-time Class Loss', 'Precision', 'Recall', 'mAP@50', 'mAP@50:95', 'Validation-time Box Loss', 'Validation-time Object Loss', 'Validation-time Class Loss']
axis = ['Epoch', 'Box Loss', 'Object Loss', 'Class Loss', 'Precision', 'Recall', 'mAP@50', 'mAP@50:95', 'Box Loss', 'Object Loss', 'Class Loss']
title_dict = dict(zip(cols, titles))
y_axis_dict = dict(zip(cols, axis))
x_axis_dict = "No. of Epochs"
model_list = ['yolov5m']
names = ['YOLOv5m']
model_dict = dict(zip(model_list, names))



def plot_metric(old_best_path, new_best_path, plot_dir, model):
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    for column in cols:
        if column == 'iteration' or 'unnamed' in column.lower():
            continue
        plt.figure(figsize=(4,4))
        plt.locator_params(axis='x', nbins=5)
        plt.locator_params(axis='y', nbins=5)
        plt.grid(alpha=0.5, linestyle='--', linewidth=0.75)
        old_model_mean = pd.read_csv(os.path.join(old_best_path, model, f'mean_{model}.csv'), header=0)
        old_model_mean = old_model_mean[column][:100]
        old_model_mean = old_model_mean.dropna(how='any')
        plt.plot(old_model_mean, label='Four-label',linewidth=0.75)
        new_model_mean = pd.read_csv(os.path.join(new_best_path, model, f'mean_{model}.csv'), header=0)
        new_model_mean = new_model_mean[column][:100]
        new_model_mean = new_model_mean.dropna(how='any')
        plt.plot(new_model_mean, label='Single-label',linewidth=0.75)
        plt.legend(loc='best', fontsize=8)
        old_model_std = pd.read_csv(os.path.join(old_best_path, model, f'std_{model}.csv'), header=0)
        old_model_std = old_model_std[column][:100]
        old_model_std = old_model_std.dropna(how='any')
        plt.fill_between(old_model_mean.index, old_model_mean - old_model_std, old_model_mean + old_model_std, alpha=0.25, label='Four-label')
        new_model_std = pd.read_csv(os.path.join(new_best_path, model, f'std_{model}.csv'), header=0)
        new_model_std = new_model_std[column][:100]
        new_model_std = new_model_std.dropna(how='any')
        plt.fill_between(new_model_mean.index, new_model_mean - new_model_std, new_model_mean + new_model_std, alpha=0.25, label='Single-label')
        plt.title(f'{title_dict[column]} for {model_dict[model]}', fontsize=12, fontweight='bold')

        plt.title(f'{title_dict[column]} for {model_dict[model]}', fontsize=12, fontweight='bold')
        plt.xlabel(x_axis_dict, fontsize=12, fontweight='bold')
        plt.ylabel(y_axis_dict[column], fontsize=12, fontweight='bold')
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'{column}.png'), dpi=300)
        plt.close()

        
    pass

if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument('-d','--old_path', type=str, default='data')
    argparse.add_argument('-n','--new_path', type=str, default='data')
    argparse.add_argument('-o','--output_dir', type=str, default='data')

    args = argparse.parse_args()
    plot_metric(args.old_path, args.new_path, args.output_dir, 'yolov5m')
