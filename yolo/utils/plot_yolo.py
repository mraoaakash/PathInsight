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

def plot_model(data_dir, plot_dir):
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    for model in os.listdir(data_dir):
        model_plot_dir = os.path.join(plot_dir, model)
        if not os.path.exists(model_plot_dir):
            os.makedirs(model_plot_dir)

        mean_df = pd.read_csv(os.path.join(data_dir, model, f'mean_{model}.csv'), header=0)
        std_df = pd.read_csv(os.path.join(data_dir, model, f'std_{model}.csv'), header=0)

        for column in mean_df.columns:
            if column == 'epoch':
                continue
            mean = mean_df[column]
            std = std_df[column]
            plt.figure(figsize=(4, 4))
            plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.95)
            plt.locator_params(axis='x', nbins=5)
            plt.locator_params(axis='y', nbins=5)
            plt.grid(alpha=0.5, linestyle='--', linewidth=0.75)
            plt.plot(mean, label='mean',linewidth=0.75)
            plt.fill_between(mean.index, mean - std, mean + std, alpha=0.5, label='std')
            plt.title(title_dict[column], fontsize=12, fontweight='bold')
            plt.xlabel(x_axis_dict, fontsize=12, fontweight='bold')
            plt.ylabel(y_axis_dict[column], fontsize=12, fontweight='bold')
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
            plt.legend(fontsize=10)
            plt.tight_layout()
            plt.savefig(os.path.join(model_plot_dir, f'{column}.png'), dpi=300)
            plt.close()


def plot_metric(data_dir, plot_dir):
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    for column in cols:
        if column == 'epoch':
            continue
        plt.figure(figsize=(4, 4))
        plt.locator_params(axis='x', nbins=5)
        plt.locator_params(axis='y', nbins=5)
        plt.grid(alpha=0.5, linestyle='--', linewidth=0.75)
        for model in os.listdir(data_dir):
            mean_df = pd.read_csv(os.path.join(data_dir, model, f'mean_{model}.csv'), header=0)
            mean = mean_df[column]
            plt.plot(mean, label=model,linewidth=0.75)
            std_df = pd.read_csv(os.path.join(data_dir, model, f'std_{model}.csv'), header=0)
            std = std_df[column]
            plt.fill_between(mean.index, mean - std, mean + std, alpha=0.25, label=model)
        # custom legend content
        handles, labels = plt.gca().get_legend_handles_labels()
        # order = [0, 2, 4, 1, 3, 5]
        plt.legend(fontsize=8)
        plt.title(f'{title_dict[column]} for YOLOv5', fontsize=12, fontweight='bold')
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
    argparse.add_argument('-d','--data_dir', type=str, default='data')
    argparse.add_argument('-o','--output_dir', type=str, default='data')

    args = argparse.parse_args()
    plot_model(args.data_dir, args.output_dir)
    plot_metric(args.data_dir, args.output_dir)
