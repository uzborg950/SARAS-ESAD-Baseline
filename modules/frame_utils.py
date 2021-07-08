import torch
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def save_image(image_tensor,ax,filename):
    image = image_tensor.detach().cpu().numpy()
    if ax is None:
        figure, ax = plt.subplots(1)
    ax.imshow(image)
    plt.savefig(filename)
    plt.close()
def show_image(image_tensor,ax,filename=None):
    image = image_tensor.detach().cpu().numpy()
    if ax is None:
        figure, ax = plt.subplots(1)
    ax.imshow(image)
    if filename is not None:
        plt.savefig(filename)
    plt.show()

def generate_frame(image_tensor, bboxes, filename, ax=None, gt=False, show=False):  # image tensor height, width, channels

    ax = superimpose_bboxes(bboxes, ax, gt)
    if show:
        show_image(image_tensor, ax, filename)

    return ax

def superimpose_bboxes(bboxes, ax, gt):

    if ax is None:
        figure, ax = plt.subplots(1)
    for b in range(bboxes.shape[0]):
        top_x, top_y, bot_x, bot_y, action = bboxes[b]
        if gt:
            color = 'g'
            linewidth=4
        else:
            cm = plt.get_cmap('CMRmap')
            color = cm(1. * (1 +action.numpy()) / 22)
            linewidth=1.5
        rect = patches.Rectangle((top_x, top_y), bot_x - top_x, bot_y - top_y, edgecolor=color , facecolor="none", linewidth=linewidth)
        ax.annotate(str(int(action.detach().cpu().numpy())), (top_x + 15, top_y - 15), color='white', weight='bold', fontsize=10, ha='center', va='center')
        ax.add_patch(rect)
    return ax
