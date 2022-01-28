import pickle

import pprint

import matplotlib.pyplot as plt
from matplotlib import patches, lines, colors
from matplotlib.patches import Polygon

from matplotlib.colors import colorConverter, to_rgba
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import os
from copy import deepcopy
import numpy as np

import pprint
from skimage.measure import find_contours
from matplotlib.patches import Polygon
import copy
import nibabel as nib
import glob


def get_largest_tumor_slice(gt):

    # this function gives prefence to enhancing, necrotic and edema (in order)

    size_of_tumor_per_slice = []

    if 4 in np.unique(gt).tolist(): tum_idx = 4
    elif 1 in np.unique(gt).tolist(): tum_idx = 1
    else: tum_idx = 2

    for sleic in range(gt.shape[2]):
        size_of_tumor_per_slice.append(np.count_nonzero(gt[:,:, sleic] == tum_idx))

    largest_tumor_slice = np.argmax(size_of_tumor_per_slice)

    return largest_tumor_slice

def get_largest_tumor_slice_binary(gt):
    gt_binary = copy.deepcopy(gt)
    gt_binary[gt_binary>0] = 1
    return get_largest_tumor_slice(gt_binary)

def add_polygon_mask_outline_to_ax(ax, mask, fc, ec, lw = 1):
    '''
    Takes 2d (128 x 128) mask as input and adds it to ax as patch
    fc, ec: (R, G, B, alpha)
    '''
    # Mask Polygon
    # Pad to ensure proper polygons for masks that touch image edges.
    padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
    padded_mask[1:-1, 1:-1] = mask
    contours = find_contours(padded_mask, 0.5)
    for verts in contours:
        # Subtract the padding and flip (y, x) to (x, y)
        verts = np.fliplr(verts) - 1
        # p = Polygon(verts, facecolor="None", edgecolor=color, lw = lw, alpha = alpha)
        # p = Polygon(verts, facecolor=color, edgecolor=color, lw = lw, alpha = alpha)
        p = Polygon(verts, facecolor=fc, edgecolor=ec, lw = lw)
        ax.add_patch(p)
    return ax

def extract_2D_bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax

def plot_prediction_3d(data, modalities, segs, seg_preds, ax_title, outfile):
    """
    plot the input images, ground truth annotations, and output predictions of a batch. If 3D batch, plots a 2D projection
    of one randomly sampled element (patient) in the batch. Since plotting all slices of patient volume blows up costs of
    time and space, only a section containing a randomly sampled ground truth annotation is plotted.
    :param batch: dict with keys: 'data' (input image), 'seg' (pixelwise annotations), 'pid'
    :param results_dict: list over batch element. Each element is a list of boxes (prediction and ground truth),
    where every box is a dictionary containing box_coords, box_score and box_type.
    """


    # if test_aug = True in config, then seg_pred might have multiple predicted segmentations for single patient (i.e. shape[1] > 1)
    # in that case, take mean of all the segmentation masks (i.e. take mean across axis = 1)
    if seg_preds.shape[1] > 1:
        seg_preds = np.mean(seg_preds, 1)[:, None]

    
    # Randomly sampled one patient of batch and project data into 2D slices for plotting.
    patient_ix = 0
    data = np.transpose(data[patient_ix], axes=(3, 0, 1, 2)) 

    largest_slice = get_largest_tumor_slice_binary(np.squeeze(segs)) if np.count_nonzero(segs) else get_largest_tumor_slice_binary(np.squeeze(seg_preds))
    start_slice = max(largest_slice-25, 0)
    end_slice = min(start_slice + 50, 155)
    slice_idx = [start_slice, end_slice]
    data = data[slice_idx[0]: slice_idx[1]]
    segs = np.transpose(segs[patient_ix], axes=(3, 0, 1, 2))[slice_idx[0]: slice_idx[1]]
    seg_preds = np.transpose(seg_preds[patient_ix], axes=(3, 0, 1, 2))[slice_idx[0]: slice_idx[1]]

    try:
        # all dimensions except for the 'channel-dimension' are required to match
        for i in [0, 2, 3]:
            assert data.shape[i] == segs.shape[i] == seg_preds.shape[i]
    except:
        raise Warning('Shapes of arrays to plot not in agreement!'
                      'Shapes {} vs. {} vs {}'.format(data.shape, segs.shape, seg_preds.shape))

    if np.count_nonzero(segs):
        show_arrays = np.concatenate([data, seg_preds, segs], axis=1).astype(float)
        modalities += ['Predicted', 'GT']
    else:
        show_arrays = np.concatenate([data, seg_preds], axis=1).astype(float)
        modalities += ['Predicted']

    step_size = 5 # every 'step_size'th column will be plotted
    approx_figshape = ((3/step_size) * show_arrays.shape[0], 3 * show_arrays.shape[1])
    fig = plt.figure(figsize=approx_figshape)
    
    gs = gridspec.GridSpec(show_arrays.shape[1] , len(range(0, show_arrays.shape[0], step_size)))
    gs.update(wspace=0.01, hspace=0.01)

    for bidx, b in enumerate(range(0, show_arrays.shape[0], step_size)): # columns
        for m in range(show_arrays.shape[1]): # rows

            ax = plt.subplot(gs[m, bidx])
            
            # This code snippet is to hide axes but show row-labels ## >>>>>
            ax.xaxis.set_visible(False)            
            ax.get_yaxis().set_ticks([])
            if bidx == 0:
                ax.set_ylabel(modalities[m], rotation=90, fontsize=10)
            ########################################################## >>>>>

            if m < show_arrays.shape[1]: 
                arr = np.fliplr(np.rot90(show_arrays[b, m]))

            # rows for data modalities
            if m < data.shape[1]: 
                plt.imshow(arr, cmap='gray', vmin=None, vmax=None)
                # Tumor outline overlay on scan
                pred_mask = np.fliplr(np.rot90(show_arrays[b, data.shape[1]]))
                add_polygon_mask_outline_to_ax(ax, pred_mask, fc = to_rgba('m', 0), ec = to_rgba('m', 1), lw = 1)
            # rows for gt and predicted seg
            else:
                arr = arr.astype(int)
                seg_cmap = colors.ListedColormap(['navy', 'r', 'g', 'k', 'gold'])
                norm = colors.BoundaryNorm([i for i in range(5)], seg_cmap.N)
                # use interpolation = 'nearest' to prevent bleeding edge
                # source: https://stackoverflow.com/questions/68875811/matplotlib-imshow-with-listedcolormap-showing-wrong-colors-on-boundaries
                plt.imshow(arr, cmap=seg_cmap, norm=norm, interpolation='nearest')

            # ax titles
            if m == 0:
                plt.title('slice = {}'.format(list(range(slice_idx[0], slice_idx[1]))[b]), fontsize=10)

    plt.suptitle('{}'.format(ax_title), fontsize=15, y = 0.95)
    print(f"Saving plot at = {outfile}")
    plt.savefig(outfile, bbox_inches = 'tight')   
    plt.close(fig)


def get_zoomed_data(data, gt, pred, zoom_out_factor = 10):
    if np.count_nonzero(gt):
        rmin, rmax, cmin, cmax = extract_2D_bbox(np.squeeze(gt))
    else:
        rmin, rmax, cmin, cmax = extract_2D_bbox(np.squeeze(pred))

    height = rmax - rmin
    width = cmax - cmin

    rmin = rmin - zoom_out_factor
    cmin = cmin - zoom_out_factor
    rmax = rmin + max(height, width) + 2*zoom_out_factor
    cmax = cmin + max(height, width) + 2*zoom_out_factor

    # print("Zooming plot to", rmin, rmax, cmin, cmax)

    data = data[:,:,rmin:rmax,cmin:cmax,:]
    gt = gt[:,:,rmin:rmax,cmin:cmax,:]
    pred = pred[:,:,rmin:rmax,cmin:cmax,:]

    return data, gt, pred

