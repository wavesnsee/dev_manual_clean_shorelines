#! /usr/bin/env python
# coding: utf8

"""
manual clean of detected coastlines, to only keep clean parts of detections

Usage: dev_manual_clean_shoreline.py <input_dir_coastline> <output_dir_coastline_selection>
"""
import pdb
from argparse import ArgumentParser
from glob import glob
import matplotlib.pyplot as plt
from mpl_point_clicker import clicker
from shapely.geometry import Point
import cv2
import os
import numpy as np


def direction(dx, dy):
    """
    Computation of the direction between 2 points A and B, defined in world coordinates (x,y), with xB = xA + dx,
    yB = yA + dy

    Parameters
    ----------
    dx: float
    dy: float

    Returns
    -------
    teta: float
    """
    # print('dx: %s,  dy:%s' %(dx, dy))
    if (dx > 0):
        teta = np.arctan(dy / dx)
    elif (dx < 0) * (dy >= 0):
        teta = np.arctan(dy / dx) + np.pi
    elif (dx < 0) * (dy <= 0):
        teta = np.arctan(dy / dx) + np.pi
    elif (dx == 0) * (dy < 0):
        teta = -np.pi / 2
    elif (dx == 0) * (dy > 0):
        teta = np.pi / 2
    # print('teta: %s' %teta)
    return teta


def compute_transect_points(pt_A, pt_B, d):
    dy = pt_B[1] - pt_A[1]
    dx = pt_B[0] - pt_A[0]
    teta = direction(float(dx), float(dy))

    transect_x_all = [pt_A[0]]
    transect_y_all = [pt_A[1]]
    transect_lists = [[pt_A[0], pt_A[1]]]
    transect_tuples = [(pt_A[0], pt_A[1])]

    cross_shore_d = [0]

    while Point(transect_tuples[-1][0], transect_tuples[-1][1]).distance(Point(pt_B[0], pt_B[1])) > 2 * d:
        transect_x_all.append(transect_x_all[-1] + d * np.cos(teta))
        transect_y_all.append(transect_y_all[-1] + d * np.sin(teta))
        transect_x = transect_tuples[-1][0] + d * np.cos(teta)
        transect_y = transect_tuples[-1][1] + d * np.sin(teta)
        transect_lists.append([transect_x, transect_y])
        transect_tuples.append((transect_x, transect_y))
        cross_shore_d.append(cross_shore_d[-1] + d)
    return transect_tuples, transect_lists, np.array(transect_x_all), np.array(transect_y_all), \
           np.array(cross_shore_d)


def oversample_shoreline_pixels(shoreline_pixels):
    line_x_oversampl = []
    line_y_oversampl = []
    line_x = shoreline_pixels[:, 0]
    line_y = shoreline_pixels[:, 1]
    for i in range(len(line_x) - 1):
        _, _, x_oversampl, y_oversampl, _ = compute_transect_points([line_x[i], line_y[i]],
                                                                    [line_x[i + 1], line_y[i + 1]], 25)
        line_x_oversampl.append(x_oversampl.tolist())
        line_y_oversampl.append(y_oversampl.tolist())
    line_x_oversampl = [item for sublist in line_x_oversampl for item in sublist]
    line_y_oversampl = [item for sublist in line_y_oversampl for item in sublist]

    shoreline_pixels_oversampl = np.zeros((len(line_x_oversampl), 2))
    shoreline_pixels_oversampl[:, 0] = line_x_oversampl
    shoreline_pixels_oversampl[:, 1] = line_y_oversampl

    return shoreline_pixels_oversampl


input_dir_average_imgs = '/home/florent/Projects/Etretat/CAM1/storm_study/atiyah/images/n_frames_min_600/A_atiyah_after_storm/'
output_dir_coastlines = '/home/florent/Projects/Etretat/CAM1/storm_study/atiyah/Shoreline_manually_created/after_storm/'


# List of all average images
ls = glob(input_dir_average_imgs + '/A_*.jpg')
ls = np.sort(ls)

for f in ls:
    _, name = os.path.split(f)
    print(f)

    #read image
    img = cv2.imread(f)

    # converting BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # mouse definition of shoreline
    fig, ax = plt.subplots(figsize=(20, 15), constrained_layout=True)
    ax.imshow(img)
    klicker = clicker(ax, ["event"], markers=["x"], **{"linestyle": "--"})
    plt.show()
    shoreline_positions = klicker.get_positions()
    shoreline_positions = shoreline_positions['event']
    shoreline_positions_oversample = oversample_shoreline_pixels(shoreline_positions)

    # plot oversampled  shoreline
    fig, ax = plt.subplots(figsize=(20, 15), constrained_layout=True)
    ax.imshow(img)
    ax.plot(shoreline_positions_oversample[:, 0], shoreline_positions_oversample[:, 1], '.r')
    plt.show()
    name_jpg = name.replace('A_', 'coast_px_A_')
    fig.savefig(os.path.join(output_dir_coastlines, name_jpg))
    print(shoreline_positions)

    # save pixel coordinates of shoreline in coastline txt file
    name_txt = name.replace('.jpg', '.txt')
    name_txt = name_txt.replace('A_', 'coast_px_A_')
    f_coast_txt = os.path.join(output_dir_coastlines, name_txt)
    with open(f_coast_txt, 'w') as f_out:
        for n in range(shoreline_positions_oversample.shape[0]):
            f_out.write('%i %i\n' %(np.int(np.around(shoreline_positions_oversample[n, 0])),
                                    np.int(np.around(shoreline_positions_oversample[n, 1]))))






