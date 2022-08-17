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
import cv2
import os
import numpy as np


input_dir_average_imgs = '/home/florent/Projects/Etretat/CAM1/storm_study/atiyah/images/n_frames_min_600/A_atiyah/'
output_dir_coastlines = '/home/florent/Projects/Etretat/CAM1/storm_study/atiyah/Shoreline_manually_created/'


# List of all average images
ls = glob(input_dir_average_imgs + '/A_*.jpg')
ls = np.sort(ls)

for f in ls:
    _, name = os.path.split(f)
    print(f)

    # mouse definition of shoreline
    fig, ax = plt.subplots(constrained_layout=True)
    img = cv2.imread(f)
    ax.imshow(img)
    klicker = clicker(ax, ["event"], markers=["x"], **{"linestyle": "--"})
    plt.show()
    shoreline_positions = klicker.get_positions()
    shoreline_positions = shoreline_positions['event']
    print(shoreline_positions)

    # save pixel coordinates of shoreline in coastline txt file
    name_txt = name.replace('.jpg', '.txt')
    name_txt = name_txt.replace('A_', 'coast_px_A')
    f_coast_txt = os.path.join(output_dir_coastlines, name_txt)
    with open(f_coast_txt, 'w') as f_out:
        for n in range(shoreline_positions.shape[0]):
            f_out.write('%i %i\n' %(np.int(np.around(shoreline_positions[n, 0])),
                                    np.int(np.around(shoreline_positions[n, 1]))))






