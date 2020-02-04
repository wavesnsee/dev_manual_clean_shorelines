#! /usr/bin/env python
# coding: utf8

"""
manual clean of detected coastlines, to only keep clean parts of detections
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import cv2
import shutil
import cams.geom as geom
import cams.img as cimg
from matplotlib import pyplot as plt
from os.path import join
from argparse import ArgumentParser
import numpy as np
from os.path import split as split
from os.path import exists as exists
from os import makedirs as makedirs
from glob import glob
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from pathlib import Path


class Opts(ArgumentParser):

    def __init__(self, *args, **kwargs):
        super(Opts, self).__init__(*args, **kwargs)

        self.add_argument(
            'Output_dir',
            help="Output directory"
        )

    def parse_args(self, *args, **kwargs):
        opts = super(Opts, self).parse_args(*args, **kwargs)
        return opts


def show(img, display=False):
    """ Create a figure of the input image `img`, that contains a shoreline. Option 'display' to show figure or not.

        Parameters
        ----------
        img, display

        Returns
        -------
        fig
        """
    import matplotlib.pyplot as plt
    im_plot = img
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cimg.showmpl(im_plot, ax=ax)
    xlim = plt.xlim()
    ylim = plt.ylim()
    plt.xlim(xlim)
    plt.ylim(ylim)
    if display:
        plt.show()
    return fig


def sl_mask_define(img, fig=None):
    """ Define shoreline mask(s) on the input image
    `img`, by interactively drawing polygon points on the image.

    Left click to add a point, right click to delete last point, middle
    click to validate.

    Parameters
    ----------
    img

    Returns
    -------
    roi : Zone
        instance of zone
    """
    roi = {}
    points = {}
    i = 0
    multiple_masks = True
    if not fig:
        fig = plt.figure(1)
        ax = fig.add_subplot(111)
    else:
        ax = fig.axes[0]

    while multiple_masks:
        msg = "PLEASE select zone by clicking on the image. Left click to "
        msg += " add a point, right \nclick to remove last point, middle"
        msg += " click or Return to validate."
        print(msg)
        ax.imshow(img[:, :, ::-1])
        points[str(i)] = fig.ginput(n=0, timeout=0)
        points[str(i)] = [(int(x), int(y)) for x, y in points[str(i)]]
        msg += " \n Do you want to mask an other part of the coastline ? (y/n)"
        other_mask = raw_input(msg)
        if other_mask == 'y':
            i += 1
        else:
            multiple_masks = False
    plt.close(1)

    for i in range(len(points)):
        roi[str(i)] = geom.Zone(points=points[str(i)], imshape=img.shape)
        if i == 0:
            roi_final = geom.Zone(points=points[str(i)], imshape=img.shape)
            roi_final_mask = roi_final.mask
        else:
            roi_final_mask *= roi[str(i)].mask

    print "\nDisplaying selected area... (Pixels OUT of ROI are red)"
    masked = cimg.paint_mask(img, roi_final_mask, display=False)
    plt.imshow(masked[:, :, ::-1])
    plt.show()
    return roi


def draw_shoreline(img, coastline, result_jpg):
    radius=5
    color = (0, 0, 255)
    thickness = -1
    for i in np.arange(1, len(coastline)-1, 1):
        x = coastline[i].split()[0]
        y = coastline[i].split()[1]
        img = cv2.circle(img, (int(float(x)), int(float(y))), radius, color, thickness)

    # Save
    cv2.imwrite(result_jpg, img)



if __name__ == '__main__':
    # Definition of the directories we need: coastline, coastline_selection, images
    opts = Opts().parse_args()
    Output_dir = opts.Output_dir
    Shoreline_dir = join(Output_dir, 'coastline/')
    Shoreline_selection_dir = join(Output_dir, 'coastline_selection/')
    images_dir = join(Output_dir, 'images/')

    # List of all coastline images
    ls = glob(Shoreline_dir + '*/*.jpg' )
    ls = np.sort(ls)

    for f in ls:
        path = Path(f)
        rep_selection_dir = join(Shoreline_selection_dir, path.parent.name)
        if not exists(rep_selection_dir):
            makedirs(rep_selection_dir)
        print(f)
        _, name_jpg = split(f)
        f_txt = f.replace('.jpg', '.txt')
        f_txt = f_txt.replace('coast_A', 'coast_px_A')
        _, name_txt = split(f_txt)
        f_txt_out = join(rep_selection_dir, name_txt)
        f_jpg_out = join(rep_selection_dir, name_jpg)
        f_jpg_A = glob('{images_dir}/*/{A_jpg}'.format(images_dir=images_dir, A_jpg=name_jpg.replace('coast_A_', 'A_')))

        # Plot input coastline image
        img = cimg.read(f)
        fig = show(img, display=True)

        # Decide if we want to keep it as it is, or perform a manual clean
        msg = "Do you want to remove any part(s) of the coastline detection? (y/n)"
        manual_clean = raw_input(msg)

        if manual_clean == 'y':
            fig = show(img, display=False)

            # Definition of the areas to be masked
            sl_mask = sl_mask_define(img, fig=fig)

            # Parsing every shoreline point, and check if we keep it or not
            with open(f_txt_out, 'w') as f_coastline_out:
                with open(f_txt) as f_coastline:
                    #skip header
                    line = f_coastline.readline()
                    line = f_coastline.readline()
                    while line:

                        # Shoreline points coordinates:
                        x = line.split()[0]
                        y = line.split()[1]
                        point = Point(float(x), float(y))

                        # check if shoreline point is inside of any of the areas to be masked:
                        spike = False
                        for i in range(len(sl_mask)):
                            polygon = Polygon(sl_mask[str(i)]._points)
                            if polygon.contains(point):
                                spike = True
                                break

                        # write clean shoreline
                        if not spike:
                            f_coastline_out.write(line)

                        line = f_coastline.readline()

            with open(f_txt_out, 'r') as f_coastline_out:
                coastline = f_coastline_out.readlines()

            # Plot cleaned shoreline
            img_A = cimg.read(f_jpg_A[0])
            draw_shoreline(img_A, coastline, f_jpg_out)

        else:
            # copy of the original coastline image and txt file
            shutil.copy(f, f_jpg_out)
            shutil.copy(f_txt, f_txt_out)










