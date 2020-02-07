#! /usr/bin/env python
# coding: utf8

"""
manual clean of detected coastlines, to only keep clean parts of detections

Usage: dev_manual_clean_shoreline.py <input_dir_coastline> <output_dir_coastline_selection>
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import pdb
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
from matplotlib.widgets import Button


class Opts(ArgumentParser):

    def __init__(self, *args, **kwargs):
        super(Opts, self).__init__(*args, **kwargs)

        self.add_argument(
            'Intput_dir',
            help="Input directory"
        )

        self.add_argument(
            'Output_dir',
            help="Output directory"
        )

        self.add_argument(
            'Images_dir',
            help="Images directory"
        )

        self.add_argument(
            '--check_coastlines',
            help="Optional choice to check coastline_txt files"
        )

    def parse_args(self, *args, **kwargs):
        opts = super(Opts, self).parse_args(*args, **kwargs)
        return opts


def _throw(event):
    global choice
    choice = 't'
    plt.close()

def _keep(event):
    global choice
    choice = 'k'
    plt.close()

def _mask(event):
    global choice
    choice = 'r'
    plt.close()

def _choose(event):
    global choice
    choice = 'c'
    plt.close()


def show(img, display=False, moment_choice=False):
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
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111)
    cimg.showmpl(im_plot, ax=ax)
    xlim = plt.xlim()
    ylim = plt.ylim()
    plt.xlim(xlim)
    plt.ylim(ylim)

    if moment_choice:

        axcut = plt.axes([0.1, 0.9, 0.15, 0.05])
        bcut_throw = Button(axcut, 'Throw away', color='brown', hovercolor='lightgreen')
        bcut_throw.on_clicked(_throw)

        axcut = plt.axes([0.3, 0.9, 0.15, 0.05])
        bcut_keep = Button(axcut, 'Keep', color='darkgreen', hovercolor='lightgreen')
        bcut_keep.on_clicked(_keep)

        axcut = plt.axes([0.5, 0.9, 0.18, 0.05])
        bcut_mask = Button(axcut, 'Mask area(s)', color='darkgreen', hovercolor='lightgreen')
        bcut_mask.on_clicked(_mask)

        axcut = plt.axes([0.7, 0.9, 0.18, 0.05])
        bcut_choose = Button(axcut, 'Choose area(s)', color='darkgreen', hovercolor='lightgreen')
        bcut_choose.on_clicked(_choose)

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
        msg += " \n Do you want to choose/mask an other part of the coastline ? (y/n)"
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


def mkdir_if_no_exist(dir):
    if not exists(dir):
        makedirs(dir)


if __name__ == '__main__':
    # Definition of the directories we need: coastline, coastline_selection, images
    opts = Opts().parse_args()
    Input_dir = opts.Intput_dir
    Output_dir = opts.Output_dir
    images_dir = opts.Images_dir
    Shoreline_dir = Input_dir
    Shoreline_selection_dir = join(Output_dir, 'coastline_selection/')
    Shoreline_defective_dir = join(Output_dir, 'coastline_defective/')
    Shoreline_check_dir = join(Output_dir, 'coastline_check/')

    if opts.check_coastlines is not None:
        check_coastlines = True
    else:
        check_coastlines = False

    if check_coastlines:
        input_dir = opts.check_coastlines
        output_dir = Shoreline_check_dir
    else:
        input_dir = Shoreline_dir
        output_dir = Shoreline_selection_dir

    # List of all coastline images
    ls = glob(input_dir + '*/coast_A*.jpg' )
    ls = np.sort(ls)

    for f in ls:
        path = Path(f)
        print(f)

        _, name_jpg = split(f)
        f_txt = f.replace('.jpg', '.txt')
        f_txt = f_txt.replace('coast_A', 'coast_px_A')
        _, name_txt = split(f_txt)
        f_txt_out = join(output_dir, path.parent.name, name_txt)
        f_jpg_out = join(output_dir, path.parent.name, name_jpg)
        f_jpg_A = glob('{images_dir}/*/{A_jpg}'.format(images_dir=images_dir, A_jpg=name_jpg.replace('coast_A_', 'A_')))

        if check_coastlines:
            rep_check_dir = join(Shoreline_check_dir, path.parent.name)
            mkdir_if_no_exist(rep_check_dir)

            f_txt_out = join(Shoreline_check_dir, name_txt)
            shutil.copy(f_txt, f_txt_out)
            with open(f_txt) as f_coastline:
                coastline = f_coastline.readlines()
            # Plot checked shorelines
            img_A = cimg.read(f_jpg_A[0])
            draw_shoreline(img_A, coastline, f_jpg_out)

        else:
            # Plot input coastline image
            img = cimg.read(f)
            fig = show(img, display=True, moment_choice=True)
            manual_clean = choice
            # Decide if we want to throw away all the coastline, or keep it as it is, or perform a manual clean
            # msg = "Do you want to Throw away all the coastline, Keep it as it is, Remove any part(s) of the coastline \
            # , or Choose any part(s) of the coastline? (t/k/r/c)"
            # manual_clean = raw_input(msg)

            if (manual_clean == 'k') + (manual_clean == 'r') + (manual_clean == 'c'):
                # Creation of sub-folder selection/date
                rep_selection_dir = join(Shoreline_selection_dir, path.parent.name)
                mkdir_if_no_exist(rep_selection_dir)


            if (manual_clean == 'r') + (manual_clean == 'c'):

                fig = show(img, display=False)

                # Definition of the areas to be masked or to be chosen
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

                            if manual_clean == 'r':
                                # check if shoreline point is inside of any of the areas to be masked:
                                spike = False
                                for i in range(len(sl_mask)):
                                    polygon = Polygon(sl_mask[str(i)]._points)
                                    if polygon.contains(point):
                                        spike = True
                                        break

                            elif manual_clean == 'c':
                                # check if shoreline point is inside of any of the areas to be chosen:
                                spike = True
                                for i in range(len(sl_mask)):
                                    polygon = Polygon(sl_mask[str(i)]._points)
                                    if polygon.contains(point):
                                        spike = False
                                        break

                            # if ok add coastline point to clean coastline
                            if not spike:
                                f_coastline_out.write(line)

                            line = f_coastline.readline()

                with open(f_txt_out, 'r') as f_coastline_out:
                    coastline = f_coastline_out.readlines()

                # Plot cleaned shoreline
                img_A = cimg.read(f_jpg_A[0])
                draw_shoreline(img_A, coastline, f_jpg_out)

            if (manual_clean == 't') + (manual_clean == 'r') + (manual_clean == 'c'):
                # Move original coastline files to rep_defective_dir
                rep_defective_dir = join(Shoreline_defective_dir, path.parent.name)
                mkdir_if_no_exist(rep_defective_dir)
                f_txt_defect = join(rep_defective_dir, name_txt)
                f_jpg_defect = join(rep_defective_dir, name_jpg)
                shutil.move(f, f_jpg_defect)
                shutil.move(f_txt, f_txt_defect)

            elif manual_clean == 'k':
                # move directly the original coastline image and txt file to coastline_selection
                shutil.move(f, f_jpg_out)
                shutil.move(f_txt, f_txt_out)










