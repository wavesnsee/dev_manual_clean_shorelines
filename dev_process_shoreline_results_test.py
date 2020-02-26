#! /usr/bin/env python
# coding: utf8
from __future__ import absolute_import
from __future__ import unicode_literals

from os.path import join as join
from os.path import split as split
import numpy as np
import camscli.bootstrap as bs
import camscli.utils as cutils
import cams.inout as io
import cams.georef as georef
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import scipy.linalg
from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import patches
from mpl_toolkits.mplot3d import Axes3D
import datetime
from datetime import timedelta
import pytz
from scipy.interpolate import griddata
import pandas as pd
import cPickle as pk
from gstools import Stable
from gstools import Gaussian
from sympy import Plane, Point3D
from metpy.interpolate  import interpolate_to_grid
from shapely.geometry import Point, Polygon
import shapely.affinity
from descartes import PolygonPatch
from shapely.figures import SIZE, GREEN, GRAY, set_limits
import pdb
import logging
from matplotlib import pyplot


bs.init(2)
#logging.disable(logging.CRITICAL)

def get_pixel_txt_from_selected_pics(fname):
    prefix = 'coast_px_'
    file_pixel_list = []

    for f in fname:
        dir = os.path.dirname(f)
        ff = os.path.basename(f)
        ff, old_ext = os.path.splitext(ff)
        ff = 'coast_px_'+ff[6:] + '.txt'
        newname = join(dir,ff)
        file_pixel_list.append(newname)
        print newname
    return file_pixel_list


def load_pixel_line(f,utczone):
    date = cutils.get_date_in_utc_from_local_timezone(f,utczone)
    pixel_line = np.loadtxt(f)
    _, name = split(f)
    duration = (name.split('2fps_')[1]).split('s_')[0]

    return date + timedelta(seconds=int(duration)/2), pixel_line, duration


def get_average_files(dir, start, end):
    fname = [dir +  '/*' + '/coast_A_' + cam_name.upper() + '_2fps_*s_*.jpg']
    average_filelist = np.sort(cutils.glob_list(fname))
    list_files = []
    for f in average_filelist:
        date = cutils.get_date_in_utc_from_local_timezone(f, utczone)
        if (date > start) * (date < end):
            list_files.append(f)
    return list_files


def read_tide_shom(fname):
    tide_dates = []
    tide_level = []
    tide_surge = []
    water_level = []
    with open(fname, 'r') as file:
        lines = file.readlines()[1:]
        for row in lines:
            date = datetime.datetime.strptime(row.split(',')[0], '%d/%m/%Y %H:%M:%S')
            tide_dates.append(date)
            tide_level.append(float(row.split(',')[1]))
            tide_surge.append(float(row.split(',')[2]))
            water_level.append(float(row.split(',')[3]))
    return tide_dates, tide_level, tide_surge, water_level


def plot_tide(tide_dates, tide_level, tide_surge, water_level, png_out):
    f, ax = plt.subplots(figsize=(16,10))
    ax.plot(tide_dates, tide_level, 'k', markersize=3, label='tide')
    ax.plot(tide_dates, water_level, 'b', markersize=3, label='water level')
    plt.legend()
    # plt.show()
    f.savefig(png_out)


def extract_tide_level(inputdates, value, newdate):
    '''
    interpolate tide level to specific dates
    Parameters
    ----------
    date_tide
    tide_level
    newdate

    Returns
    -------
    interpolated_tide: np.array , same shape as newdate
    '''
    #convert dates into seconds since 1/1/1970
    indate =[(pytz.utc.localize(t)-datetime.datetime(1970,1,1,tzinfo=pytz.UTC)).total_seconds() for t in
             inputdates]
    outdate = [(t-datetime.datetime(1970,1,1,tzinfo= pytz.UTC)).total_seconds() for t in
             newdate]
    newvalue = np.interp(np.array(outdate),np.array(indate),value)

    return newvalue


def extract_tide_level_shift(inputdates,value, newdate, shift = 0):
    '''
    interpolate tide level to specific dates
    Parameters
    ----------
    date_tide
    tide_level
    newdate

    Returns
    -------
    interpolated_tide: np.array , same shape as newdate
    '''
    #convert dates into seconds since 1/1/1970
    #correct with phase_lag between prediction and a mreasured value
    # Add 5 mn for zlevel extraction as average image are taken out of 10 mn slots

    indate =[(t-datetime.datetime(1970,1,1,
                                  tzinfo=pytz.utc)).total_seconds()+shift*60 +300 for t in
             inputdates]
    outdate = [(t-datetime.datetime(1970,1,1,tzinfo=pytz.utc)).total_seconds() for t in
             newdate]
    newvalue = np.interp(np.array(outdate),np.array(indate),value)

    return newvalue


def read_swash(fname, utczone):
    swash_dates = []
    swash_pos = []
    # tz = pytz.timezone(utczone)

    with open(fname, 'r') as file:
        lines = file.readlines()[1:]
        for row in lines:
            date = datetime.datetime.strptime(row.split(',')[1], '%Y-%m-%d %H:%M')
            # local_date = tz.localize(date)
            # ut_date = pytz.utc.normalize(local_date)
            swash_dates.append(date)
            swash_pos.append(row.split(',')[2])

    return swash_dates, np.array(swash_pos, dtype=float)


def write_pk(average_filelist, seuil_spike):
    coast_pixel_line = get_pixel_txt_from_selected_pics(average_filelist)
    shoreline = {'date': [], 'ind_no_spikes': [], 'duration': [], 'pixline': [], 'geoline': [], 'xvalue': [],
                 'yvalue': [], 'zvalue': [], 'smooth_geoline': [], 'smooth_geoline_no_spikes': [], 'swash_pos': [],
                 'tide': [], 'water_level': []}
    datelist = []
    duration_list = []
    line_list = []
    for f in coast_pixel_line:
        date, pixel_line, duration = load_pixel_line(f, utczone)
        datelist.append(date)
        duration_list.append(duration)
        line_list.append(pixel_line)

    # tide, water level, and swash extraction at the shoreline dates
    tide_extract = extract_tide_level(tide_dates, tide_level, datelist)
    wl_extract = extract_tide_level(tide_dates, water_level, datelist)
    pos_swash_extract = extract_tide_level(swash_dates, swash_pos, datelist)

    for date, pixeline, tide, wl, duration in zip(datelist, line_list, tide_extract, wl_extract, duration_list):
        geo_line = georef.pix2geo_list(pixeline, rvec, tvec, camera_matrix,
                                       dist_coeffs,
                                       tide + zmean)

        geo_line[:, 2] -= zmean
        smooth_lines = smooth_line(geo_line, 31)

        shoreline['date'].append(date)
        shoreline['duration'].append(duration)
        shoreline['pixline'].append(pixeline)
        shoreline['geoline'].append(geo_line)
        shoreline['smooth_geoline'].append(smooth_lines)
        shoreline['tide'].append(tide)
        shoreline['water_level'].append(wl)

    # shoreline[xvalue], shoreline[yvalue], shoreline[zvalue]
    line = 'smooth_geoline'
    xvalue = []
    yvalue = []
    zvalue = []

    for i, l in enumerate(shoreline[line]):
        xvalue.extend(l[:, 0])
        yvalue.extend(l[:, 1])
        n = 0
        while n < len(l[:, 1]):
            zvalue.append(shoreline['water_level'][i])
            n += 1
    shoreline['xvalue'] = xvalue
    shoreline['yvalue'] = yvalue
    shoreline['zvalue'] = zvalue

    shoreline['smooth_geoline_no_spikes'], shoreline['ind_no_spikes'] = remove_zshifted_shoreline(shoreline, seuil_spike)

    pk.dump(shoreline, open(f_name_pickle, 'w'))
    return shoreline


def write_pk_interp(xvalue, yvalue, zvalue, xi, yi, interp_method):
    zi = griddata((xvalue, yvalue), zvalue, (xi[None, :], yi[:, None]), method=interp_method)
    pk.dump(zi, open(f_name_pickle_interp, 'w'))
    return zi


def plot_compare(x1, y1, label1, x2, y2, label2):
    f, axarr = plt.subplots(figsize=(20, 12))
    axarr.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
    f.autofmt_xdate()
    axarr.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    axarr.set_xlim([max(min(x1),min(x2)), min(max(x1), max(x2))])
    # le plot supérieur correspond au suivi de la moyenne
    axarr.grid(True)
    axarr.set_ylabel('cm')
    axarr.plot(x1, y1, '-', color='darkblue', label=label1, linewidth=2)
    axarr.plot(x2, y2, '.', color='green', label=label2, linewidth=2)
    axarr.legend(loc='best')
    return f


def remove_zshifted_shoreline(shoreline, seuil_spike):
    # Scatter points editing
    smooth_geoline_no_spikes = []
    ind_no_spikes = []
    date_spikes = []
    logger = logging.getLogger()
    log_file = join(datadir, cam_name.upper(), region + 'Analysis', 'Intertidal_topography', 'waterline_method',
'logs_edited_lines', 'log_editing_{start}_{end}_seuil_{seuil}.log'.format(
            start=start.strftime('%Y%m%d'),
            end= end.strftime('%Y%m%d'), seuil=seuil_spike))
    fhandler = logging.FileHandler(filename=log_file, mode='w')
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.DEBUG)
    for i, date in enumerate(shoreline['date']):
        print('%s spiky ?' %date)
        x = shoreline['smooth_geoline'][i][:, 0]
        y = shoreline['smooth_geoline'][i][:, 1]
        z = shoreline['smooth_geoline'][i][0, 2]
        indx = np.where(abs(np.median(x) - x) == min(abs(np.median(x) - x)))[0][0]
        indy = np.where(abs(np.median(y) - y) == min(abs(np.median(y) - y)))[0][0]
        xc = x[indx]
        yc = y[indy]
        # Median values inside ellipse
        xvalue_in_ell = []
        yvalue_in_ell = []
        zvalue_in_ell = []
        # semi_minor = 8
        # semi_major= 50
        semi_minor = 6
        semi_major = 35
        rot = 135
        ell = create_ellipse((xc,yc), (semi_minor, semi_major), rot)

        for p in range(len(shoreline['xvalue'])):
            if Point(shoreline['xvalue'][p], shoreline['yvalue'][p]).within(ell):
                xvalue_in_ell.append(shoreline['xvalue'][p])
                yvalue_in_ell.append(shoreline['yvalue'][p])
                zvalue_in_ell.append(shoreline['zvalue'][p])
        median = np.median(zvalue_in_ell)

        spiky_day = datetime.datetime(date.year, date.month, date.day) in date_spikes
        if (abs(z - median) < seuil_spike) * (not spiky_day):
            smooth_geoline_no_spikes.append(shoreline['smooth_geoline'][i])
            ind_no_spikes.append(i)

        else:
            logging.info('Edited shoreline on %s, with seuil=%s, duration=%s s, and z -z_median=%s !'
                         % (date, seuil_spike, shoreline['duration'][i], z - median))
            print('-> Yes, Spiky shoreline on %s, with z -z_median=%s !' % (date, z - median))
            date_spikes.append(datetime.datetime(date.year, date.month, date.day))

            # fig = pyplot.figure(figsize=(18,7))
            # ax = fig.add_subplot(121)
            # patch = PolygonPatch(ell, fc=GREEN, ec=GRAY, alpha=0.5, zorder=2)
            # ax.add_patch(patch)
            # pyplot.xlim((xmin, xmax))
            # pyplot.ylim((ymin, ymax))
            # ax.scatter(shoreline['xvalue'], shoreline['yvalue'], c='lightgray',s=1)
            # ax.scatter(x, y, c='darkblue',s=1)
            # ax = fig.add_subplot(122)
            # pyplot.xlim((xmin, xmax))
            # pyplot.ylim((ymin, ymax))
            # scat = ax.scatter(xvalue_in_ell, yvalue_in_ell, c=zvalue_in_ell, cmap='viridis', vmin=-4, vmax=4, s=0.5)
            # pyplot.title('Z in considered shoreline: %.2f, Median z in ellipse: %.2f' %(z, np.median(zvalue_in_ell)))
            # pyplot.colorbar(scat)
            # pyplot.show()

    return smooth_geoline_no_spikes, ind_no_spikes



def curve_fitting(xvalue, yvalue, zvalue, X, Y, order):
    data = np.column_stack((xvalue, yvalue, zvalue))
    XX = X.flatten()
    YY = Y.flatten()

    if order == 1:
        # best-fit linear plane
        A = np.c_[data[:, 0], data[:, 1], np.ones(data.shape[0])]
        C, _, _, _ = scipy.linalg.lstsq(A, data[:, 2])  # coefficients

        # evaluate it on grid
        Z = C[0] * X + C[1] * Y + C[2]

        # or expressed using matrix/vector product
        # Z = np.dot(np.c_[XX, YY, np.ones(XX.shape)], C).reshape(X.shape)

    elif order == 2:
        # best-fit quadratic curve
        A = np.c_[np.ones(data.shape[0]), data[:, :2], np.prod(data[:, :2], axis=1), data[:, :2] ** 2]
        C, _, _, _ = scipy.linalg.lstsq(A, data[:, 2])

        # evaluate it on a grid
        Z = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX * YY, XX ** 2, YY ** 2], C).reshape(X.shape)

    # plot points and fitted surface
    # fig = go.Figure(go.Surface(
    #     x=xi,
    #     y=yi,
    #     z=Z,
    #     cmin=-4,
    #     cmax=4
    # ))
    # fig.update_layout(
    #     width=1000,
    #     height=800,
    #     scene={
    #         "xaxis": {"nticks": 5, "range": [emprise[0], emprise[1]]},
    #         "yaxis": {"nticks": 5, "range": [emprise[2], emprise[3]]},
    #         "zaxis": {"nticks": 5, "range": [-4, 4]},
    #         'camera_eye': {"x": 0.9, "y": 1.1, "z": 0.5},
    #         "aspectratio": {"x": 1, "y": 1, "z": 0.2}
    #     },
    #     paper_bgcolor="LightSteelBlue")
    # fig.add_scatter3d(x=xvalue, y=yvalue, z=zvalue,
    #                   mode='markers',
    #                   marker = dict(size=2, color=zvalue, colorscale='Viridis',
    #                                 cmin=-4, cmax=4, opacity=0.8,
    #                   line=dict(width=0), colorbar=dict(thickness=20)))
    # fig.show()

    #fig = plt.figure()
    #ax = fig.gca(projection='3d')
    #ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)
    #ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='r', s=50)
    #plt.xlabel('X')
    #plt.ylabel('Y')
    #ax.set_zlabel('Z')
    #ax.axis('equal')
    #ax.axis('tight')
    #plt.show()

    return Z


def curve_fitting_local(xvalue, yvalue, zvalue, X, Y, order):
    Z = curve_fitting(xvalue, yvalue, zvalue, X, Y, order)

    # plot points and fitted surface
    fig = go.Figure(go.Surface(
        x=xi_chunk,
        y=yi_chunk,
        z=Z,
        cmin=-5,
        cmax=5
    ))
    fig.update_layout(
        width=1000,
        height=800,
        scene={
            "xaxis": {"nticks": 5, "range": [emprise[0], emprise[1]]},
            "yaxis": {"nticks": 5, "range": [emprise[2], emprise[3]]},
            "zaxis": {"nticks": 5, "range": [-5, 5]},
            'camera_eye': {"x": 0.9, "y": 1.1, "z": 0.5},
            "aspectratio": {"x": 1, "y": 1, "z": 0.2}
        },
        paper_bgcolor="LightSteelBlue")
    fig.add_scatter3d(x=xvalue, y=yvalue, z=zvalue,
                      mode='markers',
                      marker = dict(size=1, color=zvalue, colorscale='Viridis',
                                    cmin=-5, cmax=5, opacity=0.8,
                      line=dict(width=0), colorbar=dict(thickness=20)))
    # fig.show()
    return Z


def create_ellipse(coords, axes_values, rotation):
    # 1st elem = center point (x,y) coordinates
    # 2nd elem = the two semi-axis values (along x, along y)
    # 3rd elem = angle in degrees between x-axis of the Cartesian base
    #            and the corresponding semi-axis
    ellipse = (coords, axes_values, rotation)

    # Let create a circle of radius 1 around center point:
    circ = shapely.geometry.Point(ellipse[0]).buffer(1)

    # Let create the ellipse along x and y:
    ell = shapely.affinity.scale(circ, int(ellipse[1][0]), int(ellipse[1][1]))

    # Let rotate the ellipse (clockwise, x axis pointing right):
    ellr = shapely.affinity.rotate(ell, ellipse[2])


    # fig = plotly_2d_scatter_plot(xvalue, yvalue, zvalue, emprise)

    # ax = fig.add_subplot(111)
    # patch = PolygonPatch(elrv, fc=GREEN, ec=GRAY, alpha=0.5, zorder=2)
    # ax.add_patch(patch)
    # set_limits(ax, -10, 10, -10, 10)
    # pyplot.show()
    return ellr

def Ordinary_Kriging(xvalue, yvalue, zvalue, xi, yi):
    #data = np.column_stack((xvalue, yvalue, zvalue))

    # Create the ordinary kriging object. Required inputs are the X-coordinates of the data points, the Y-coordinates of
    # the data points, and the Z-values of the data points. Variogram is handled as in the ordinary kriging case.
    # drift_terms is a list of the drift terms to include; currently supported terms are 'regional_linear', 'point_log',
    # and 'external_Z'.  Refer to UniversalKriging.__doc__ for more information.

    # cov_model = Gaussian(dim=2, len_scale=4, anis=.2, angles=-.5, var=.5, nugget=.1)
    # cov_model = 'gaussian'
    cov_model = Stable(dim=2, len_scale=1, anis=0.2, angles=-0.5, var=0.5, nugget=0.1)

    UK = OrdinaryKriging(xvalue, yvalue, zvalue, cov_model)

    # UK = OrdinaryKriging(xvalue, yvalue, zvalue, variogram_model='exponential', verbose=True, enable_plotting=False)
    #UK = UniversalKriging(xvalue, yvalue, zvalue, variogram_model='linear',drift_terms=['regional_linear'])

    # Creates the kriged grid and the variance grid. Allows for kriging on a rectangular grid of points, on a masked
    # rectangular grid of points, or with arbitrary points. (See UniversalKriging.__doc__ for more information.)
    z, var = UK.execute('grid', xi, yi)
    return z, var


def plotly_3d_scatter_plot(shoreline, emprise, var, line):
    layout = go.Layout(
        width=1000,
        height=800,
        title_text="Detected shorelines:",
        scene=dict(
            xaxis=dict(title='Easting (m)', titlefont=dict(size=18)),
            yaxis=dict(title='Northing (m)', titlefont=dict(size=18)),
            zaxis=dict(title='Height (m)', titlefont=dict(size=18)),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.30)))

    fig = go.Figure(layout=layout)
    if line == 'smooth_geoline':
        N = len(shoreline['date'])
        shoreline_date = shoreline['date']
        shoreline_water_level = shoreline['water_level']
        shoreline_dur = shoreline['duration']
        shoreline_tide = shoreline['tide']
        shoreline_var = shoreline[var]
    elif line == 'smooth_geoline_no_spikes':
        N = len(np.array(shoreline['date'])[shoreline['ind_no_spikes']])
        shoreline_date = np.array(shoreline['date'])[shoreline['ind_no_spikes']]
        shoreline_water_level = np.array(shoreline['water_level'])[shoreline['ind_no_spikes']]
        shoreline_dur = np.array(shoreline['duration'])[shoreline['ind_no_spikes']]
        shoreline_tide = np.array(shoreline['tide'])[shoreline['ind_no_spikes']]
        shoreline_var = np.array(shoreline[var])[shoreline['ind_no_spikes']]

    for i in range(N):
        d = {'x': shoreline[line][i][:, 0],
             'y': shoreline[line][i][:, 1],
             'z': shoreline_var[i] * np.ones(np.shape(shoreline[line][i][:, 0])),
             'marker_size': 2
             }

        df = pd.DataFrame(d)

        fig.add_trace(go.Scatter3d(
            x=df['x'],
            y=df['y'],
            z=df['z'],
            name=shoreline_date[i].strftime('%Y/%m/%d/ %H:%M:%S'),
            hovertemplate='water level: {wl}, '.format(wl=format(shoreline_water_level[i], '.2f')) +
            'duration: {dur}s, '.format(dur= shoreline_dur[i]) + 'surge: {surge} m '\
                              .format(surge=format(shoreline_water_level[i] - shoreline_tide[i],'.2f')),
            showlegend=False,
            mode='markers',
            marker=dict(size=df["marker_size"], color=df['z'], colorscale='Viridis', cmin=-4, cmax=4, opacity=0.8,
                        line=dict(width=0), colorbar=dict(thickness=20)),
        ))
        fig.update_layout(
            scene=dict(
            xaxis=dict(nticks=5, range=[emprise[0], emprise[1]], ),
            yaxis=dict(nticks=5, range=[emprise[2], emprise[3]], ),
            zaxis=dict(nticks=5, range=[-4, 4], ),
            camera_eye=dict(x=0.9,y=1.1, z=0.5),
        ))

    return fig


def plotly_2d_scatter_plot(xvalue, yvalue, zvalue, emprise):
    fig = go.Figure(data=go.Scatter(
        x = xvalue,
        y = yvalue,
        mode='markers',
        marker=dict(
            size=2,
            color=zvalue,
            colorscale='Viridis',  # one of plotly colorscales
            showscale=True
        )
    ))
    fig.update_layout(
    scene=dict(
        xaxis=dict(nticks=5, range=[emprise[0], emprise[1]], ),
        yaxis=dict(nticks=5, range=[emprise[2], emprise[3]], ),
    ),
    width = 1000,
    height = 800,
    paper_bgcolor="LightSteelBlue",
    )
    # fig.update_yaxes(automargin=True)
    return fig


def plotly_surface_plot(xi, yi, zi, emprise):
    fig = go.Figure(go.Surface(
                    x = xi,
                    y = yi,
                    z = zi,
                    cmin=-4,
                    cmax=4
                ))
    fig.update_layout(
        width = 1000,
        height = 800,
        title_text="Intertidal bathymetry derived from detected shorelines:",
        scene={
            "xaxis": {"nticks": 5, "range": [emprise[0], emprise[1]], "title": 'Easting(m)', "titlefont": {'size': 40}},
            "yaxis": {"nticks": 5, "range": [emprise[2], emprise[3]], "title": 'Northing(m)', "titlefont": {'size': 20}},
            "zaxis": {"nticks": 5, "range": [-4, 4], "title": 'Height(m)', "titlefont": {'size': 20}},
            'camera_eye': {"x": 0.9, "y":1.1, "z": 0.5},
            "aspectratio": {"x": 1, "y": 1, "z": 0.3}
        },
        paper_bgcolor="LightSteelBlue")
    return fig


def subplot_scat_contour_pcolor(xvalue, yvalue, zvalue, xi, yi, zi, emprise):
    contour_step = 0.25
    fig, axs = plt.subplots(2, 2, figsize=(20, 12))
    ax = axs[0, 0]
    marker_size = 0.1
    scat = ax.scatter(xvalue, yvalue, s=marker_size, c=zvalue, alpha=0.5, edgecolors='face', cmap="RdYlBu_r")
    ax.set_title("Points of detected coastlines")
    ax.set_xlim([emprise[0], emprise[1]])
    ax.set_ylim([emprise[2], emprise[3]])
    fig.colorbar(scat, ax=ax)

    ax = axs[0, 1]
    ax.contour(xi, yi, zi, levels=np.arange(-4, 4, contour_step), linewidths=0.5, colors='k')
    cntr = ax.contourf(xi, yi, zi, levels=np.arange(-4, 4, contour_step), cmap="RdYlBu_r")
    fig.colorbar(cntr, ax=ax)
    ax.set_xlim([emprise[0], emprise[1]])
    ax.set_ylim([emprise[2], emprise[3]])
    ax.set_title('contour')

    ax = axs[1, 0]
    pcol = ax.pcolor(Xi, Yi, np.ma.array(data=zi, mask=(np.isnan(zi))), cmap="RdYlBu_r", vmin=-4, vmax=4)
    fig.colorbar(pcol, ax=ax)
    ax.set_xlim([emprise[0], emprise[1]])
    ax.set_ylim([emprise[2], emprise[3]])
    plt.subplots_adjust(hspace=0.1)

    return fig


def check_shorelines_phase_diff(shoreline, line, tide_dates, water_level):
    sorted_date = np.sort(shoreline['date'])
    for ind_t, t in enumerate(sorted_date):
        if ((ind_t >= 2) * (ind_t < len(shoreline['date']) - 2)):

            ind_t_m2 = np.where(np.asarray(shoreline['date']) == sorted_date[ind_t - 2])[0][0]
            ind_t_m1 = np.where(np.asarray(shoreline['date']) == sorted_date[ind_t - 1])[0][0]
            ind_t0 = np.where(np.asarray(shoreline['date']) == t)[0][0]
            ind_t_p1 = np.where(np.asarray(shoreline['date']) == sorted_date[ind_t + 1])[0][0]
            ind_t_p2 = np.where(np.asarray(shoreline['date']) == sorted_date[ind_t + 2])[0][0]

            fig = make_subplots(rows=2, cols=1)
            layout = go.Layout(
                width=1024,
                height=1024
            )

            fig.add_trace(
                go.Scatter(
                    x=shoreline[line][ind_t_m2][:, 0],
                    y=shoreline[line][ind_t_m2][:, 1],
                    name=shoreline['date'][ind_t_m2].strftime("%Y/%m/%d, %H:%M:%S"),
                    line=dict(color='lightgray', width=2)
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=shoreline[line][ind_t_m1][:, 0],
                    y=shoreline[line][ind_t_m1][:, 1],
                    name=shoreline['date'][ind_t_m1].strftime("%Y/%m/%d, %H:%M:%S"),
                    line=dict(color='darkgray', width=2)
                ),
                row=1, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=shoreline[line][ind_t0][:, 0],
                    y=shoreline[line][ind_t0][:, 1],
                    name=shoreline['date'][ind_t0].strftime("%Y/%m/%d, %H:%M:%S"),
                    line=dict(color='black', width=2)
                ),
                row=1, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=shoreline[line][ind_t_p1][:, 0],
                    y=shoreline[line][ind_t_p1][:, 1],
                    name=shoreline['date'][ind_t_p1].strftime("%Y/%m/%d, %H:%M:%S"),
                    line=dict(color='steelblue', width=2)
                ),
                row=1, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=shoreline[line][ind_t_p2][:, 0],
                    y=shoreline[line][ind_t_p2][:, 1],
                    name=shoreline['date'][ind_t_p2].strftime("%Y/%m/%d, %H:%M:%S"),
                    line=dict(color='lightblue', width=2)
                ),
                row=1, col=1
            )

            fig.update_xaxes(range=[500, 3000], row=1, col=1)
            fig.update_yaxes(range=[1050, 1350], row=1, col=1)

            fig.add_trace(
                go.Scatter(
                    x=[shoreline['date'][ind_t_m2]],
                    y=[shoreline['water_level'][ind_t_m2]],
                    mode='markers',
                    marker=dict(color='lightgray', size=20)
                ),
                row=2, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=[shoreline['date'][ind_t_m1]],
                    y=[shoreline['water_level'][ind_t_m1]],
                    mode='markers',
                    marker=dict(color='darkgray', size=20)
                ),
                row=2, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=[shoreline['date'][ind_t0]],
                    y=[shoreline['water_level'][ind_t0]],
                    mode='markers',
                    marker=dict(color='black', size=20)
                ),
                row=2, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=[shoreline['date'][ind_t_p1]],
                    y=[shoreline['water_level'][ind_t_p1]],
                    mode='markers',
                    marker=dict(color='steelblue', size=20)
                ),
                row=2, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=[shoreline['date'][ind_t_p2]],
                    y=[shoreline['water_level'][ind_t_p2]],
                    mode='markers',
                    marker=dict(color='lightblue', size=20)
                ),
                row=2, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=tide_dates,
                    y=water_level,
                    name='Water level (m)',
                    line=dict(color='black', width=1, dash='dash')
                ),
                row=2, col=1
            )
            fig.update_xaxes(title_text="Time", \
                             range=[shoreline['date'][ind_t0] - timedelta(0.25),
                                    shoreline['date'][ind_t0] + timedelta(0.25)], row=2, col=1
                             )
            png = join(datadir, cam_name.upper(), region + 'Analysis', 'Intertidal_topography', 'waterline_method', \
                       'check_sl_phi_diff_with_wl', 'sl_{line}/check_{date}.png'.format(line=line, \
                                                                                        date=shoreline['date'][
                                                                                            ind_t0].strftime(
                                                                                            "%Y%m%d_%H_%M")))
            if line == 'pixline':
                precision = 'pixel coordinates'
            else:
                precision = 'utm coordinates'

            fig.update_layout(title_text="Comparison of shoreline positions ({precision}) with water level".
                              format(precision=precision), width=1021, height=1024)
            fig.write_image(png)


def filter_dict(dict,start, end):
    """ Extract a subdictionnary between dates , maximum distance and depth"""
    dates = np.array(dict['date'])
    dist = np.array(dict['dist'])
    pixline = np.array(dict['pixline'])
    geoline = np.array(dict['geoline'])
    smooth_line = np.array(dict['smooth_geoline'])
    # Filter through dates
    date_condition = (dates>=start) & (dates<=end)
    newdates = dates[date_condition]
    newdist = dist[date_condition]
    newpixline = pixline[date_condition]
    newgeoline = geoline[date_condition]
    newsmooth = smooth_line[date_condition]
    newdict = {}
    newdict['date'] = newdates
    newdict['dist'] = newdist
    newdict['pixline'] = newpixline
    newdict['geoline'] = newgeoline
    newdict['smooth_geoline'] = newsmooth
    return newdict


def filter_dict_key(dict,keyname,minvalue, maxvalue):
    """ Extract a subdictionnary between dates , maximum distance and depth"""
    A = np.array(dict[keyname])
    condition = (A>=minvalue) & (A<=maxvalue)
    newdict = {}
    for key in dict.keys():
        B = np.array(dict[key])
        newdict[key] = B[condition]
    return newdict


def filter_dict_key_cond(dict,condition):
    """ Extract a subdictionnary between dates , maximum distance and depth"""
    newdict = {}
    for key in dict.keys():
        B = np.array(dict[key])
        newdict[key] = B[condition]
    return newdict


def get_rvalue(z,maxz,minz):
    return (z-minz)/(maxz-minz)


def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        #raise ValueError, "Input vector needs to be bigger than window size."
        window_len = window_len / 2 + 1

    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]

    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y


def smooth_line(line,window_len=11):
    zz = line[:,2].mean()
    new_y = smooth(line[:,1],window_len,window='hanning')
    new_z = zz*np.ones(new_y.shape[0])
    new_x = smooth(line[:,0],window_len,window='hanning')
    newline = np.stack([new_x, new_y,new_z], axis=-1)
    return newline


def find_outliers_rollingmedian(distance, value, window_size=10):
    counter = 0
    indicator = np.zeros(distance.shape)
    for d in distance:
        # find closest value
        indx = np.argsort(np.abs(distance - d))[:window_size]
        dd = np.abs(value[indx] - np.median(value[indx]))/window_size
        mdev = np.median(dd)
        s = dd / mdev if mdev else 0.
        indicator[indx] += dd
        counter += 1
    return indicator


def removeoutliers(dict_lines, window_size=10, quantile=80):
    dist = dict_lines['dist']
    dista = np.array(dist)
    Zlist = [z[:,2].mean() for z in dict_lines['geoline']]
    zlista = np.array(Zlist)
    dispersion_indicator = find_outliers_rollingmedian(dista,zlista,window_size)
    # Compute 80 centile
    q10 = np.percentile(dispersion_indicator,quantile)
    cond1 = dispersion_indicator <= q10
    dictlines_filtered = filter_dict_key_cond(dict_lines, cond1)
    return dispersion_indicator,q10,dictlines_filtered



# Period
utczone = 'Europe/Paris'
utc = pytz.UTC
cycle_ve_me = 29
start_date = utc.localize(datetime.datetime(2018,6,28))
period = {
    'Periode1' : (start_date, start_date + 3*timedelta(cycle_ve_me)),
    'Sous_Periodes1' : ((start_date, start_date + timedelta(cycle_ve_me)),
    (start_date + timedelta(cycle_ve_me), start_date + 2* timedelta(cycle_ve_me)),
    (start_date + 2 * timedelta(cycle_ve_me), start_date + 3 * timedelta(cycle_ve_me))),
    'Periode2' : (start_date + 3*timedelta(cycle_ve_me), start_date + 6*timedelta(cycle_ve_me)),
    'Sous_Periodes2' : ((start_date + 3*timedelta(cycle_ve_me), start_date + 4*timedelta(cycle_ve_me)),
    (start_date + 4*timedelta(cycle_ve_me), start_date + 5* timedelta(cycle_ve_me)),
    (start_date + 5 * timedelta(cycle_ve_me), start_date + 6 * timedelta(cycle_ve_me))),
    'Periode3' : (start_date + 6*timedelta(cycle_ve_me), start_date + 9*timedelta(cycle_ve_me)),
    'Periode4' : (start_date + 9*timedelta(cycle_ve_me), start_date + 12*timedelta(cycle_ve_me)),
    'Periode5' : (start_date + 15*timedelta(cycle_ve_me), start_date + 18*timedelta(cycle_ve_me)),
    'Periode6' : (start_date + 21*timedelta(cycle_ve_me), start_date + 24*timedelta(cycle_ve_me)),
}

# period = {
#     'Sous_Periodes2' : ((utc.localize(datetime.datetime(2018,10,1)), utc.localize(datetime.datetime(2018,10,20))),
#     (start_date + 4*timedelta(cycle_ve_me), start_date + 5* timedelta(cycle_ve_me)),
#     (start_date + 5 * timedelta(cycle_ve_me), start_date + 6 * timedelta(cycle_ve_me))),
# }

# options d'execution/affichage
write_pickle = True
write_pickle_interp = False
compare_swash_tide = False
check_phase_diff_shoreline_tide = False
matplotlib_subplot = False
plotly_3d = True
plotly_2d = False
plotly_surface = False


# data
region = 'Etretat'
# phaselist = ['Periode1', 'Periode2']
# phaselist = ['Sous_Periodes1', 'Sous_Periodes2']
# phaselist = ['Periode2']
# phaselist = ['Periode3']
# phaselist = ['Sous_Periodes1']
phaselist = ['Sous_Periodes2']
datadir ='/home/florent/Projects/Etretat/'


# info files (calibration, roi, georef)
cam_name = 'cam1'
calib_file = join(datadir, cam_name.upper() ,'info/calibration/calibration_{0}.json'.format(cam_name))
roi_file = join(datadir, cam_name.upper(), 'info/roi/roi_large.txt')
pos_file = join(datadir, cam_name.upper(), 'info/georef_compute_int_ransac/georef_ransac.json')
zmean = 0.0
calib = io.read_json_to_dict(calib_file)
dist_coeffs = calib["dist_coeffs"]
camera_matrix = calib["camera_matrix"]
imshape = (calib['width'], calib['height'])
pose = io.read_json_to_dict(pos_file)
rvec = pose["rvec"]
tvec = pose["tvec"]
utmzone = pose["utm"]
camera_pose = georef.camera_position(rvec,tvec)


# Coastline files
wl_dir = join(datadir, cam_name.upper(), region+'Analysis', 'Shoreline_manual_clean', 'ratio_white', \
                      'coastline_selection')

# Tide file
tide_file = join(datadir, 'etretat_water_level.csv')
tide_dates, tide_level, tide_surge, water_level = read_tide_shom(tide_file)
file_png = '/home/florent/test.png'
# plot_tide(tide_dates, tide_level, tide_surge, water_level, file_png)
maxz = max(water_level)
minz = min(water_level)


# Swash file
swash_file = join(datadir, cam_name.upper(), region+'Analysis', 'swash_pos_{cam}_C1.csv'.format(cam=cam_name.upper()))
swash_dates, swash_pos = read_swash(swash_file, utczone)
if compare_swash_tide:
    # Comparison swash file and tide file
    f = plot_compare(tide_dates, tide_level, 'tide', swash_dates, (swash_pos-np.mean(swash_pos))/50 -7.5, 'swash_pos')
    png = join(datadir, cam_name.upper(), region + 'Analysis', 'Intertidal_topography', 'waterline_method','comparison_\
    swash_tide.png')
    f.savefig(png)


start = 0
end = 0


for phase_key in phaselist:
    p = 0
    if 'Sous_Periodes' in phase_key:
        last_period = max(max((period[phase_key])))
    else:
        last_period = max((period[phase_key]))
    while end != last_period:

        if 'Sous_Periode' in phase_key:
            start, end = period[phase_key][p]
        else:
            start, end = period[phase_key]
        print(start)
        print(end)
        start_str = start.strftime("%Y%m%d")
        end_str = end.strftime("%Y%m%d")

        ###
        # creation pickle shoreline_{period}.pk that contains all data (shorelines, shorelines whithout spiky ones,
        # dates, water level)
        ###
        # seuil_spike = 1.0
        seuil_spike = 0.85
        # Homogeneous optimums for all plots
        xmin = 298075.0
        xmax = 298285.0
        ymin = 5509945.0
        ymax = 5510125.0
        emprise = [xmin, xmax, ymin, ymax]
        f_name_pickle = join(datadir, cam_name.upper(), region + 'Analysis', 'Intertidal_topography',
                             'waterline_method', 'pickle',
                             'dict_' + cam_name + '_' + phase_key + '_' + start_str + '_' + end_str + '.pk')
        if write_pickle:
            average_filelist = get_average_files(wl_dir, start, end)
            shoreline = write_pk(average_filelist, seuil_spike)
        else:
            shoreline = pk.load(open(f_name_pickle, 'r'))
        p += 1


        ################
        # Plots
        ################

        # plotly 3D scatter plot
        if plotly_3d:
            for line in ['smooth_geoline', 'smooth_geoline_no_spikes']:
                fig = plotly_3d_scatter_plot(shoreline ,emprise, var='water_level', line=line)
                png = join(datadir, cam_name.upper(), region + 'Analysis', 'Intertidal_topography', 'waterline_method',
    '3d_scatter_plot_{phase_key}_{period}_{line}_{seuil}.png'.format(phase_key=phase_key,
                                                period=start_str + '_' + end_str, line=line, seuil=seuil_spike))
                fig.write_image(png)
                plotly.offline.plot(fig, filename=png.replace('.png', '.html'), auto_open=False)


        # Creation of vectors xvalue, yvalue, zvalue
        line = 'smooth_geoline'
        xvalue = []
        yvalue = []
        zvalue = []

        for i,l in enumerate(shoreline[line]):
            xvalue.extend(l[:, 0])
            yvalue.extend(l[:, 1])
            n=0
            while n < len(l[:, 1]):
                zvalue.append(shoreline['water_level'][i])
                n += 1




        ## divide xi and yi arrays into chunks
        # n_chunks = 5
        # xi_chunks = np.split(xi, n_chunks)
        # yi_chunks = np.split(yi, n_chunks)
        #
        # for xi_chunk in xi_chunks:
        #     for yi_chunk in yi_chunks:
        #         Xi_chunk, Yi_chunk = np.meshgrid(xi_chunk, yi_chunk)
        #         coords_chunk = [(min(xi_chunk), min(yi_chunk)), (max(xi_chunk), min(yi_chunk)),
        #                         (max(xi_chunk), max(yi_chunk)), (min(xi_chunk), max(yi_chunk)),
        #                                                         (min(xi_chunk), min(yi_chunk))]
        #         poly = Polygon(coords_chunk)
        #         # Find points(xvalue, yvalue) inside chunck area:
        #         xvalue_in_chunk = []
        #         yvalue_in_chunk = []
        #         zvalue_in_chunk = []
        #         for p in range(len(xvalue)):
        #             # Editing_from_curve_fitting(xvalue, yvalue, zvalue, Xi, Yi, order=2)
        #             if Point(xvalue[p], yvalue[p]).within(poly):
        #                 xvalue_in_chunk.append(xvalue[p])
        #                 yvalue_in_chunk.append(yvalue[p])
        #                 zvalue_in_chunk.append(zvalue[p])
        #
        #         Z = curve_fitting_local(xvalue_in_chunk, yvalue_in_chunk, zvalue_in_chunk, Xi_chunk, Yi_chunk, order=1)
                # Distance from


        # plotly 2D scatter plot
        if plotly_2d:
            fig = plotly_2d_scatter_plot(xvalue, yvalue, zvalue, emprise)
            png = join(datadir, cam_name.upper(), region + 'Analysis', 'Intertidal_topography', 'waterline_method',\
'2d_scatter_plot_{phase_key}_{period}.png'.format(phase_key=phase_key, period=start_str + '_' + end_str))
            fig.write_image(png)
            plotly.offline.plot(fig, filename=png.replace('.png', '.html'), auto_open=False)


        #check_shorelines_phase_diff with tide data
        if check_phase_diff_shoreline_tide:
            line = 'pixline'
            check_shorelines_phase_diff(shoreline, line, tide_dates, water_level)

        ###
        # Grid Interpolation
        ###

        xvalue = xvalue[::30]
        yvalue = yvalue[::30]
        zvalue = zvalue[::30]
        gridstep = 2
        xi = np.arange(xmin, xmax, gridstep)
        yi = np.arange(ymin, ymax, gridstep)
        Xi, Yi = np.meshgrid(xi, yi)

        # Linear interpolation
        #curve_fitting(xvalue, yvalue, zvalue, Xi, Yi, order=2)

        interp_methods = ['linear', 'cubic']

        for interp_method in interp_methods:
            f_name_pickle_interp = join(datadir, cam_name.upper(), region + 'Analysis', 'Intertidal_topography',
                                        'waterline_method', 'pickle',
                                        'dict_Interp_' + interp_method + '_' + cam_name + '_' + phase_key + '_' +
                                        start_str + '_' + end_str + '.pk')
            if write_pickle_interp:
                zi = write_pk_interp(xvalue, yvalue, zvalue, xi, yi, interp_method)
            else:
                zi = pk.load(open(f_name_pickle_interp, 'r'))


            ###
            ## matplotlib scatter, contour(f), pcolor
            ###
            if matplotlib_subplot:
                fig = subplot_scat_contour_pcolor(xvalue, yvalue, zvalue, xi, yi, zi, emprise)
                png = join(datadir, cam_name.upper(), region + 'Analysis', 'Intertidal_topography', 'waterline_method',
        'matplotlib_plot_{phase_key}_{period}_interp_{interp}.png'.format(
                phase_key=phase_key, period=start_str + '_' + end_str, interp=interp_method))
                plt.savefig(png)


            ###
            ## plotly surface plot
            ###
            if plotly_surface:
                fig = plotly_surface_plot(xi, yi, zi, emprise)

                png = join(datadir, cam_name.upper(), region + 'Analysis', 'Intertidal_topography', 'waterline_method',
    'surface_plot_{phase_key}_{period}_interp_{interp}.png'\
                        .format(phase_key=phase_key, period=start_str + '_' + end_str, interp=interp_method))
                fig.write_image(png)
                plotly.offline.plot(fig, filename=png.replace('.png', '.html'), auto_open=False)


        # Interpolation Tests: Kriging, Metpy (Barnes, Cressman)

        krig = 'OrdinaryKriging'
        xshift = 298075
        yshift = 5509945
        xvalue = np.array(xvalue) - xshift
        yvalue = np.array(yvalue) - yshift
        zvalue = np.array(zvalue)
        xi = xi -xshift
        yi = yi -yshift

        # Tests Kriging
        # zi,var = Ordinary_Kriging(xvalue, yvalue, zvalue, xi, yi)
        # xi, yi, zi = interpolate_to_grid(xvalue, yvalue, zvalue, interp_type='rbf', hres=2, rbf_func='linear')

        # Tests Metpy:
        # xi, yi, zi = interpolate_to_grid(xvalue, yvalue, zvalue, interp_type='linear', hres=2)
        # xi, yi, zi = interpolate_to_grid(xvalue, yvalue, zvalue, interp_type='natural_neighbor', hres=2)
        xi, yi, zi = interpolate_to_grid(xvalue, yvalue, zvalue, interp_type='cressman', minimum_neighbors=1,
                                  hres=2, search_radius=3)
        # xi, yi, zi = interpolate_to_grid(xvalue, yvalue, zvalue, interp_type='barnes', hres=2,
        #                     search_radius=3)
        # xi, yi, zi = interpolate_to_grid(xvalue, yvalue, zvalue, interp_type='rbf', hres=2, rbf_func='gaussian',
        #                     rbf_smooth=0)

        xi = xi + xshift
        yi = yi + yshift


        xvalue = np.array(xvalue) + xshift
        yvalue = np.array(yvalue) + yshift

        fig, axs = plt.subplots(1, 2, figsize=(18, 5))
        # ax = axs[0,0]
        ax = axs[0]
        ax.axis('equal')
        scat = ax.scatter(xvalue, yvalue, s=8, c=zvalue, alpha=0.5, edgecolors='face', cmap="RdYlBu_r", vmin=-4, vmax=4)
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        fig.colorbar(scat, ax=ax)

        # ax = axs[0,1]
        ax = axs[1]
        ax.axis('equal')
        # pcol = ax.pcolor(Xi, Yi, zi, cmap="RdYlBu_r", vmin=-4, vmax=4)
        pcol = ax.pcolor(xi, yi, zi, cmap="RdYlBu_r", vmin=-4, vmax=4)
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        fig.colorbar(pcol, ax=ax)

        # ax = axs[1,0]
        # pcol = ax.pcolor(Xi, Yi, var, cmap="RdYlBu_r")
        # ax.set_xlim([xmin, xmax])
        # ax.set_ylim([ymin, ymax])
        # fig.colorbar(pcol, ax=ax)

        # plt.show()