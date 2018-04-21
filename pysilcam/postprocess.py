import pandas as pd
import numpy as np
import os
import imageio as imo
import matplotlib.pyplot as plt
from skimage.filters.rank import median
from skimage.morphology import disk
import skimage
import pysilcam.process as scpr
from scipy import ndimage as ndi
import skimage
from skimage.exposure import rescale_intensity
import h5py
from pysilcam.config import PySilcamSettings
from enum import Enum

class outputPartType(Enum):
    '''
    Enum for all (1), oil (2) or gas (3)
    '''
    all = 1
    oil = 2
    gas = 3

def d50_from_stats(stats, settings):
    '''
    Calculate the d50 from the stats and settings
    
    Args:
        stats (DataFrame)           : particle statistics from silcam process
        settings (PySilcamSettings) : settings associated with the data, loaded with PySilcamSettings
        
    Returns:
        d50 (float)                 : the 50th percentile of the cumulative sum of the volume distributon, in microns
    '''

    # the volume distribution needs calculating first
    dias, vd = vd_from_stats(stats, settings)

    # then the d50
    d50 = d50_from_vd(vd,dias)
    return d50

def d50_from_vd(vd,dias):
    '''
    Calculate d50 from a volume distribution
    
    Args:
        vd (array)           : particle volume distribution calculated from vd_from_stats()
        dias (array)         : mid-points in the size classes corresponding the the volume distribution,
                               returned from get_size_bins()
        
    Returns:
        d50 (float)                 : the 50th percentile of the cumulative sum of the volume distributon, in microns
    '''

    # calcualte cumulative sum of the volume distribution
    csvd = np.cumsum(vd/np.sum(vd))

    # find the 50th percentile and interpolate if necessary
    d50 = np.interp(0.5,csvd,dias)
    return d50

def get_size_bins():
    '''
    Retrieve size bins for PSD analysis
    
    Returns:
        bin_mids_um (array)     : mid-points of size bins
        bin_limits_um (array)   : limits of size bins
    '''
    # pre-allocate
    bin_limits_um = np.zeros((53),dtype=np.float64)

    # define the upper limit of the smallest bin (same as LISST-100x type-c)
    bin_limits_um[0] = 2.72 * 0.91

    # loop through 53 size classes and calculate the bin limits
    for I in np.arange(1,53,1):
        # each bin is 1.18 * larger than the previous
        bin_limits_um[I] = bin_limits_um[I-1] * 1.180

    # pre-allocate
    bin_mids_um = np.zeros((52),dtype=np.float64)

    # define the middle of the smallest bin (same as LISST-100x type-c)
    bin_mids_um[0] = 2.72

    # loop through 53 size classes and calculate the bin mid-points
    for I in np.arange(1,52,1):
        # each bin is 1.18 * larger than the previous
        bin_mids_um[I]=bin_mids_um[I-1]*1.180

    return bin_mids_um, bin_limits_um

def vd_from_nd(count,psize,sv=1):
    '''
    Calculate volume concentration from particle count

    sv = sample volume size (litres)

    e.g:
    sample_vol_size=25*1e-3*(1200*4.4e-6*1600*4.4e-6); %size of sample volume in m^3
    sv=sample_vol_size*1e3; %size of sample volume in litres
    
    Args:
        count (array) : particle number distribution
        psize (float) : pixel size of the SilCam contained in settings.PostProcess.pix_size from the config ini file
        sv=1 (float)  : the volume of the sample which should be used for scaling concentrations
        
    Returns:
        vd (array)    : the particle volume distribution
    '''

    psize = psize *1e-6  # convert to m

    pvol = 4/3 *np.pi * (psize/2)**3  # volume in m^3

    tpvol = pvol * count * 1e9  # volume in micro-litres

    vd = tpvol / sv  # micro-litres / litre

    return vd


def nc_from_nd(count,sv):
    ''' 
    Calculate the number concentration from the count and sample volume
    
    Args:
        count (array) : particle number distribution
        sv=1 (float)  : the volume of the sample which should be used for scaling concentrations
        
    Returns:
        nc (float)    : the total number concentration in #/L
    '''
    nc = np.sum(count) / sv
    return nc

def nc_vc_from_stats(stats, settings, oilgas=outputPartType.all):
    '''
    Calculates important summary statistics from a stats DataFrame
    
    Args:
        stats (DataFrame)           : particle statistics from silcam process
        settings (PySilcamSettings) : settings associated with the data, loaded with PySilcamSettings
        oilgas=oc_pp.outputPartType.all : the oilgas enum if you want to just make the figure for oil, or just gas (defulats to all particles)
    
    Returns:
        nc (float)            : the total number concentration in #/L
        vc (float)            : the total volume concentration in uL/L
        sample_volume (float) : the total volume of water sampled in L
        junge (float)         : the slope of a fitted juge distribution between 150-300um
    '''
    # get the path length from the config file
    path_length = settings.path_length

    # get pixel_size from config file
    pix_size = settings.pix_size

    # calculate the sample volume per image
    sample_volume = get_sample_volume(pix_size, path_length=path_length, imx=2048, imy=2448)

    # count the number of images analysed
    nims = count_images_in_stats(stats)

    # scale the sample volume by the number of images recorded
    sample_volume *= nims

    # extract only wanted particle stats
    if oilgas==outputPartType.oil:
        from pysilcam.oilgas import extract_oil
        stats = extract_oil(stats)
    elif oilgas==outputPartType.gas:
        from pysilcam.oilgas import extract_gas
        stats = extract_gas(stats)

    # calculate the number distribution
    dias, necd = nd_from_stats(stats, settings)

    # calculate the volume distribution from the number distribution
    vd = vd_from_nd(necd, dias, sample_volume)

    # calculate the volume concentration
    vc = np.sum(vd)

    # calculate the number concentration
    nc = nc_from_nd(necd, sample_volume)

    # convert nd to units of nc per micron per litre
    nd = nd_rescale(dias, necd, sample_volume)

    # remove data from first bin which will be part-full
    ind = np.argwhere(nd>0)
    nd[ind[0]] = np.nan

    # calcualte the junge distirbution slope
    junge = get_j(dias,nd)

    return nc, vc, sample_volume, junge


def nd_from_stats_scaled(stats, settings):
    ''' calcualte a scaled number distribution from stats and settings
    units of nd are in number per micron per litre
    '''
    # calculate the number distirbution (number per bin per sample volume)
    dias, necd = nd_from_stats(stats,settings)

    # calculate the sample volume per image
    sample_volume = get_sample_volume(settings.pix_size,
            path_length=settings.path_length)

    # count the number of images
    nims = count_images_in_stats(stats)

    # scale the sample volume by the number of images
    sample_volume *= nims

    # re-scale the units of the number distirbution into number per micron per
    # litre
    nd = nd_rescale(dias, necd, sample_volume)

    # nan the first bin in measurement because it will always be part full
    ind = np.argwhere(nd>0)
    nd[ind[0]] = np.nan

    return dias, nd


def nd_from_stats(stats, settings):
    ''' calcualte  number distirbution from stats
    units are number per bin per sample volume
    '''

    # convert the equiv diameter from pixels into microns
    ecd = stats['equivalent_diameter'] * settings.pix_size

    # ignore nans
    ecd = ecd[~np.isnan(ecd)]

    # get the size bins into which particles will be counted
    dias, bin_limits_um = get_size_bins()

    # count particles into size bins
    necd, edges = np.histogram(ecd,bin_limits_um)

    # make it float so other operations are easier later
    necd = np.float64(necd)

    return dias, necd


def vd_from_stats(stats, settings):
    ''' calculate volume distribution from stats
    units of miro-litres per sample volume
    '''

    # obtain the number distribution
    dias, necd = nd_from_stats(stats, settings)

    # convert the number distribution to volume in units of micro-litres per
    # sample volume
    vd = vd_from_nd(necd,dias)

    return dias, vd


class TimeIntegratedVolumeDist:
    ''' class used for summarising recent stats in real-time

    @todo - re-implement this later
    '''
    def __init__(self, settings):
        self.settings = settings
        self.window_size = settings.window_size
        self.times = []
        self.vdlist = []

        self.vd_mean = None
        self.dias = None

    def update_from_stats(self, stats, timestamp):
        '''Update size distribution from stats'''
        dias, vd = vd_from_stats(stats, self.settings)
        self.dias = dias

        #Add the new data
        self.times.append(timestamp)
        self.vdlist.append(vd)

        #Remove data until we are within window size
        while (timestamp - self.times[0]).seconds > self.window_size:
            self.times.pop(0)
            self.vdlist.pop(0)

        #Calculate time-integrated volume distribution
        if len(self.vdlist)>1:
            self.vd_mean = np.nanmean(self.vdlist, axis=0)
        else:
            self.vd_mean = self.vdlist[0]


def montage_maker(roifiles, roidir, pixel_size, msize=2048, brightness=255,
        tightpack=False, eyecandy=True):
    '''
    makes nice looking matages from a directory of extracted particle images

    use make_montage to call this function
    '''

    # pre-allocate an empty canvas
    montage = np.zeros((msize,msize,3),dtype=np.uint8())
    # pre-allocate an empty test canvas
    immap_test = np.zeros_like(montage[:,:,0])
    print('making a montage - this might take some time....')

    # loop through each extracted particle and attempt to add it to the canvas
    for files in roifiles:
        # get the particle image from the HDF5 file
        particle_image = export_name2im(files, roidir)

        # measure the size of this image
        [height, width] = np.shape(particle_image[:,:,0])

        # sanity-check on the particle image size
        if height >= msize:
            continue
        if width >= msize:
            continue

        if eyecandy:
            # contrast exploding:
            particle_image = explode_contrast(particle_image)

            # eye-candy normalization:
            peak = np.median(particle_image.flatten())
            bm = brightness - peak
            particle_image = np.float64(particle_image) + bm
        else:
            particle_image = np.float64(particle_image)
        particle_image[particle_image>255] = 255

        # tighpack checks fitting within the canvas based on an approximation
        # of the particle area. If not tightpack, then the fitting will be done
        # based on bounding boxes instead
        if tightpack:
            imbw = scpr.image2blackwhite_accurate(np.uint8(particle_image[:,:,0]), 0.95)
            imbw = ndi.binary_fill_holes(imbw)

            for J in range(5):
                imbw = skimage.morphology.binary_dilation(imbw)

        # initialise a counter
        counter = 0

        # try five times to fit the particle to the canvas by randomly moving
        # it around
        while (counter < 5):
            r = np.random.randint(1,msize-height)
            c = np.random.randint(1,msize-width)

            # tighpack checks fitting within the canvas based on an approximation
            # of the particle area. If not tightpack, then the fitting will be done
            # based on bounding boxes instead
            if tightpack:
                test = np.max(immap_test[r:r+height,c:c+width]+imbw)
            else:
                test = np.max(immap_test[r:r+height,c:c+width,None]+1)


            # if the new particle is overlapping an existing object in the
            # canvas, then try again and increment the counter
            if (test>1):
                counter += 1
            else:
                break

        # if we reach this point and there is still an overlap, then forget
        # this particle, and move on
        if (test>1):
            continue

        # if we reach here, then the particle has found a position in the
        # canvas with no overlap, and can then be inserted into the canvas
        montage[r:r+height,c:c+width,:] = np.uint8(particle_image)

        # update the testing canvas so it is ready for the next particle
        if tightpack:
            immap_test[r:r+height,c:c+width] = imbw
        else:
            immap_test[r:r+height,c:c+width,None] = immap_test[r:r+height,c:c+width,None]+1

    # now the montage is finished
    # here are some small eye-candy scaling things to tidy up
    montageplot = np.copy(montage)
    montageplot[montage>255] = 255
    montageplot[montage==0] = 255
    print('montage complete')

    return montageplot


def make_montage(stats_csv_file, pixel_size, roidir,
        auto_scaler=500, msize=1024, maxlength=100000,
        oilgas=outputPartType.all):
    ''' wrapper function for montage_maker
    '''

    # obtain particle statistics from the csv file
    stats = pd.read_csv(stats_csv_file)

    # remove nans because concentrations are not important here
    stats = stats[~np.isnan(stats['major_axis_length'])]
    stats = stats[(stats['major_axis_length'] *
            pixel_size) < maxlength]

    # extract only wanted particle stats
    if oilgas==outputPartType.oil:
        from pysilcam.oilgas import extract_oil
        stats = extract_oil(stats)
    elif oilgas==outputPartType.gas:
        from pysilcam.oilgas import extract_gas
        stats = extract_gas(stats)

    # sort the particles based on their length
    stats.sort_values(by=['major_axis_length'], ascending=False, inplace=True)

    roifiles = gen_roifiles(stats, auto_scaler=auto_scaler)

    eyecandy = True
    if not (oilgas==outputPartType.all):
        eyecandy = False

    montage = montage_maker(roifiles, roidir, pixel_size, msize, eyecandy=eyecandy)

    return montage


def gen_roifiles(stats, auto_scaler=500):

    roifiles = stats['export name'][stats['export name'] !=
            'not_exported'].values

    # subsample the particles if necessary
    print('rofiles:',len(roifiles))
    IMSTEP = np.max([np.int(np.round(len(roifiles)/auto_scaler)),1])
    print('reducing particles by factor of {0}'.format(IMSTEP))
    roifiles = roifiles[np.arange(0,len(roifiles),IMSTEP)]
    print('rofiles:',len(roifiles))

    return roifiles


def get_sample_volume(pix_size, path_length=10, imx=2048, imy=2448):
    ''' calculate the sample volume of one image
    '''
    sample_volume_litres = imx*pix_size/1000 * imy*pix_size/1000 * path_length*1e-6

    return sample_volume_litres


def get_j(dias, nd):
    ''' calculates the junge slope from a correctly-scale number distribution
    (number per micron per litre must be the units of nd)
    '''
    # conduct this calculation only on the part of the size distribution where
    # LISST-100 and SilCam data overlap
    ind = np.isfinite(dias) & np.isfinite(nd) & (dias<300) & (dias>150)

    # use polyfit to obtain the slope of the ditriubtion in log-space (which is
    # assumed near-linear in most parts of the ocean)
    p = np.polyfit(np.log(dias[ind]),np.log(nd[ind]),1)
    j = p[0]
    return j


def count_images_in_stats(stats):
    ''' count the number of raw images used to generate stats
    '''
    u = pd.to_datetime(stats['timestamp']).unique()
    n_images = len(u)

    return n_images


def extract_nth_largest(stats,settings,n=0):
    ''' return statistics of the nth largest particle
    '''
    stats.sort_values(by=['equivalent_diameter'], ascending=False, inplace=True)
    stats = stats.iloc[n]
    return stats


def extract_nth_longest(stats,settings,n=0):
    ''' return statistics of the nth longest particle
    '''
    stats.sort_values(by=['major_axis_length'], ascending=False, inplace=True)
    stats = stats.iloc[n]
    return stats


def d50_timeseries(stats, settings):
    ''' Calculates time series of d50 from stats
    '''

    from tqdm import tqdm

    stats = stats.sort_values(by='timestamp')

    td = pd.to_timedelta('00:00:' + str(settings.window_size/2.))
    d50 = []
    time = []

    u = pd.to_datetime(stats['timestamp']).unique()

    for t in tqdm(u):
        dt = pd.to_datetime(t)
        stats_ = stats[(pd.to_datetime(stats['timestamp'])<(dt+td)) & (pd.to_datetime(stats['timestamp'])>(dt-td))]
        d50.append(d50_from_stats(stats_, settings))
        time.append(t)

    if len(time) == 0:
        d50 = np.nan
        time = np.nan

    return d50, time



def explode_contrast(im):
    ''' eye-candy function for exploding the contrast of a particle iamge (roi)
    '''
    # make sure iamge is float
    im = np.float64(im)

    # re-scale the instensities in the image to chop off some ends
    p1, p2 = np.percentile(im, (0, 80))
    im_mod = rescale_intensity(im, in_range=(p1, p2))

    # set minimum value to zero
    im_mod -= np.min(im_mod)

    # set maximum value to one
    im_mod /= np.max(im_mod)

    # re-scale to match uint8 max
    im_mod *= 255

    # convert to unit8
    im_mod = np.uint8(im_mod)
    return im_mod


def bright_norm(im,brightness=255):
    ''' eye-candy function for normalising the image brightness
    '''
    peak = np.median(im.flatten())
    bm = brightness - peak

    im = np.float64(im) + bm
    im[im>255] = 255

    im =np.uint8(im)
    return im


def nd_rescale(dias, nd, sample_volume):
    ''' rescale a number distribution from
            number per bin per sample volume
        to
            number per micron per litre
    '''
    nd = np.float64(nd) / sample_volume # nc per size bin per litre

    # convert nd to units of nc per micron per litre
    dd = np.gradient(dias)
    nd /= dd
    nd[nd<0] = np.nan # and nan impossible values!

    return nd

def add_depth_to_stats(stats, time, depth):
    ''' if you have a depth time-series, use this function to find the depth of
    each line in stats
    '''
    # get times
    sctime = pd.to_datetime(stats['timestamp'])
    # interpolate depths into the SilCam times
    stats['Depth'] = np.interp(np.float64(sctime), np.float64(time), depth)
    return stats


def export_name2im(exportname, path):
    ''' returns an image from the export name string in the -STATS.csv file

    get the exportname like this: exportname = stats['export name'].values[0]
    '''

    # the particle number is defined after the time info
    pn = exportname.split('-')[1]
    # the name is the first bit
    name = exportname.split('-')[0] + '.h5'

    # combine the name with the location of the exported HDF5 files
    fullname = os.path.join(path, name)

    # open the H5 file
    fh = h5py.File(fullname ,'r')

    # extract the particle image of interest
    im = fh[pn]

    return im


def extract_latest_stats(stats, window_size):
    ''' extracts the stats data from within the last number of seconds specified
    by window_size.

    returns stats dataframe (from the last window_size seconds)
    '''
    end = np.max(pd.to_datetime(stats['timestamp']))
    start = end - pd.to_timedelta('00:00:' + str(window_size))
    stats = stats[pd.to_datetime(stats['timestamp'])>start]
    return stats


def silc_to_bmp(directory):
    files = [s for s in os.listdir(directory) if s.endswith('.silc')]
    
    for f in files:
        try:
            with open(os.path.join(directory, f), 'rb') as fh:
                im = np.load(fh, allow_pickle=False)
                fout = os.path.splitext(f)[0] + '.bmp'
            outname = os.path.join(directory, fout)
            imo.imwrite(outname, im)
        except:
            print(f, ' failed!')
            continue

    print('Done.')


def make_timeseries_vd(stats, settings):
    '''makes a dataframe of time-series volume distribution and d50

    Args:
        stats (silcam stats dataframe): loaded from a *-STATS.csv file
        settings (silcam settings): loaded from PySilCamSettings

    Returns:
        dataframe: of time series
    '''

    from tqdm import tqdm

    u = pd.to_datetime(stats['timestamp']).unique()
    
    sample_volume = get_sample_volume(settings.PostProcess.pix_size, path_length=settings.PostProcess.path_length)

    vdts = []
    d50 = []
    timestamp = []
    dias = []
    for s in tqdm(u):
        dias, vd = vd_from_stats(stats[pd.to_datetime(stats['timestamp'])==s],
                settings.PostProcess)
        nims = count_images_in_stats(stats[pd.to_datetime(stats['timestamp'])==s])
        sv = sample_volume * nims
        vd /= sv
        d50_ = d50_from_vd(vd, dias)
        d50.append(d50_)
        timestamp.append(pd.to_datetime(s))
        vdts.append(vd)

    if len(vdts) == 0:
        dias, limits = get_size_bins()
        vdts = np.zeros_like(dias) * np.nan

        time_series = pd.DataFrame(data=[np.squeeze(vdts)], columns=dias)

        time_series['D50'] = np.nan
        time_series['Time'] = np.nan

        return time_series

    time_series = pd.DataFrame(data=np.squeeze(vdts), columns=dias)

    time_series['D50'] = d50
    time_series['Time'] = timestamp

    return time_series


def stats_to_xls_png(config_file, stats_filename, oilgas=outputPartType.all):
    '''summarises stats in two excel sheets of time-series PSD and averaged
    PSD.

    Args:
        config_file (string)            : Path of the config file for this data
        stats_filename (string)         : Path of the stats csv file
        oilgas=oc_pp.outputPartType.all : the oilgas enum if you want to just make the figure for oil, or just gas (defulats to all particles)

    Returns:
        dataframe: of time series
        files: in the proc folder)
    '''
    settings = PySilcamSettings(config_file)
    
    stats = pd.read_csv(stats_filename)
    oilgasTxt = ''

    if oilgas==outputPartType.oil:
        from pysilcam.oilgas import extract_oil
        stats = extract_oil(stats)
        oilgasTxt = 'oil'
    elif oilgas==outputPartType.gas:
        from pysilcam.oilgas import extract_gas
        stats = extract_gas(stats)
        oilgasTxt = 'gas'

    df = make_timeseries_vd(stats, settings)

    df.to_excel(stats_filename.strip('-STATS.csv') +
            '-TIMESERIES' + oilgasTxt + '.xlsx')
    
    sample_volume = get_sample_volume(settings.PostProcess.pix_size, path_length=settings.PostProcess.path_length)
   
    dias, vd = vd_from_stats(stats,
                settings.PostProcess)
    nims = count_images_in_stats(stats)
    sv = sample_volume * nims
    vd /= sv
    
    d50 = d50_from_vd(vd, dias)
    
    dfa = pd.DataFrame(data=[vd], columns=dias)
    dfa['d50'] = d50
    
    timestamp = np.min(pd.to_datetime(df['Time']))
    dfa['Time'] = timestamp
    
    dfa.to_excel(stats_filename.strip('-STATS.csv') +
            '-AVERAGE' + oilgasTxt + '.xlsx')

    return df


def trim_stats(stats_csv_file, start_time, end_time, write_new=False, stats=[]):
    '''Chops a STATS.csv file given a start and end time'''
    if len(stats)==0:
        stats = pd.read_csv(stats_csv_file)

    start_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time)

    trimmed_stats = stats[
        (pd.to_datetime(stats['timestamp']) > start_time) & (pd.to_datetime(stats['timestamp']) < end_time)]

    if np.isnan(trimmed_stats.equivalent_diameter.max()) or len(trimmed_stats) == 0:
        print('No data in specified time range!')
        outname = ''
        return trimmed_stats, outname

    actual_start = pd.to_datetime(trimmed_stats['timestamp'].min()).strftime('D%Y%m%dT%H%M%S.%f')
    actual_end = pd.to_datetime(trimmed_stats['timestamp'].max()).strftime('D%Y%m%dT%H%M%S.%f')

    path, name = os.path.split(stats_csv_file)

    outname = os.path.join(path, name.strip('-STATS.csv')) + '-Start' + str(actual_start) + '-End' + str(
        actual_end) + '-STATS.csv'

    if write_new:
        trimmed_stats.to_csv(outname)

    return trimmed_stats, outname
