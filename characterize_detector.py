#!/usr/env/python

## Import General Tools
import sys
import argparse
import logging
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy import stats
from astropy.table import Table, Column
from astropy.modeling import models, fitting
from astropy.visualization import MinMaxInterval, PercentileInterval, ImageNormalize
import ccdproc
from ccdproc import ImageFileCollection as IFC

from matplotlib import pyplot as plt

import warnings
from astropy.utils.exceptions import AstropyDeprecationWarning as ADW
warnings.filterwarnings('ignore', category=ADW, append=True)


##-------------------------------------------------------------------------
## Parse Command Line Arguments
##-------------------------------------------------------------------------
## create a parser object for understanding command-line arguments
p = argparse.ArgumentParser(description='''
''')
## add flags
p.add_argument("-v", "--verbose", dest="verbose",
    default=False, action="store_true",
    help="Be verbose! (default = False)")
p.add_argument("-p", "--plot", "--plots", dest="plot",
    default=False, action="store_true",
    help="Generate plots")
## add options
p.add_argument("--aduthreshold", dest="aduthreshold", type=float,
    default=65000,
    help="ADU threshold above which files are ignored.")
p.add_argument("--exptime", dest="exptime", type=str,
    default="EXPTIME",
    help="Header keyword for exposure time in seconds.")
p.add_argument("--imtype", dest="imtype", type=str,
    default="IMAGETYP",
    help="Header keyword for image type.")
p.add_argument("--ccdtemp", dest="ccdtemp", type=str,
    default="CCD-TEMP",
    help="Header keyword for CCD temperature.  None to ignore.")
p.add_argument("--gain", dest="gain", type=str,
    default="GAIN",
    help="Header keyword for gain in e/ADU.")
p.add_argument("--trimpix", dest="trimpix", type=int,
    default=0,
    help="Number of pixels to trim from edges before analysis.")
p.add_argument("--clippingsigma", dest="clippingsigma", type=float,
    default=5,
    help="Clipping sigma.")
p.add_argument("--clippingiters", dest="clippingiters", type=int,
    default=3,
    help="Number of sigma clipping iterations.")
p.add_argument("--hpthresh", dest="hpthresh", type=int,
    default=10,
    help="Threshold for tagging as a hot pixel (in ADU/s).")
p.add_argument("--darkfilter", dest="darkfilter", type=str,
    default="None",
    help="""If CCD uses a dark filter rather than an IMTYPE to distingush bias
    and dark files, set the filter keyword here which will be searched for 
    "dark".  Set to None to ignore.""")

## add arguments
p.add_argument('files', nargs='*',
               help="Input files")
args = p.parse_args()


##-------------------------------------------------------------------------
## Create logger object
##-------------------------------------------------------------------------
log = logging.getLogger('CharacterizeDetector')
log.setLevel(logging.DEBUG)
## Set up console output
LogConsoleHandler = logging.StreamHandler()
LogConsoleHandler.setLevel(logging.DEBUG)
LogFormat = logging.Formatter('%(asctime)s %(levelname)8s: %(message)s',
                              datefmt='%Y-%m-%d %H:%M:%S')
LogConsoleHandler.setFormatter(LogFormat)
log.addHandler(LogConsoleHandler)
## Set up file output
# LogFileName = None
# LogFileHandler = logging.FileHandler(LogFileName)
# LogFileHandler.setLevel(logging.DEBUG)
# LogFileHandler.setFormatter(LogFormat)
# log.addHandler(LogFileHandler)

##-------------------------------------------------------------------------
## get_mode
##-------------------------------------------------------------------------
def get_mode(im):
    '''
    Return mode of image.  Assumes int values (ADU), so uses binsize of one.
    '''
    if type(im) == ccdproc.CCDData:
        data = im.data.ravel()
    elif type(im) == fits.HDUList:
        data = im[0].data.ravel()
    else:
        data = im
    
    bmin = np.floor(min(data)) - 1./2.
    bmax = np.ceil(max(data)) + 1./2.
    bins = np.arange(bmin,bmax,1)
    hist, bins = np.histogram(data, bins=bins)
    centers = (bins[:-1] + bins[1:]) / 2
    w = np.argmax(hist)
    mode = int(centers[w])
    return mode


##-------------------------------------------------------------------------
## Determine Read Noise
##-------------------------------------------------------------------------
def determine_read_noise(ifc):
    log.info('Determining read noise')
    buf = args.trimpix

    bias_type_names = ['bias', 'dark', 'zero']
    if args.darkfilter is not "None":
        bias_type_names.append('light frame')
    bias_match = (ifc.summary[args.exptime] < 0.001)
    bias_match &= np.array([t.lower() in bias_type_names for t in ifc.summary['IMAGETYP']])
    if args.darkfilter is not "None":
        bias_match &= np.array([f.lower() == 'dark' for f in ifc.summary[args.darkfilter]])

    bias_files = ifc.summary[bias_match]
    print(bias_files)
    biases = []
    for i,bias_file_name in enumerate(bias_files['file']):
        bias_file = Path(ifc.location).joinpath(bias_file_name)
        if i == 0:
            bias0 = ccdproc.fits_ccddata_reader(bias_file, unit='adu')
            ny, nx = bias0.data.shape
            mean, median, stddev = stats.sigma_clipped_stats(
                                         bias0.data[buf:ny-buf,buf:nx-buf],
                                         sigma=args.clippingsigma,
                                         iters=args.clippingiters) * u.adu
            mode = get_mode(bias0)
            print(f'  Bias (mean, med, mode, std) = {mean.value:.1f}, {median.value:.1f}, {mode:d}, {stddev.value:.2f}')
        else:
            biases.append(ccdproc.fits_ccddata_reader(bias_file, unit='adu'))

    log.info('  Making master bias')
    master_bias = ccdproc.combine(biases, combine='average',
                                  sigma_clip=True,
                                  sigma_clip_low_thresh=args.clippingsigma,
                                  sigma_clip_high_thresh=args.clippingsigma)
    ny, nx = master_bias.data.shape
    mean, median, stddev = stats.sigma_clipped_stats(
                                 master_bias.data[buf:ny-buf,buf:nx-buf],
                                 sigma=args.clippingsigma,
                                 iters=args.clippingiters) * u.adu
    mode = get_mode(master_bias)
    log.info(f'  Master Bias (mean, med, mode, std) = {mean.value:.1f}, {median.value:.1f}, {mode:d}, {stddev.value:.2f}')

    diff = bias0.subtract(master_bias)
    ny, nx = diff.data.shape
    mean, median, stddev = stats.sigma_clipped_stats(
                                 diff.data[buf:ny-buf,buf:nx-buf],
                                 sigma=args.clippingsigma,
                                 iters=args.clippingiters) * u.adu
    mode = get_mode(diff)
    log.info(f'  Bias Difference (mean, med, mode, std) = {mean.value:.1f}, {median.value:.1f}, {mode:d}, {stddev.value:.2f}')

    RN = stddev / np.sqrt(1.+1./(len(biases)))
    log.info(f'  Read Noise is {RN:.2f}')

    # Generate Bias Plots
    if args.plot is True:
        log.info(f'Generating plot for example bias file: {bias_files[0]["file"]}')
        data = biases[0].data[buf:ny-buf,buf:nx-buf]
        std = np.std(data)
        binwidth = 10*int(std)
        binsize = 1
        med = np.median(data)
        bins = [x+med for x in range(-binwidth,binwidth,binsize)]
        norm = ImageNormalize(data, interval=PercentileInterval(98))

        plt.figure(figsize=(18,18))
        plt.subplot(2,1,1)
        plt.title(bias_files[0]['file'])
        plt.imshow(data, origin='lower', norm=norm)
        plt.subplot(2,1,2)
        plt.hist(data.ravel(), log=True, bins=bins, color='g', alpha=0.5)
        plt.xlabel('Value (ADU)')
        plt.ylabel('N Pix')
        plt.grid()
        plot_file = Path(ifc.location).joinpath(bias_files[0]["file"].replace('.fits', '.png').replace('.fit', '.png'))
        plt.savefig(plot_file, bbox_inches='tight', pad_inches=0.10)

        # Master Bias
        log.info(f'Generating plot for master bias')
        data = master_bias.data[buf:ny-buf,buf:nx-buf]
        std = np.std(data)
        binwidth = 10*int(std)
        binsize = 1
        med = np.median(data)
        bins = [x+med for x in range(-binwidth,binwidth,binsize)]
        norm = ImageNormalize(data, interval=PercentileInterval(98))

        plt.figure(figsize=(18,18))
        plt.subplot(2,1,1)
        plt.title('Master Bias')
        plt.imshow(data, origin='lower', norm=norm)
        plt.subplot(2,1,2)
        plt.hist(data.ravel(), log=True, bins=bins, color='g', alpha=0.5)
        plt.xlabel('Value (ADU)')
        plt.ylabel('N Pix')
        plt.grid()
        plot_file = Path(ifc.location).joinpath('MasterBias.png')
        plt.savefig(plot_file, bbox_inches='tight', pad_inches=0.10)

    return RN, master_bias


##-------------------------------------------------------------------------
## Determine Dark Current
##-------------------------------------------------------------------------
def determine_dark_current(ifc, master_bias):

    log.info('Determining dark current')
    buf = args.trimpix

    dark_type_names = ['dark']
    if args.darkfilter is not "None":
        dark_type_names.append('light frame')
    dark_match = (ifc.summary[args.exptime] > 0)
    dark_match &= np.array([t.lower() in dark_type_names for t in ifc.summary['IMAGETYP']])
    if args.darkfilter is not "None":
        dark_match &= np.array([f.lower() == 'dark' for f in ifc.summary[args.darkfilter]])

    dark_files = ifc.summary[dark_match]
    dark_table = Table(names=('filename', 'exptime', 'mean', 'median', 'stddev', 'nhotpix'),\
                       dtype=('a100', 'f4', 'f4', 'f4', 'f4', 'i4'))
    print(dark_files)

    for i,entry in enumerate(dark_files):
        dark_file = Path(ifc.location).joinpath(entry['file'])
        exptime = entry[args.exptime]

        dark = ccdproc.fits_ccddata_reader(dark_file, unit='adu')
        dark_diff = ccdproc.subtract_bias(dark, master_bias)
        ny, nx = dark_diff.data.shape
        mean, median, stddev = stats.sigma_clipped_stats(
                                     dark_diff.data[buf:ny-buf,buf:nx-buf],
                                     sigma=args.clippingsigma,
                                     iters=args.clippingiters) * u.adu
        thresh = args.hpthresh*exptime
        nhotpix = len(dark_diff.data.ravel()[dark_diff.data.ravel() > thresh])
        dark_table.add_row([dark_file.name, exptime, mean, median, stddev, nhotpix])

    # Fit Line to Dark Level to Determine Dark Current
    line = models.Linear1D(intercept=0, slope=0)
    line.intercept.fixed = True
    fitter = fitting.LinearLSQFitter()

    longest_exptime = int(max(dark_table['exptime']))
    long_dark_table = dark_table[np.array(dark_table['exptime'], dtype=int) == longest_exptime]

    dc_fit = fitter(line, dark_table['exptime'], dark_table['mean'])
    dark_current = dc_fit.slope.value * u.adu/u.second

    nhotpix = int(np.mean(long_dark_table['nhotpix'])) * u.pix
    nhotpixstd = int(np.std(long_dark_table['nhotpix'])) / np.sqrt(len(long_dark_table['nhotpix'])) * u.pix
    dark_stats = [dark_current, nhotpix, nhotpixstd]
    log.info(f'  Dark Current = {dark_current.value:.3f} ADU/s')
    log.info(f'  N Hot Pixels = {nhotpix:.0f} +/- {nhotpixstd:.0f}')

    # Plot Dark Current Fit
    if args.plot is True:
        plt.figure(figsize=(11,5))
        ax = plt.gca()
        ax.plot(dark_table['exptime'], dark_table['mean'], 'ko', alpha=1.0,
                label='mean count level in ADU')
        ax.plot([0, longest_exptime], [dc_fit(0), dc_fit(longest_exptime)], 'k-', alpha=0.3,
                label=f'dark current = {dark_stats[0].value:.2f} ADU/s')
        plt.xlim(-0.02*max(dark_table['exptime']), 1.10*max(dark_table['exptime']))
        min_level = np.floor(min(dark_table['mean']))
        max_level = np.ceil(max(dark_table['mean']))
        plt.ylim(min([0,min_level]), max_level)
        ax.set_xlabel('Exposure Time (s)')
        ax.set_ylabel('Dark Level (ADU)')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid()
        plot_file = Path(ifc.location).joinpath('DarkCurrent.png')
        plt.savefig(plot_file, bbox_inches='tight', pad_inches=0.10)


##-------------------------------------------------------------------------
## Main Program
##-------------------------------------------------------------------------
def main():
    files = [Path(f) for f in args.files]
    keywords = [args.exptime, args.imtype, args.ccdtemp]
    if args.darkfilter is not "None":
        keywords.append(args.darkfilter)
    ifc = IFC(location=files[0].parent, filenames=files, keywords=keywords)
    print(ifc.summary)

    RN, master_bias = determine_read_noise(ifc)
    DC = determine_dark_current(ifc, master_bias)



if __name__ == '__main__':
    main()
