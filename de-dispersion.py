#!/usr/bin/env python


from astropy.io import fits
import glob
import numpy as np
import math
import matplotlib.pyplot as plt
#from scipy import stats
from astropy.stats import median_absolute_deviation # the scipy version on Pawsey don't have this function, so use astropy instead


paths = sorted(glob.glob('/astro/mwasci/phancock/D0009/Jun_test/data_1st_paper/GRB191004A/05s_24c/*-pbcorr-I.fits')) # path to the 0.5s coarse-channel images


image_size = 256 # image size in pixels
pixel_size = 1.6*60 # pixel size in unit of asec
GRB_radius = 1.1*60 # GRB position error in unit of asec
GRB_radius = math.ceil(GRB_radius/pixel_size) # GRB position error in pixels
GRB_center = np.array([image_size/2,image_size/2]).astype(int) # pixel taken by GRB at the image center
pixel_test = GRB_center # each time read one pixel because all images take too much memory


image = np.zeros((image_size, image_size))
image_pixel = np.zeros((len(paths)))
for i, path in enumerate(paths):
    hdu = fits.open(path)
    image[:,:] = hdu[0].data[:,:]
    image_pixel[i] = image[pixel_test[0],pixel_test[1]] # get the flux at the chosen pixel of every coarse-channel images
    hdu.close()


time = np.zeros((len(paths)))
channel = np.zeros((len(paths)))
for i in range(len(paths)):
    time[i] = float(paths[i].split('-')[3][1:])*0.5+float(paths[i].split('-')[0].split('/')[-1]) # get the time stamp of each image
    channel[i] = float(paths[i].split('/')[-1].split('-')[5]) # get the channel stamp for each image


# subtract the background from each channel
chan_num = 24 # the number of coarse channels
for i in range(chan_num):
    image_pixel[np.array(channel==i)] = image_pixel[np.array(channel==i)] - np.nanmean(image_pixel[np.array(channel==i)])


# do de-dispersion
chan_width = 1.28 # 1.28MHz channel width
channel = (133+channel)*chan_width # channel frequency = channel no. * channel width
DM_min = 30
DM_max = 3000
DM_step = 12 # set minimum and maximum DM and DM step
DM = np.zeros(((DM_max-DM_min)//DM_step+1))
de_time = np.zeros(((DM_max-DM_min)//DM_step+1, len(paths))) # de-dispersed time stamp of each image corresponding to different DMs
for i in range((DM_max-DM_min)//DM_step+1):
    DM[i] = DM_min + i*DM_step
    de_time[i,:] = time - 4.15/1000.*DM[i]/(channel/1000.)**2 # time delay_s = 4.15_ms * DM * (1/v_GHz)^2 with infinite reference energy


flux_bin = np.full(((DM_max-DM_min)//DM_step+1,100000),np.nan) # use nan to fill in the array because of uncertainty of bin number
time_bin = np.full(((DM_max-DM_min)//DM_step+1,100000),np.nan) # and it's easy to exclude nan values in the following calculation
norm = np.full(((DM_max-DM_min)//DM_step+1,100000),np.nan)
bin = 5 # bin size in sec
for i in range((DM_max-DM_min)//DM_step+1):
    t_min = de_time[i,:].min()
    t_max = de_time[i,:].max()
    bins = np.arange(t_min,t_max,bin)
    inds = np.digitize(de_time[i,:],bins) # return the bin No. each time stamp belongs to
    for j in range(len(bins)):
        time_bin[i,j] = t_min+j*bin # digitized time stamp
        flux_bin[i,j] = np.nansum(image_pixel[np.array(inds==j+1)]) # add flux in the same bin, be careful the inds starts from 1
        norm[i,j] = len(image_pixel[np.array(inds==j+1)]) # number of images falling in a bin, used later for normalization


SNR_max = np.zeros((DM_max-DM_min)//DM_step+1) # maximum SNR for each DM trial
for i in range((DM_max-DM_min)//DM_step+1):
    norm[i,:] = norm[i,:]/np.nansum(norm[i,:])
    flux_max = np.nanmax(flux_bin[i,:])
    rms = np.nansum(norm[i,:]*flux_bin[i,:]**2)
    rms = np.sqrt(rms) # weighted stdev
    SNR_max[i] = flux_max/rms


plt.plot(DM,SNR_max)
plt.savefig('SNR_vs_DM_'+str(pixel_test[0])+'_'+str(pixel_test[1])+'.png') # save the plot named by the pixel position
plt.close()
