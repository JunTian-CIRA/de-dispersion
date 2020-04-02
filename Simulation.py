#!/usr/bin/env python

import numpy as np
import math
import glob
from astropy.io import fits


DM = np.random.rand(100)*(3000-100)+100 # create randome DMs
Fluence = np.random.rand(100)*(1000-100)+100 # create random fluences in unit of Jy.ms, assuming duration of 1ms
t0 = 1254248036 # assuming the burst happens at some time


chan_24c = np.arange(24) # 24 coarse channels
chan_width = 1.28 # 1.28MHz channel width
chan_24c = (133+chan_24c)*chan_width # channel frequency = channel no. * channel width
Flux = np.zeros((len(DM)))
Flux = Fluence/24 # assuming the spectrum is flat within the 30MHz bandwidth
t = np.zeros((len(DM),len(chan_24c)))
for i in range(len(DM)):
    t[i,:] = t0+4.15/1000.*DM[i]/(chan_24c/1000.)**2 # channels are delayed differently


# read original image files and inject simulated pulses
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
    image_pixel[i] = image[pixel_test[0],pixel_test[1]] # read the flux at the specified pixel
    hdu.close()


time = np.zeros((len(paths)))
channel = np.zeros((len(paths)))
for i in range(len(paths)):
    time[i] = float(paths[i].split('-')[3][1:])*0.5+float(paths[i].split('-')[0].split('/')[-1]) # get the time stamp of each image
    channel[i] = float(paths[i].split('/')[-1].split('-')[5]) # get the channel stamp for each image


# make a 2D dynamic spectrum for the pixel
dyn_spec = np.zeros((len(paths)//len(chan_24c),len(chan_24c))) # reshape the image_pixel as time no. * chan no.
T = np.zeros((len(paths)//len(chan_24c))) # reduced time stamp
for i in range(len(paths)):
    dyn_spec[i//len(chan_24c),i%len(chan_24c)] = image_pixel[i]
    T[i//len(chan_24c)] = time[i]
dyn_spec = dyn_spec[np.argsort(T),:] # sorted according to time
T = np.sort(T)


simu_spec = np.zeros((len(DM),len(paths)//len(chan_24c),len(chan_24c)))
for i in range(len(DM)):
    inds = np.digitize(t[i,:],T) # return the interval No. each dispersed pulse time belongs to
    simu_spec[i,:,:] = dyn_spec
    for j in range(len(chan_24c)):
        if T[inds[j]]-T[inds[j]-1]==0.5: # exclude the gaps longer than 0.5s
            simu_spec[i,inds[j]-1,j] = simu_spec[i,inds[j]-1,j]+Flux[i] # add the simulated signal to pre-existing flux per channel


for i in range(len(DM)):
    header = fits.Header()
    header['DM'] = DM[i]
    header.comments['DM'] = 'DM value of the simulated signal (pc cm^{-3})'
    header['Fluence'] = Fluence[i]
    header.comments['Fluence'] = 'fluence of the simulated signal (Jy.ms)'
    header['comment'] = 'the signal is assumed to last 1ms'
    fits.writeto('DM_{0}_F_{1}_sim.fits'.format(int(DM[i]),int(Fluence[i])), simu_spec[i,:,:].transpose(), header, overwrite=True)
