#!/usr/bin/env python

from astropy.io import fits
import glob
import numpy as np
import math
import matplotlib.pyplot as plt


# retrive time steps
paths = sorted(glob.glob('/astro/mwasci/phancock/D0009/Jun_test/data_1st_paper/GRB191004A/05s_24c/*-pbcorr-I.fits'))
time = np.zeros((len(paths)))
channel = np.zeros((len(paths)))
for i in range(len(paths)):
    time[i] = float(paths[i].split('-')[3][1:])*0.5+float(paths[i].split('-')[0].split('/')[-1])
    channel[i] = float(paths[i].split('/')[-1].split('-')[5])
T = np.zeros((len(paths)//24))
for i in range(len(paths)):
    T[i//24] = time[i]
T = np.sort(T)


# do de-dispersion
chan_num = 24
chan_width = 1.28
chan = np.arange(133,156+1)*chan_width
DM_min = 100
DM_max = 3000
DM_step = 10
Delta_t = np.zeros(((DM_max-DM_min)//DM_step+1,chan_num))
DM = np.zeros(((DM_max-DM_min)//DM_step+1))
for i in range((DM_max-DM_min)//DM_step+1):
    DM[i] = DM_min+i*DM_step
    Delta_t[i,:] = 4.15/1000.*DM[i]/(chan/1000.)**2
t_de_dis = np.zeros(((DM_max-DM_min)//DM_step+1,len(T),chan_num))
for i in range((DM_max-DM_min)//DM_step+1):
    for j in range(chan_num):
        t_de_dis[i,:,j] = T-Delta_t[i,j]


# read simulated dynamic spectrum, and process one by one
paths = sorted(glob.glob('/astro/mwasci/phancock/D0009/Jun_test/data_1st_paper/GRB191004A/simulation/DM_1014_F_508_sim.fits'))
for k, path in enumerate(paths):
    hdu = fits.open(path)
    image = hdu[0].data[:,:]
    image = np.transpose(image)
    hdu.close
    # calculate max SNR
    flux_bin = np.full(((DM_max-DM_min)//DM_step+1,10000),np.nan)
    time_bin = np.full(((DM_max-DM_min)//DM_step+1,10000),np.nan)
    norm = np.full(((DM_max-DM_min)//DM_step+1,10000),np.nan)
    bin = 5
    for i in range((DM_max-DM_min)//DM_step+1):
        t_min = t_de_dis[i,:,:].min()
        t_max = t_de_dis[i,:,:].max()
        bins = np.arange(t_min,t_max,bin)
        inds = np.digitize(t_de_dis[i,:,:],bins)
        for j in range(len(bins)):
            flux_bin[i,j] = np.nansum(image[np.array(inds==j+1)])
            time_bin[i,j] = t_min+j*bin
            norm[i,j] = len(image[np.array(inds==j+1)])
    SNR_max = np.zeros((DM_max-DM_min)//DM_step+1)
    flux_max = np.zeros((DM_max-DM_min)//DM_step+1)
    rms = np.zeros((DM_max-DM_min)//DM_step+1)
    for i in range((DM_max-DM_min)//DM_step+1):
        norm[i,:] = norm[i,:]/np.nansum(norm[i,:])
        flux_max[i] = np.nanmax(flux_bin[i,:])
        rms[i] = np.nansum(norm[i,:]*flux_bin[i,:]**2)
        rms[i] = np.sqrt(rms[i])
        SNR_max[i] = flux_max[i]/rms[i]
    # plot SNR_max vs DM
    tmp = path.split('/')[-1].split('_')[1]
    tmpp = path.split('/')[-1].split('_')[3]
    tmppp = int((float(tmp)-DM_min)/DM_step) + 2
    plt.plot(time_bin[tmppp,:], flux_bin[tmppp,:])
#    plt.axvline(int(tmp), color='red', linestyle='--')
    plt.axhline(rms[tmppp], color='red', linestyle='--')
    plt.axhline(3*rms[tmppp], color='red', linestyle='--')
    plt.axhline(6*rms[tmppp], color='red', linestyle='--')
    plt.text(time_bin[tmppp,0], rms[tmppp], '$1\sigma$', fontsize=12)
    plt.text(time_bin[tmppp,0], 3*rms[tmppp], '$3\sigma$', fontsize=12)
    plt.text(time_bin[tmppp,0], 6*rms[tmppp], '$6\sigma$', fontsize=12)
    plt.xlabel("Time (sec)")
    plt.ylabel("Flux (Jy)")
    plt.title('fluence =' + str(tmpp) + 'Jy.ms' + ', DM =' + str(int(DM[tmppp])) + '$pc\,cm^{-3}$')
    plt.savefig('DM_' + str(int(DM[tmppp])) + '_F_' + str(tmpp) + '_lc.png')
    plt.close()
