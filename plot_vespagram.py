#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 10:44:38 2023

@author: matteo
"""
import time as time_module
from obspy.core.util import AttribDict
import obspy
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from obspy.signal import array_analysis as aran
from obspy import taup
from matplotlib.animation import FuncAnimation
from vespa import Vespagram
import pickle

plt.style.use("ggplot")
plt.ion()

# %% parameters
# data collection
pre_cut_t = 440
post_cut_t = 40
dmin, dmax = (100, 178)
# vespagram
channel = "T"
case = "PP"
freqmin, freqmax= 1/50, 1/10
do_filt = True
smin, smax, ds = 0, 1.4, .01
dwin, overlap = 11, .9

plt.close("all")

# %%
rundir = "runC201509130814A/"
keys = ["PR", "PP"]
groups = {k: obspy.read(rundir + k + ".MSEED") for k in keys}
cat = obspy.read_events(rundir + "event.xml")
event = cat[0]
inventory = obspy.read_inventory(rundir + "inventory.XML")
origin = event.preferred_origin()
slat, slon = origin.latitude, origin.longitude
z_src_km = origin.depth / 1000.0
channels = "RTZ"
with open(rundir + "dbmetadata.pkl", "rb") as f:
    dbmeta = pickle.load(f)
titles = {"PR":"PREM", "PP":"BEAM 1D", "MC":"Slab 1D"}

# %% reassessing coordinates
stations = inventory[0].stations

for key in groups:
    gr = groups[key]
    for station in stations:
        loc = AttribDict(latitude=station.latitude,
                         longitude=station.longitude)
        for trace in gr.select(station=station.code):
            trace.stats.coordinates = loc
            
# %% collect
data_groups = {}
phase_list = ["SS"]  
model = obspy.taup.TauPyModel("prem")
streams_pre = {}
streams = {}
for ik, key in enumerate(groups):
    gr = groups[key]
    st = obspy.Stream()
    new_st = obspy.Stream()
    old_st = obspy.Stream()
    distances = []
    names = []
    azs = []
    # build list of accepted stations
    for stat in stations:
        rlat, rlon = stat.latitude, stat.longitude
        dist = obspy.geodetics.locations2degrees(slat, slon, rlat, rlon)
        if dist <= dmax and dist >= dmin:
            distances.append(dist)
            names.append(stat.code)
            st += gr.select(id="*%s*" % stat.code)
            _, az, _ = obspy.geodetics.gps2dist_azimuth(slat, slon, rlat, rlon)
            azs.append(az)
    # build list of arrival times
    ttimes = []
    inc_angles = []
    max_depths = []
    for dist, code in zip(distances, names):
        args = dict(source_depth_in_km=z_src_km, distance_in_degree=dist,
                    phase_list=phase_list)
        arrival = model.get_travel_times(**args)[0]
        ppoints = model.get_pierce_points(**args)[0]
        ttimes.append(arrival.time)
        inc_angles.append(arrival.incident_angle)
        max_depths.append(max([tup[-1] for tup in ppoints.pierce]))
    #
    peak_times = []
    for u, (code, ttime) in enumerate(zip(names, ttimes)):
        selection = st.select(id="*%s*" % code).copy()
        #selection.rotate('ZNE->LQT', 180, inc_angles[u])
        tr = selection.select(channel="*T*")[0]
        tr.detrend("simple")
        tr.filter("bandpass", freqmin=freqmin, freqmax=freqmax)
        reftime = tr.stats.starttime + ttime
        trimmed = tr.trim(reftime - post_cut_t, reftime + post_cut_t, 
                          pad=True, nearest_sample=False)
        trim_data = trimmed.data
        trim_data_max = np.max(trim_data)
        trim_time = trimmed.times()
        time_max = trim_time[trim_data == trim_data_max]
        peak_times += [time_max[0] - post_cut_t]

    # collect traces (aligned and non-aligned)
    data = {}
    for c in channels:
        traces = []
        times_traces = []
        for u, (code, ttime) in enumerate(zip(names, ttimes)):
            selection = st.select(id="*%s*" % code).copy()
            tr = selection.select(channel="*%s*" % c)[0]
            reftime = tr.stats.starttime + ttime + peak_times[u]
            sr = tr.stats.sampling_rate
            trimmed = tr.trim(reftime - pre_cut_t, reftime + post_cut_t, 
                              pad=True, nearest_sample=False)
            traces.append(trimmed.data)
            times_traces.append(trimmed.times(reftime=reftime))
            new_st += trimmed
            old_st += selection

        data[c] = [times_traces, traces, max_depths]
    data_groups[key] = data
    streams[key] = new_st
    streams_pre[key] = old_st

# %% create data
st_pre = streams[case].select(channel="*" + channel + "*").copy()
st_raw = st_pre.copy()
for tr in st_raw:
    tr.stats.starttime -= tr.stats.starttime
mint = min([tr.stats.endtime.timestamp for tr in st_raw])
for tr in st_raw:
    tr.trim(tr.stats.starttime, tr.stats.starttime + mint, 
             nearest_sample=False)
    tr.stats.coordinates.elevation = 0
    
st_filt = st_raw.copy()

for tr in st_filt:
    tr.detrend("simple")
    tr.filter("bandpass", freqmin=freqmin, freqmax=freqmax)

this_st = st_filt if do_filt else st_raw

data = np.c_[[tr.data for tr in this_st]]
dt = this_st[0].stats.delta
time = np.arange(0, data.shape[-1] * dt, dt)
meandata = np.mean(data, axis=1)
data = (data.T - meandata).T 
maxdata = np.max(np.abs(data), axis=1)
data_norm = (data.T / maxdata).T
#data_norm = data / data.max()

# %% plot data
plt.figure()
plt.pcolormesh(time - pre_cut_t, distances, data_norm, cmap="bone_r", 
               vmin=-.25, vmax=.25)
plt.colorbar(extend="both", label="Normalized Amplitude [a.u.]")
plt.xlabel("Time - SS theoretical arrival time [s]")
plt.ylabel("Offset [deg]")
plt.title("Seismograms for %s" % titles[case])

# %% compute vespagram
offsets = np.r_[distances] - min(distances)
v = Vespagram(offsets, time, data_norm, smin, smax, ds, dwin, overlap)
delays, slownesses, vespagram = v.compute()

#%% plot vespagram
plt.figure()
plt.pcolor(delays - pre_cut_t, -slownesses, vespagram[:,::-1], cmap="seismic", 
           vmin=-.01, vmax=0.01)
plt.ylim([-smin, -smax])
plt.colorbar(extend="both")
plt.xlabel("Time - SS theoretical arrival time [s]")
plt.ylabel("Slowness [s/deg]")
plt.title("Vespagram for %s" % titles[case])

