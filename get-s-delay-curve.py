#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 10:44:50 2023

@author: matteodesiderio
"""
import time as time_module
from obspy.core.util import AttribDict
import obspy
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from obspy.signal import array_analysis as aran
from obspy import taup
import matplotlib
from vespa import Vespagram
import pickle

plt.style.use("ggplot")
plt.ion()
plt.close("all")
cmap = matplotlib.cm.get_cmap("magma")

# %% parameters
# data collection
pre_cut_t = 440
post_cut_t = 40
dmin, dmax = (100, 178)
case = "PP"
freqmin, freqmax= 1/50, 1/10

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
inventory = obspy.read_inventory(rundir + "inventory.XML")
stations = inventory[0].stations

for key, c in zip(groups, "rkg"):
    gr = groups[key]
    for station in stations:
        loc = AttribDict(latitude=station.latitude,
                         longitude=station.longitude)
        for trace in gr.select(station=station.code):
            trace.stats.coordinates = loc
            
# %% collect
data_groups = {}
ref_model = taup.TauPyModel("reflectors-coarse")

distances = []
names = []
for stat in stations:
    rlat, rlon = stat.latitude, stat.longitude
    dist = obspy.geodetics.locations2degrees(slat, slon, rlat, rlon)
    if dist <= dmax and dist >= dmin:
        distances.append(dist)
        names.append(stat.code)
        _, az, _ = obspy.geodetics.gps2dist_azimuth(slat, slon, rlat, rlon)

SS_times = []
SS_slownesses = []
for dist in distances:
    args = dict(source_depth_in_km=z_src_km, distance_in_degree=dist)
    arrival_SS = ref_model.get_travel_times(**args, phase_list=["SS"])[0]
    SS_times += [arrival_SS.time]
    SS_slownesses += [arrival_SS.ray_param_sec_degree]
    
precursors = []
depths_ref = np.arange(0, 2900, 100)
for i in depths_ref:
    # build list of arrival times
    arrivals = []
    for dist in distances:
        args = dict(source_depth_in_km=z_src_km, distance_in_degree=dist)
        arrival_SS = ref_model.get_travel_times(**args, phase_list=["SS"])[0]
        try:
            arrival = ref_model.get_travel_times(**args, 
                                                 phase_list=["S^%iS"%i])[0]
            in_window = (arrival.time <= arrival_SS.time + post_cut_t  and
                         arrival.time >= arrival_SS.time - pre_cut_t )
        except IndexError:
            in_window = False
        
        if in_window:    
            arrivals.append(arrival)
        else:
            arrivals.append([])
    precursors.append(arrivals)

# %%
slowness_precursors = []
time_precursors = []

for precursor in precursors:
    slownesses = []
    times = []
    for ar in precursor:
        if ar:
            ang = ar.takeoff_angle * np.pi / 180
            slow = ar.ray_param_sec_degree
            slownesses.append(slow)
            times.append(ar.time)
        else:
            slownesses.append(np.nan)
            times.append(np.nan)
    slowness_precursors.append(np.r_[slownesses])
    time_precursors.append(np.r_[times])


all_times = np.c_[time_precursors]
all_time_diffs = np.c_[time_precursors] - np.r_[SS_times]
all_slownesses = np.c_[slowness_precursors]
d_reference = 125
ref_i = np.argmin(np.abs(d_reference - np.r_[distances]))

time_reference = all_time_diffs[:, ref_i]
slownesses_reference = all_slownesses[:, ref_i]

