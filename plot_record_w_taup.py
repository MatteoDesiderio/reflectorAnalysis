#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 10:45:20 2023

@author: matteo
"""

import instaseis
from obspy.core.util import AttribDict
from obspy.taup import plot_travel_times, plot_ray_paths
import obspy
import matplotlib.pyplot as plt
import numpy as np
from obspy import geodetics
from string import ascii_uppercase
import pickle
plt.style.use("ggplot")
plt.ion()

# %%
rundir = "runExplosion_90N_0W/"
keys = ["MC", "PP", "PR"]
groups = {k:obspy.read(rundir + k + ".MSEED") for k in keys}
cat = obspy.read_events(rundir + "event.xml")
event = cat[0]
inventory = obspy.read_inventory(rundir + "inventory.XML")
origin = event.preferred_origin()
slat, slon = origin.latitude, origin.longitude
z_src_km = origin.depth / 1000.0

# %% reassing coordinates
inventory = obspy.read_inventory(rundir + "inventory.XML")
stations = inventory[0].stations
for key, c in zip(groups, "rkg"):
    gr = groups[key]
    for station in stations:
        loc = AttribDict(latitude=station.latitude,
                         longitude=station.longitude)
        for trace in gr.select(station=station.code):
            trace.stats.coordinates = loc

# %%
plt.close("all")
phase_list = ["PP", "SS", "PS", "SP"] #("ttbasic",) 
#phase_list = ("ttbasic",) 
for key, c in zip(groups, ["k", "k", "k"]):
    thrx = 30
    bg_model = obspy.taup.TauPyModel("prem")
    fig = plt.figure(figsize=[8, 8])
    ax = plot_travel_times(source_depth=z_src_km, phase_list=phase_list,
                           fig=fig, model="prem")
    [l.set_data(l.get_data()[0], l.get_data()[-1]*60) for l in ax.lines]
    ax.set_ylabel("Time (seconds)")
    
    gr = groups[key]
    gr.plot(fig=fig, ax=fig.axes[0], type="section", draw=False, 
            ev_coord=(slat, slon),
            dist_degree=True, color=c, offset_max=180, scale=3)
    ax.set_xlim([90, 180])
    fig.savefig(rundir + key + ".pdf")

#ax.grid()
#ax.axvline(thrx)