#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 10:45:20 2023

@author: matteo
"""

from obspy.core.util import AttribDict
import obspy
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from obspy.signal import array_analysis as aran
from obspy.signal.array_analysis import dump

plt.style.use("ggplot")
plt.ion()


def rms(x, **kwargs):
    return np.sqrt(np.sum(x ** 2, **kwargs) / len(x))


class AGC:
    def __init__(self, t, x, n, target=100):
        self.t = t
        self.x = x
        self.n = n
        self.target = target

    def create(self):
        dt = t[-1] - t[-2]
        t_ = [self.t[i:i+self.n] for i in range(0, len(self.x), self.n)]
        g_ = [self.x[i:i+self.n] for i in range(0, len(self.x), self.n)]
        dn = len(t_[-2]) - len(t_[-1])
        if dn > 0:
            pad_t = np.arange(t_[-1][-1], t_[-1][-1] + dn * dt, dt) + dt
            t_[-1] = np.concatenate([t_[-1], pad_t])
            g_[-1] = np.concatenate([g_[-1], [g_[-1][-1]] * dn])
        t_ = np.c_[t_]
        g_ = np.c_[g_]
        t_ = np.mean(t_, axis=1)
        g_ = self.target / rms(g_, axis=1)
        return t_, g_

    def apply(self, **kwargs):
        t_, g_ = self.create()
        f = interpolate.interp1d(t_, g_, bounds_error=False,
                                 fill_value="extrapolate", **kwargs)
        return f(self.t) * self.x


# %%
rundir = "runC201509130814A/"
keys = ["MC", "PP", "PR"]
groups = {k: obspy.read(rundir + k + ".MSEED") for k in keys}
cat = obspy.read_events(rundir + "event.xml")
event = cat[0]
inventory = obspy.read_inventory(rundir + "inventory.XML")
origin = event.preferred_origin()
slat, slon = origin.latitude, origin.longitude
z_src_km = origin.depth / 1000.0
channels = "RTZ"

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
plt.close("all")
dt = 60
dmin, dmax = (100, 180)
data_groups = {}
phase_list = ["SS"]  # ["PP", "SS", "P", "S", "PS", "SP"] #("ttbasic",)
model = obspy.taup.TauPyModel("prem")
streams = {}
for ik, key in enumerate(groups):
    gr = groups[key]
    st = obspy.Stream()
    new_st = obspy.Stream()
    distances = []
    names = []
    azs = []
    for stat in stations:
        rlat, rlon = stat.latitude, stat.longitude
        dist = obspy.geodetics.locations2degrees(slat, slon, rlat, rlon)
        if dist <= dmax and dist >= dmin:
            distances.append(dist)
            names.append(stat.code)
            st += gr.select(id="*%s*" % stat.code)
            _, az, _ = obspy.geodetics.gps2dist_azimuth(slat, slon, rlat, rlon)
            azs.append(az)
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

    data = {}
    for c in channels:
        traces = []
        times_traces = []
        for u, (code, ttime) in enumerate(zip(names, ttimes)):
            selection = st.select(id="*%s*" % code).copy()
            #selection.rotate('ZNE->LQT', 180, inc_angles[u])
            tr = selection.select(channel="*%s*" % c)[0]
            tr.filter("bandpass", freqmin=1/50, freqmax=1/10)
            reftime = tr.stats.starttime + ttime
            sr = tr.stats.sampling_rate
            trimmed = tr.trim(reftime - 8*dt, reftime + dt, pad=True,
                              nearest_sample=False)
            traces.append(trimmed.data)
            times_traces.append(trimmed.times(reftime=reftime))
            new_st += trimmed

        data[c] = [times_traces, traces, max_depths]
    data_groups[key] = data
    streams[key] = new_st

# %%
plt.close("all")
fig, axs = plt.subplots(1, len(keys), sharey="all", sharex="all")
gaps = []
mult = 1
gain = 1
n = 12
do_agc = False
titles = dict(PP="Blob", MC="Slab", PR="PREM")
for i, mod in enumerate(data_groups):
    st = obspy.Stream()
    data = data_groups[mod]["T"]
    dmax = np.nanmax(np.concatenate(data[1]))

    for ic, code in enumerate(names):
        ink = int(data[-1][ic] <= 2000 and data[-1][ic] >= 1000) / 2
        color = (ink, .2, .5)
        if ic % 1 == 0:
            d_ = data[1][ic] / np.nanmax(data[1][ic])
            t = data[0][ic]
            agc = AGC(t, d_, n, gain)
            d = agc.apply() if do_agc else d_
            ax = axs[i]
            ax.set_title(titles[mod])
            ax.plot(t, mult * d + ic, c="teal")
            #ax.fill_between(t, ic, mult * d + ic, color=color)
            #ax.text(t[0], d[0] + ic, code, ha="right", va="center")
            #ax.set_ylabel("Displ / %.0e / m " % dmax)
            ax.set_yticklabels("")
