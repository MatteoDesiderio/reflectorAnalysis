#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 11:43:40 2023

@author: matteo
"""
import numpy as np
from segmentaxis import segment_axis
import obspy
import pickle
from obspy.core.util import AttribDict
import matplotlib.pyplot as plt


def hyperbola(x, slowness, t0):
    return np.sqrt((x * slowness) ** 2 + t0 ** 2)

def define_trace_location(group, stations):
    for station in stations:
        loc = AttribDict(latitude=station.latitude,
                         longitude=station.longitude)
        for trace in group.select(station=station.code):
            trace.stats.coordinates = loc

class Vespagram:
    def __init__(self, offsets, time, data, smin, smax, ds, dwin, overlap=.5):
        self.data = data
        self.smin = smin
        self.smax = smax
        self.dwin = dwin
        self.overlap = overlap
        self.time = time
        self.dt = np.mean(np.diff(time))
        self.windows = []
        self.offsets = offsets
        self.slownesses = np.arange(smin, smax + ds, ds)

    def prepare(self):
        dwin_samples = int(self.dwin / self.dt) + 1 
        d = int(self.dwin * self.overlap)
        self.windows = segment_axis(self.time, self.dwin, d)

    def compute(self):
        self.prepare()
        time = self.time
        windows = self.windows
        offsets = self.offsets
        slownesses = self.slownesses
        delays = np.mean(windows, axis=1)
        halfwin = np.abs(windows[0, 0] - windows[0, -1]) / 2
        data = self.data[:, ::-1]
        lenwin = windows.shape[1]
        vespagram = np.zeros((len(slownesses), len(windows)))
        for i_sl, sl in enumerate(slownesses):
            for i_de, delay in enumerate(delays):
                #taus = delay + sl * offsets
                taus = hyperbola(offsets, sl, delay)
                n, beam = 0, np.zeros(len(windows[0]))
                for signal, tau in zip(data, taus):
                    i_tau = np.argmin(np.abs(tau - time))
                    d = int(halfwin // self.dt)
                    t_in_window = time[i_tau - d : i_tau + d + 1]
                    taum, taup = tau - halfwin, tau + halfwin
                    signal_in_window = signal[i_tau - d : i_tau + d + 1]
                    lendiff = lenwin - len(signal_in_window)
                    if taup > time.max() and taum <= time.max():
                        beam += np.r_[signal_in_window, np.zeros(lendiff)]
                        n += 1
                    elif taum > time.max():
                        beam += np.zeros(lenwin)
                    else:
                        beam += signal_in_window
                        n += 1
                beam /= n
               
                vespagram[i_sl, i_de] = np.mean(beam)
                
        return delays, slownesses, vespagram

class Section:
    def __init__(self, rundir, mseed_file, channels="RTZ", pre_cut_t=440, 
                 post_cut_t=40, dmin=100, dmax=178, freqmin=0.02, freqmax=0.1,
                 trim_args={"pad":True, "nearest_sample":False}):
        # data collection
        self.rundir = rundir
        self.pre_cut_t = pre_cut_t
        self.post_cut_t = post_cut_t
        self.dmin, self.dmax = dmin, dmax
        self.freqmin, self.freqmax= freqmin, freqmax
        self.trim_args = trim_args
        
        self.mseed_file = mseed_file.replace(".MSEED", "")
        self.group = obspy.read(rundir + self.mseed_file + ".MSEED")
        cat = obspy.read_events(rundir + "event.xml")
        self.event = cat[0]
        inventory = obspy.read_inventory(rundir + "inventory.XML")
        self.origin = self.event.preferred_origin()
        self.slat, self.slon = self.origin.latitude, self.origin.longitude
        self.z_src_km = self.origin.depth / 1000.0
        self.channels = channels
        self.stations = inventory[0].stations
        
        with open(rundir + "dbmetadata.pkl", "rb") as f:
            dbmeta = pickle.load(f)
            
        define_trace_location(self.group, self.stations)
        
        self.streams_pre = obspy.Stream()
        self.streams = obspy.Stream()
        
        self.distances = []
        self.names = []
    
        self.ttimes = []
        self.peak_times = []
        
        self.model = []
        self.chosen_phase = ""
        
    def get_stations_in_range(self):
        gr = self.group
        distances = []
        names = []
        # build list of accepted stations
        for stat in self.stations:
            rlat, rlon = stat.latitude, stat.longitude
            slat, slon = self.slat, self.slon
            dist = obspy.geodetics.locations2degrees(slat, slon, 
                                                     rlat, rlon)
            if dist <= self.dmax and dist >= self.dmin:
                distances.append(dist)
                names.append(stat.code)
                self.streams_pre += gr.select(id="*%s*" % stat.code)
                    
        self.distances = np.r_[distances]
        self.names = np.r_[names]
        
    def get_arrival_times(self, phase_list=["SS"], model="prem"):
        ttimes = []
        self.model = obspy.taup.TauPyModel(model)

        for dist, code in zip(self.distances, self.names):
            args = dict(source_depth_in_km=self.z_src_km, 
                        distance_in_degree=dist,
                        phase_list=phase_list)
            arrival = self.model.get_travel_times(**args)[0]
            ttimes.append(arrival.time)
        self.ttimes = np.r_[ttimes]
    
    def get_max_t_diff(self):
        peak_times = []
        st = self.streams_pre
        for u, (code, ttime) in enumerate(zip(self.names, self.ttimes)):
            selection = st.select(id="*%s*" % code).copy()
            ch = "*T*" if "T" in self.channels else "*E*"
            tr = selection.select(channel=ch)[0]
            tr.detrend("simple")
            tr.filter("bandpass", freqmin=self.freqmin, 
                                  freqmax=self.freqmax)
            reftime = tr.stats.starttime + ttime
            trimmed = tr.trim(reftime - self.post_cut_t, 
                              reftime + self.post_cut_t, 
                              **self.trim_args)
            trim_data = trimmed.data
            trim_data_max = np.max(trim_data)
            trim_time = trimmed.times()
            time_max = trim_time[trim_data == trim_data_max]
            peak_times += [time_max[0] - self.post_cut_t]
        self.peak_times = np.r_[peak_times]
    
    def clear_phase_arrival(self):
        self.ttimes = []
        self.peak_times = []
    
    def collect_aligned(self, phase="SS", model="prem"):
        self.get_stations_in_range()
        
        self.chosen_phase = phase
        phase_list = [phase]
        if not self.ttimes:
            self.get_arrival_times(phase_list, model)
        else:
            print("Already computed %s arrivals for %s" % (phase, model[:-2]))
        
        if not self.peak_times:
            self.get_max_t_diff()
        else:
            print("Already computed time for %s peak" % phase)
       
        for c in self.channels:
            traces = []
            times_traces = []
            for u, (code, ttime) in enumerate(zip(self.names, self.ttimes)):
                selection = self.streams_pre.select(id="*%s*" % code).copy()
                tr = selection.select(channel="*%s*" % c)[0]
                reftime = tr.stats.starttime + ttime + self.peak_times[u]
                sr = tr.stats.sampling_rate
                trimmed = tr.trim(reftime - self.pre_cut_t, 
                                  reftime + self.post_cut_t, 
                                  **self.trim_args)
                traces.append(trimmed.data)
                times_traces.append(trimmed.times(reftime=reftime))
                self.streams += trimmed
    
    def to_numpy_data(self, channel="T", do_filt=True):
        st_pre = self.streams.select(channel="*" + channel + "*").copy()
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
            tr.filter("bandpass", freqmin=self.freqmin, freqmax=self.freqmax)

        this_st = st_filt if do_filt else st_raw

        data = np.c_[[tr.data for tr in this_st]]
        dt = this_st[0].stats.delta
        time = np.arange(0, data.shape[-1] * dt, dt)
        meandata = np.mean(data, axis=1)
        data = (data.T - meandata).T 
        maxdata = np.max(np.abs(data), axis=1)
        data_norm = (data.T / maxdata).T
        
        return channel, time - self.pre_cut_t, data_norm, data
     
    def plot_data(self, time, data, close_all=True,
                  args={"cmap":"bone_r", "vmin":-0.25, "vmax":0.25}):
        if close_all:
            plt.close("all")
            
        plt.figure()
        plt.pcolormesh(time, self.distances, data, **args)
        plt.colorbar(extend="both", label="Normalized Amplitude [a.u.]")
        xlab = "Time - %s theoretical arrival time [s]"
        plt.xlabel(xlab %  self.chosen_phase)
        plt.ylabel("Offset [deg]")
        plt.title("Seismograms for %s" % self.mseed_file)
