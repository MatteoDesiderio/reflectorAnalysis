#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 11:43:40 2023

@author: matteo
"""
import numpy as np
from segmentaxis import segment_axis
import obspy
from obspy import taup
import pickle
from obspy.core.util import AttribDict
import matplotlib.pyplot as plt
import os


def define_trace_location(group, stations):
    for station in stations:
        loc = AttribDict(latitude=station.latitude,
                         longitude=station.longitude)
        for trace in group.select(station=station.code):
            trace.stats.coordinates = loc

def get_path(self, nm):
    try:
        path = getattr(self, "_" + nm)  
    except AttributeError:
        path = self._where + "/" + nm
    return path
    

def getter_np(self, nm):
    path = get_path(self, nm)
    try:
        val = np.load(path + ".npy")
    except FileNotFoundError:
        val = []
    return val

def setter_np(self, value, nm):
    path = self._where + "/" + nm
    np.save(path, value)
    setattr(self,  "_" + nm, path)

class Vespagram:
    def __init__(self, offsets, time, data, slownesses, dwin, overlap=.5):
        
        if isinstance(slownesses, tuple):
            smin, smax, ds = slownesses
            slownesses = np.arange(smin, smax + ds, ds)
        elif isinstance(slownesses, np.ndarray):
            slownesses = slownesses.squeeze()
        else:
            msg = "slownesses must be either a tuple (smin, smax, ds) "
            msg += " or numpy array (with either slownesses at the reference"
            msg += " distance, or slownesses as function of distance)"
            raise TypeError(msg)

        self.slownesses = slownesses
        self.data = data
        self.dwin = dwin
        self.overlap = overlap
        self.time = time
        self.dt = np.mean(np.diff(time))
        self.windows = []
        self.offsets = offsets

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
        data = self.data
        lenwin = windows.shape[1]
        vespagram = np.zeros((len(slownesses), len(windows)))
        for i_sl, sl in enumerate(slownesses):
            for i_de, delay in enumerate(delays):
                taus = delay + sl * offsets
                #taus = hyperbola(offsets, sl, delay)
                n, beam = 0, np.zeros(lenwin)
                for signal, tau in zip(data, taus):
                    i_tau = np.argmin(np.abs(tau - time))
                    d = lenwin // 2
                    end = i_tau + d if lenwin % 2 == 0 else i_tau + d + 1
                    t_in_window = time[i_tau - d : end]
                    taum, taup = tau - halfwin, tau + halfwin
                    signal_in_window = signal[i_tau - d : end]
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
        
        self._where = self.rundir + "/" + self.mseed_file
        
        try:
            os.mkdir(self._where)
        except FileExistsError:
            pass
        
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
        
        self.model = []
        self.chosen_phase = ""

    @property
    def streams_pre(self):
        path = get_path(self, "streams_pre")
        try:
            stream = obspy.read(path.replace(".MSEED" ,"") + ".MSEED")
        except FileNotFoundError:
            stream = obspy.Stream()
        return stream
    @streams_pre.setter
    def streams_pre(self, value):
        nm = "streams_pre"
        path = self._where + "/" + nm + ".MSEED"
        value.write(path, format="MSEED")
        self._streams_pre = path
        
    @property
    def streams(self):
        path = get_path(self, "streams")
        try:
            stream = obspy.read(path.replace(".MSEED" ,"") + ".MSEED")
        except FileNotFoundError:
            stream = obspy.Stream()
        return stream
    @streams.setter
    def streams(self, value):
        name = "streams"
        path = self._where + "/" + name + ".MSEED"
        value.write(path, format="MSEED")
        self._streams = path

    @property
    def distances(self):
        return getter_np(self, "distances")
    @distances.setter
    def distances(self, value):
        setter_np(self, value, "distances")
                
    @property
    def names(self):
        return getter_np(self, "names")
    @names.setter
    def names(self, value):
        setter_np(self, value, "names")
    
    @property
    def ttimes(self):
        return getter_np(self, "ttimes")
    @ttimes.setter
    def ttimes(self, value):
        setter_np(self, value, "ttimes")
    
    @property
    def peak_times(self):
        return getter_np(self, "peak_times")
    @peak_times.setter
    def peak_times(self, value):
        setter_np(self, value, "peak_times")
        
    def get_stations_in_range(self):
        gr = self.group
        distances = []
        names = []
        st = obspy.Stream()
        # build list of accepted stations
        for stat in self.stations:
            rlat, rlon = stat.latitude, stat.longitude
            slat, slon = self.slat, self.slon
            dist = obspy.geodetics.locations2degrees(slat, slon, 
                                                     rlat, rlon)
            if dist <= self.dmax and dist >= self.dmin:
                distances.append(dist)
                names.append(stat.code)
                st += gr.select(id="*%s*" % stat.code)
                
        self.streams_pre = st
        self.distances = np.r_[distances]
        self.names = np.r_[names]
        
    def get_arrival_times(self, phase_list=["SS"], model="prem"):
        ttimes = []
        self.model = taup.TauPyModel(model)

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
        self.get_arrival_times(phase_list, model)
        self.get_max_t_diff()
        
        st = obspy.Stream()
        streams_pre = self.streams_pre
        for c in self.channels:
            for u, (code, ttime) in enumerate(zip(self.names, self.ttimes)):
                selection = streams_pre.select(id="*%s*" % code).copy()
                tr = selection.select(channel="*%s*" % c)[0]
                reftime = tr.stats.starttime + ttime + self.peak_times[u]
                sr = tr.stats.sampling_rate
                trimmed = tr.trim(reftime - self.pre_cut_t, 
                                  reftime + self.post_cut_t, 
                                  **self.trim_args)
                st += trimmed
        self.streams = st
    
    def to_numpy_data(self, channel="T", do_filt=True):
        st_pre = self.streams.copy()
        st_pre = st_pre.select(channel="*" + channel + "*")
        st_raw = st_pre.copy()
        dt = st_raw[0].stats.delta
        for tr in st_raw:
            tr.stats.starttime -= tr.stats.starttime
        mint = min([tr.stats.endtime.timestamp for tr in st_raw])
        mint = min([len(tr) for tr in st_raw]) * dt
        for tr in st_raw:
            tr.trim(tr.stats.starttime, tr.stats.starttime + int(mint), 
                     nearest_sample=False, pad=False)            
        st_filt = st_raw.copy()

        for tr in st_filt:
            tr.detrend("simple")
            tr.filter("bandpass", freqmin=self.freqmin, freqmax=self.freqmax)

        this_st = st_filt if do_filt else st_raw

        data = np.c_[[tr.data for tr in this_st]]
        time = np.arange(0, data.shape[-1] * dt, dt)
        meandata = np.mean(data, axis=1)
        data = (data.T - meandata).T 
        maxdata = np.max(np.abs(data), axis=1)
        data_norm = (data.T / maxdata).T
        
        return time - self.pre_cut_t, data_norm, data, channel
     
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
        
    def save(self):
        with open(self._where + "/section.pkl", "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    @staticmethod
    def load(path):
        with open(path + "/section.pkl", "rb") as f:
            s = pickle.load(f)
        return s

class Phases:
    def __init__(self, 
                 rundir, mseed_file, ref_model, depths_ref, d_reference=125):
        self.rundir = rundir
        self.mseed_file = mseed_file
        self._where = self.rundir + "/" + self.mseed_file        
        self.sec = Section.load(self._where)
        self.pre_cut_t = self.sec.pre_cut_t
        self.post_cut_t = self.sec.post_cut_t
        self.dmin, self.dmax = (self.sec.dmin, self.sec.dmax)
        self.z_src_km = self.sec.z_src_km
        self.ttimes = self.sec.ttimes
        self.distances = self.sec.distances
        self.d_reference = d_reference
        self.depths_ref = depths_ref
        self.ref_model = taup.TauPyModel(ref_model)
        self.chosen_phase = self.sec.chosen_phase
        
        self.precursors = []
        
    @property
    def slowness_precursors(self):
        return getter_np(self, "slowness_precursors")
    @slowness_precursors.setter
    def slowness_precursors(self, value):
        setter_np(self, value, "slowness_precursors")
        
    @property
    def time_precursors(self):
        return getter_np(self, "time_precursors")
    @time_precursors.setter
    def time_precursors(self, value):
        setter_np(self, value, "time_precursors")
        
    @property
    def line(self):
        return getter_np(self, "line")
    @line.setter
    def line(self, value):
        setter_np(self, value, "line")
        
    @property
    def line_reduced(self):
        return getter_np(self, "line_reduced")
    @line_reduced.setter
    def line_reduced(self, value):
        setter_np(self, value, "line_reduced")
        
    @property
    def distances_reduced(self):
        return getter_np(self, "distances_reduced")
    @distances_reduced.setter
    def distances_reduced(self, value):
        setter_np(self, value, "distances_reduced")
        
    @property
    def times_reduced(self):
        return getter_np(self, "times_reduced")
    @times_reduced.setter
    def times_reduced(self, value):
        setter_np(self, value, "times_reduced")
    
    def get_precursors(self, chosen_phase="SS", reflector_phase="S^%iS", 
                       at_reference=True):
        
        if at_reference:
            distances = [self.d_reference]
            print("ah!")
        else:
            distances = self.distances
        
        cp = [self.chosen_phase if chosen_phase is None else chosen_phase]
        mod = self.ref_model
        depths_ref = self.depths_ref
        precursors = []
        for i, z in enumerate(depths_ref):
            print(z)
            rp = [reflector_phase % z]
            # build list of arrival times
            arrivals = []
            for j, dist in enumerate(distances):
                args = dict(source_depth_in_km = self.z_src_km, 
                            distance_in_degree = dist)
                arr_phase = mod.get_travel_times(**args, phase_list=cp)[0]
                try:
                    arr = mod.get_travel_times(**args, phase_list=rp)[0]
                    """
                    condition1 = (arr.time <= arr_phase.time + self.post_cut_t)
                    condition2 = (arr.time >= arr_phase.time - self.pre_cut_t )
                    in_window = condition1 and condition2
                    """
                    in_window = True
                except IndexError:
                    in_window = False
                
                if in_window:    
                    arrivals.append(arr)
                else:
                    precursors.append(np.nan)
                    
            self.precursors.append(arrivals)
                    
        # wont work w getter and setters 
        # if you dont change from list to array
        # self.precursors = precursors 
        
    def get_slownesses_t_precursors(self):
        precursors = self.precursors
        shape = (len(precursors), len(precursors[0]))
        
        slowness_precursors = np.zeros(shape)
        time_precursors = np.zeros(shape)
    
        for i, precursor in enumerate(precursors):
            for j, arr in enumerate(precursor):
                if not (arr is np.nan):
                    slow = arr.ray_param_sec_degree
                    t = arr.time
                else:
                    slow = np.nan
                    t = np.nan
                slowness_precursors[i,j] = slow
                time_precursors[i,j] = t
                
        self.slowness_precursors = slowness_precursors
        self.time_precursors = time_precursors
        if shape[1] > 1:
            ref_i = np.argmin(np.abs(self.d_reference - self.distances))
            s_line = slowness_precursors[:, ref_i]
            t_line = time_precursors[:, ref_i] 
            self.line = np.c_[t_line, s_line]
            
            times_reduced = time_precursors - self.ttimes
            self.times_reduced = times_reduced        
            self.distances_reduced = self.distances - self.d_reference 
            t_line_red = times_reduced[:, ref_i] 
            s_line_red = s_line - s_line[0]
            self.line_reduced = np.c_[t_line_red, s_line_red]
        else:
            ref_i = np.argmin(np.abs(self.d_reference - self.distances))
            
            s_line = slowness_precursors[:, 0]
            t_line = time_precursors[:, 0] 
            self.line = np.c_[t_line, s_line]
            
            times_reduced = time_precursors - self.ttimes
            t_line_red = times_reduced[:, ref_i] 
            s_line_red = s_line - s_line[0]
            self.line_reduced = np.c_[t_line_red, s_line_red]
            self.times_reduced = t_line_red
            self.distances_reduced = self.distances - self.d_reference


    def plot_precursors(self, show_ref=True, annotate=True, args_ref_plot={}, 
                        **args_plot):
        times_reduced = self.times_reduced
        distances = self.distances
        if times_reduced.ndim > 1:
            plt.plot(times_reduced.T, distances, **args_plot)
        else:
            d_ref = np.ones(len(times_reduced)) * self.d_reference
            plt.plot(times_reduced, d_ref,  "o", **args_plot)
            
        if show_ref:
            t, s = self.line_reduced.T
            d = self.distances_reduced

            plt.plot( d[:, np.newaxis] * s + t, distances, 
                     **args_ref_plot)
            
    def get_slownesses_for_vespagram(self):
        return self.line_reduced[:,-1]
        """
        all_times = np.c_[time_precursors]
        all_time_diffs = np.c_[time_precursors] - np.r_[SS_times]
        all_slownesses = np.c_[slowness_precursors]
        d_reference = 125
        ref_i = np.argmin(np.abs(d_reference - np.r_[distances]))
    
        time_reference = all_time_diffs[:, ref_i]
        slownesses_reference = all_slownesses[:, ref_i]
        """
