#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 13:55:29 2023
This script is to load 3 databases, then download an event from 
your favourite catalog via iris and compute the waveforms 
for a number of regularly spaced stations around its centroid   
@author: matteo
"""
import requests
import pickle 
import os
import numpy as np
from random import choice
import matplotlib.pyplot as plt
import instaseis
import obspy
from obspy.clients.fdsn import Client
from obspy import geodetics
from obspy import UTCDateTime
from obspy.core.util import AttribDict
from obspy.core.inventory import Network, Station, Channel

plt.style.use("ggplot")
plt.ion()
dyneCm_to_Nn = 1e-7

address = "https://ds.iris.edu/spudservice/item?"
query = "evtminlat={}&evtmaxlat={}&evtminlon={}&evtmaxlon={}" + \
        "&evtmindepth={}&evtmaxdepth={}&evtstartdate={}&evtenddate={}" + \
        "&evtminmag={}&evtmaxmag={}"
src_url = "http://ds.iris.edu/spudservice/momenttensor/gcmtid/%s/ndk"
      
# %% User parameters
# database location
root = '/home/matteo/axisem_latest/axisem-master/SOLVER/'
runPR = 'prem_iso_5s-min90/'
runMC = 'prem_iso_slab_d390z10_5s-min90/database/'
runPP = 'prem_iso_blob_d350z1_5s-min90/database/'
kwargs = dict(read_on_demand=True, buffer_size_in_mb=100)
# source
minlat = -90
maxlat = 90
minlon = -180
maxlon = 180
mindepth = 10
maxdepth = 30
startdate = UTCDateTime(2010, 1, 1, 1).format_iris_web_service()
enddate = UTCDateTime(2022, 1, 1, 1).format_iris_web_service()
minmag = 6.5
maxmag = 7.5
explosion = False
# receivers 
nw_name = "X"
rminlat = -89
rmaxlat = 89
rminlon = -179
rmaxlon = 179
n_lat = 180 # n_lat receivers between rminlat and rmaxlat deg
n_lon = 1 # n_lon receivers between rminlon and rmaxlon deg
dmin, dmax = 100, 180 # expunge receivers between these radii [deg]
inventory_name = "inventory"
# other
label = "" # other useful info to store in the run name

# %%load databases
dbPR = instaseis.open_db(root+runPR, **kwargs)
dbMC = instaseis.open_db(root+runMC, **kwargs)
dbPP = instaseis.open_db(root+runPP, **kwargs)
databases = dict(MC=dbMC, PP=dbPP, PR=dbPR)
dbmetadata = {k:databases[k].__str__() for k in databases}

# %% candidate sources
url = address + query.format(minlat, maxlat, minlon, maxlon, mindepth, 
                                 maxdepth, startdate, enddate, minmag, maxmag)
req = requests.get(url)
xml = req.text
req.close()
xml = xml.rsplit("\n")
condition = lambda x : "&lt;EventName&gt;" in x and "&lt;/EventName&gt;" in x
gcmtids = [l.rsplit("&gt;")[1].rsplit("&lt;")[0] for l in xml if condition(l)]

# %% source
n_id = 0
gcmtid = gcmtids[n_id]
cat = obspy.read_events(src_url % gcmtid)
event = cat[0]
src = instaseis.Source.parse(src_url % gcmtid)
obspy.imaging.beachball.beachball(src.tensor/src.M0)

# re placing the source, this is not the real earth
src.latitude = -90
src.longitude = 0
# resetting 
src.origin_time = UTCDateTime(0)

# for plotting to be consistent
for o in cat[0].origins:
    o.latitude, o.longitude = src.latitude, src.longitude

if explosion:
    src.m_rr, src.m_pp, src.m_tt = [src.M0] * 3
    src.m_rp, src.m_tp, src.m_rt = [0] * 3
    abs_lat, abs_lon = np.abs([src.latitude, src.longitude])
    string_lat = "%sN" % abs_lat if src.latitude > 0 else "%sS" % abs_lat
    string_lon = "%sE" % abs_lon if src.longitude > 0 else "%sW" % abs_lon
    gcmtid = "Explosion_{}_{}".format(string_lat, string_lon)
    
print(gcmtid)

# %% make directories
runname = "run%s/" % (gcmtid + "-" + label) 
os.mkdir(runname)
event.write(runname + "event.xml", format="QUAKEML")
with open(runname + 'dbmetadata.pkl', 'wb') as f:
    pickle.dump(dbmetadata, f)
    
# %% receivers
inventory = obspy.Inventory()
inventory.networks = [Network(nw_name)]
lat_minmax = (rminlat, rmaxlat)
lon_minmax = (rminlon, rmaxlon) if n_lon > 1 else (0, 0)
ntot = n_lat * n_lon

lats, lons = np.linspace(*lat_minmax, n_lat), np.linspace(*lon_minmax, n_lon)
lats, lons = np.meshgrid(lats, lons)

receivers = np.empty(ntot, dtype=instaseis.Receiver)
for i, (lat, lon) in enumerate(zip(lats.flatten(), lons.flatten())):
    code = "%s%i" % ("X", i)
    receivers[i] = instaseis.Receiver(latitude=lat, longitude=lon,
                                      network=nw_name, station=code)
    
    chans = [Channel(code="%s" % i,  location_code="", latitude=lat,
                     longitude=lon, elevation=0, depth=0) for i in "ZRT"]
    inventory.networks[0].stations.append(Station(code, lat, lon, 0, 
                                                  channels=chans))
    
    

# %% expunge receivers
flags = []
for i, rec in enumerate(receivers):
    rlat, rlon = rec.latitude, rec.longitude
    slat, slon = src.latitude, src.longitude
    dist = geodetics.locations2degrees(rlat, rlon, slat, slon)
    flag = [(dist > dmin and dist < dmax)]
    flags += flag
    station = inventory.networks[0].stations[i]
    # just to stay safe, make sure they are the same station
    assert(station.code == rec.station)
    station.is_active = flag
inventory.write(runname + inventory_name + ".XML", "STATIONXML")
    
# %% prelim plotting
flags = np.r_[flags]
map_fig = inventory.plot("global", show=False, label=False, size=10)
cat.plot(fig=map_fig)
obspy.imaging.beachball.beachball(src.tensor/src.M0)

# %% streams
groups = {key:obspy.Stream() for key in databases}
for key in databases:
    print("Extracting seismograms from DB", key)
    db = databases[key]
    for flag, rec in zip(flags, receivers):
        if flag:
            print(rec)
            st = db.get_seismograms(src, rec, "ZRT").sort(["channel"])
            loc = AttribDict(latitude=rec.latitude,
                             longitude=rec.longitude)
            for tr in st:
                tr.stats.coordinates = loc
            groups[key] += st
            
# %% saving
for key, c in zip(groups, "rkg"):
    gr = groups[key]
    gr.write(runname + key + ".MSEED", format="MSEED")

# %% 
fig = plt.figure()
for key, c in zip(groups, "rkg"):
    gr = groups[key]
    if not (key == "PR" ):
        gr.plot(fig=fig, type="section", draw=False, ev_coord=(slat, slon),
                dist_degree=True, color=c, label=key)


