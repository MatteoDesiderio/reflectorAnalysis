#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 16:53:54 2023

@author: matteodesiderio
"""

from vespa import Section, Phases
import matplotlib.pyplot as plt
import numpy as np
plt.ion()

# %%
sec = Section("runC201509130814A/", "PR")
sec.collect_aligned() # to compute from the top
sec.save()
time, data, _, _ = sec.to_numpy_data()

# %%
reflector_depths = np.arange(0, 700, 100)
phs = Phases("runC201509130814A/", "PR", "reflectors-coarse", reflector_depths)

phs.get_precursors(at_reference=False)

# %%
phs.get_slownesses_t_precursors()

# %%
sec.plot_data(time, data)
phs.plot_precursors(False, color="orange")
# %%
s = phs.slowness_precursors
t = phs.time_precursors
d = phs.distances
ref_i = np.argmin(np.abs(phs.d_reference - np.r_[phs.distances]))
s_line = s[:, ref_i]
t_line = t[:, ref_i] 

# %%
sec.plot_data(time, data)
plt.figure()
plt.plot((t-sec.ttimes).T , d, "b")
tt = (t - sec.ttimes)[:, ref_i] 
s_ref = s_line[0]
for t0, slowness in zip(tt, s_line):
    if slowness > 0:
        plt.plot( (d - phs.d_reference) * (slowness -  s_ref) + t0, d, "r")
plt.xlim((-400, 40))
plt.ylim((d[-1], d[0]))

