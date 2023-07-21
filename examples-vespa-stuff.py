#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 16:53:54 2023

@author: matteodesiderio
"""

from vespa import Section, Phases
import matplotlib.pyplot as plt
import numpy as np

# %%
sec = Section("runC201509130814A/", "PR")
#sec.collect_aligned() # to compute from the top
sec.save()
time, data, _, _ = sec.to_numpy_data()
sec.plot_data(time, data)

# %%
reflector_depths = np.arange(100, 3000, 100)
phs = Phases("runC201509130814A/", "PR", "reflectors-coarse", reflector_depths)

phs.get_precursors()

# %%
phs.get_slownesses_t_precursors()

# %%

ref_i = np.argmin(np.abs(phs.d_reference - np.r_[phs.distances]))
d = phs.distances[ref_i]
s = phs.slowness_precursors[:, ref_i]
t = phs.time_precursors[:, ref_i]
