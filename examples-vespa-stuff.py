#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 16:53:54 2023

@author: matteodesiderio
"""

from vespa import Section, Phases, Vespagram
import matplotlib.pyplot as plt
import numpy as np
plt.ion()

# %%
sec = Section("runC201509130814A/", "PR")
sec.collect_aligned() # to compute from the top
sec.save()
time, data, _, _ = sec.to_numpy_data()

# %%
reflector_depths = np.arange(0, 1600, 100)
phs = Phases("runC201509130814A/", "PR", "reflectors-coarse", reflector_depths,
             d_reference=139)

# %%
phs.get_precursors(at_reference=True)
phs.get_slownesses_t_precursors()

# %%
sec.plot_data(time, data)
phs.plot_precursors(True, color="orange", args_ref_plot={"color":"teal"})
plt.xlim((-440, 40))

# %%
s = phs.get_slownesses_for_vespagram()

# %%
offsets = np.r_[phs.distances] - min(phs.distances)
v = Vespagram(offsets, time-min(time), data, -s, 11, .9)
delays, slownesses, vespagram = v.compute()

