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
plt.style.use("ggplot")

case = "PP"

# %% load
sec = Section("runC201509130814A/", case)
sec.collect_aligned() # to compute from the top
sec.save()
time, data, _, _ = sec.to_numpy_data()
sec.plot_data(time, data)

# %% init precursors
reflector_depths = np.arange(0, 1500, 5)
phs = Phases("runC201509130814A/", case, "SdS_0.0_2.5.2", reflector_depths,
             d_reference=125)

# %% g
phs.get_precursors(at_reference=True)
phs.get_slownesses_t_precursors()

# %%
#phs.plot_precursors(False, color="r", args_ref_plot={"color":"w", "ls":"--"})
plt.xlim((-440, 40))

# %%
s = phs.get_slownesses_for_vespagram()

# %%
v = Vespagram(phs.distances, time, data, s, 11, .9)
delays, slownesses, vespagram = v.compute()

# %%
vmin=-.03
vmax=0.03
lvls=np.linspace(vmin, vmax, 32)
f, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]},
                             sharex="all")
args = (delays, slownesses, vespagram)
ax1.contourf(*args, cmap="seismic", levels=lvls, extend="both")
ax1.set_ylim([slownesses.max(), slownesses.min()])
#plt.colorbar()
ax2.set_xlabel("Time - SS peak time [s]")
ax1.set_ylabel("Slowness [s/deg]")
ax1.plot(*phs.line_reduced.T, "k:")

ax1.axvline(*phs.line_reduced[reflector_depths==1000].squeeze())
ax1.axvline(*phs.line_reduced[reflector_depths==660].squeeze())
ax1.axvline(*phs.line_reduced[reflector_depths==410].squeeze())
ax1.axvline(*phs.line_reduced[reflector_depths==860].squeeze(), color="k")

ax1.set_title("Vespagram for %s" % case)

# %%
limit = -58
x, w = Vespagram.cross_section(phs.line_reduced, delays, slownesses, vespagram)
w /= w.max()
magnification = np.ones_like(x)
magnification[x <= limit] /= np.max(w[x <= limit]) 
ax2.plot(x, w * magnification, "k")
ax2.axvline(*phs.line_reduced[reflector_depths==1000].squeeze())
ax2.axvline(*phs.line_reduced[reflector_depths==660].squeeze())
ax2.axvline(*phs.line_reduced[reflector_depths==410].squeeze())
ax2.axvline(limit, linestyle="--", color="k")
