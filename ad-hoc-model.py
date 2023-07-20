#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 17:59:19 2023

@author: matteodesiderio
"""

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from obspy.taup import taup_create

model = np.loadtxt("prem.nd")
dz = 15 # new resolution
#  mantle, core, inner core depths
discs = np.r_[15, 24.40, 2891.00, 5149.50]
names_discs = ["", "mantle", "outer-core", "inner-core"]

is_coarse = True

namemodel = "reflectors"
namemodel += "-coarse.nd" if is_coarse else ".nd" 

#%%

discs1, discs2 = np.r_[0, discs], np.r_[discs, np.inf]
z = model[:, 0]

actual_discs = [] # maybe needed
for i, z_i in enumerate(z):
    try:
        is_disc = z_i == z[i + 1]
    except IndexError:
        pass
    if is_disc:
        actual_discs += [z_i]

intervals = [(z >= z1) & (z <= z2) for z1, z2 in zip(discs1, discs2) ]
layers = [model[interval] for interval in intervals]

lower_cuts = [None if i == 0 else 1 for i in range(len(layers))]
upper_cuts = [None if i == len(layers) -1 else -1 for i in range(len(layers))]

layers = [l[lc:uc] for l, lc, uc in zip(layers, lower_cuts, upper_cuts)]
z_layers = [l[:,0] for l in layers]
interpolators = [interp1d(l[:,0], l[:,1:], axis=0) for l in layers]

znew_layers = [np.arange(zl.min(), zl.max(), dz ) for zl in z_layers] 

for il, znl in enumerate(znew_layers):
    try:
        znew_layers[il] = np.r_[znl, znew_layers[il + 1][0]]
    except IndexError:
        znew_layers[il] = np.r_[znl, z_layers[il][-1]]

new_layers = [i(zl) for i, zl in zip(interpolators, znew_layers)]

plt.close("all")
plt.figure()
[plt.plot(znl, nl[:,1]) for znl, nl in zip(znew_layers, new_layers) ]

if is_coarse:
    new_layers = [l[:, 1:] for l in layers]
    znew_layers = z_layers

val_example = []
with open(namemodel, 'w') as f:
    for j, (znl, nl) in enumerate(zip(znew_layers, new_layers)):
        jj = min(len(discs) - 1, j)
        for i, (z_i, l_i) in enumerate(zip(znl, nl)):
            fmtstr = "%.2f  %.5f  %.5f  %.5f  %.1f  %.1f"
            values = np.array(np.r_[z_i, l_i])
            line = fmtstr % (*values,)
            f.write(line)
            f.write('\n')
            val_example.append(values)
            if i > 0 and i < len(nl) - 1 and z_i < 2891:
                # be careful: should not exceed the value of the deeper point
                _values = np.array(np.r_[z_i, l_i * 1.00011])
                _line = fmtstr % (*_values,)
                f.write(_line)
                f.write('\n')
                val_example.append(_values)
            if z_i == discs[jj] and i == len(nl) - 1:
                ndisc = names_discs[jj]
                f.write(ndisc)
                if ndisc != "":
                    f.write('\n')

val_example = np.vstack(val_example)

# %% create model
taup_create.build_taup_model(namemodel)

