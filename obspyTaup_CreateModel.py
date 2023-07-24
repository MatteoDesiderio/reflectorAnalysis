#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 11:49:45 2023

@author: matteodesiderio
"""
from obspy.taup import taup_create
from sys import argv

namemodel = argv[1]
taup_create.build_taup_model(namemodel)
