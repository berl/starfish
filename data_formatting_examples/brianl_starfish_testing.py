
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'notebook')
import format_local_Allen_data as flad
from  importlib import reload

import argparse
import io
import json
import os
import glob
import re
import zipfile,time
from typing import Mapping, Tuple, Union

import numpy as np
import requests
from skimage.io import imread
from slicedimage import ImageFormat

from starfish import Codebook
from starfish.experiment.builder import FetchedTile, TileFetcher
from starfish.experiment.builder import write_experiment_json
from starfish.types import Coordinates, Features, Axes, Number
from starfish.util.argparse import FsExistsType



import matplotlib.pyplot as plt
import numpy as np
import os,json

from starfish import data, FieldOfView
from starfish.types import Axes
from starfish import Experiment
from importlib import reload

from starfish.spots import SpotFinder
from starfish.image import Filter
import starfish.plot


import starfish.display


SHAPE = 2048, 2048
PIXEL_SIZE = 1e-7
Z_SPACING = 3e-7
TC_STRING = "TileConfig_405.txt"
CH_STRINGS = ["Bandpass405", "Bandpass488", "Bandpass561", "Bandpass640"]


reload(flad)


# In[107]:


experiment = Experiment.from_json("/home/brianl/mFISHrig2_rexp1/rexp1_acq4_data/Mouse_smFISH/Allen_Mouse_Panel_1/experiment.json")
#experiment = Experiment.from_json("/home/brianl/mFISHrig2_rexp1/rexp1_acq4_data/Mouse_smFISH/CZI_test_01/experiment.json")
#experiment = Experiment.from_json("/home/brianl/Desktop/temp/CZI_test_01/experiment.json")


# In[108]:


experiment


# In[109]:


# create the runnables for the pipeline, using in_place = False for the 
# first pipeline so the original data can still be looked at. This should be ok on a workstation
# since it should be reset for each fov

kwargs = dict(
    spot_diameter=5, # must be odd integer
    min_mass=0.02,
    max_size=2,  # this is max _radius_
    separation=7,
    noise_size=0.65,  # this is not used because preprocess is False
    preprocess=False,
    percentile=10,  # this is irrelevant when min_mass, spot_diameter, and max_size are set properly
    verbose=True,
    is_volume=True,
)
tlmpf = SpotFinder.TrackpyLocalMaxPeakFinder(**kwargs)
bandpass = Filter.Bandpass(lshort=.5, llong=7, threshold=0.0)
sigma=(1, 0, 0)  # filter only in z, do nothing in x, y
glp = Filter.GaussianLowPass(sigma=sigma, is_volume=True)
clip1 = Filter.Clip(p_min=50, p_max=100)
clip2 = Filter.Clip(p_min=99, p_max=100, is_volume=True)

def allen_pipeline(fov, codebook):
    primary_image = fov[FieldOfView.PRIMARY_IMAGES]
    new_image = clip1.run(primary_image, verbose=True, in_place=False, n_processes = 7)
    bandpass.run(new_image, verbose=True, in_place=True, n_processes = 7)
    glp.run(new_image, in_place=True, verbose=True, n_processes = 7)
    clip2.run(new_image, verbose=True, in_place=True, n_processes = 7)
    spot_attributes = tlmpf.run(new_image)
    decoded = codebook.decode_per_round_max(spot_attributes)
    return decoded[decoded["total_intensity"]>.025]


# process all the fields of view, not just one
def process_experiment_allen(experiment: Experiment):
    decoded_intensities = []
    for i, (name_, fov) in enumerate(experiment.items()):
        try:
            decoded = allen_pipeline(fov, experiment.codebook)
            decoded_intensities.append({"name":name_,"decoded": decoded})
        except:
            print("pipeline failed for fov "+name_)
            decoded_intensities.append({"name":name_,"decoded": None})
    return decoded_intensities



# In[112]:


# "pipeline"   currently very slow, but probably mostly io limited
print(time.asctime())
decoded_intensities = []
top_dir = "/home/brianl/mFISHrig2_rexp1/rexp1_acq4_data/Mouse_smFISH/CZI_test_output/"
for i, (name_, fov) in enumerate(experiment.items()):
    decoded = allen_pipeline(fov, experiment.codebook)
    decoded_intensities.append({"name":name_,"decoded": decoded})
    print("finished tile "+name_+" at "+str(time.asctime()))
    decoded.save(top_dir+ name_+".ncdf")


# ## plot below can show spots analyzed data from the whole experiment

# In[115]:


prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

plt.figure()
for ii,fov in enumerate(decoded_intensities):
    targetlist_index=0
    for targetname,targetdata in list(fov["decoded"].groupby("target")):
        if ii==10:
            plt.plot(targetdata["xc"], targetdata["yc"],'.', label = targetname, color = colors[targetlist_index])
        else:
            plt.plot(targetdata["xc"], targetdata["yc"],'.', label = '_nolegend_', color = colors[targetlist_index])
            
        targetlist_index = np.mod(targetlist_index + 1,len(colors))


#plt.plot(it002[it002["target"]=="Gad2"]["xc"], it002[it002["target"]=="Gad2"]["yc"],'o')
plt.legend()

plt.axis('equal')


# In[8]:


# this is supposed to be cool, but doesn't work here. should probably be tested first with a smaller dataset.
#  starfish.display.stack(experiment["fov_002"]["primary"], it002)
#


# # Cell below created the experiment used in this notebook.
# 
# actual data is 77 FOVs, the version saved here only uses 3

# In[35]:





output_dir = "/home/brianl/mFISHrig2_rexp1/rexp1_acq4_data/Mouse_smFISH/CZI_test_01/"
input_dir = "/home/brianl/mFISHrig2_rexp1/rexp1_acq4_data/Mouse_smFISH/CZI_AIBS-Boyden_Broad-Inhibitory_102318/"

write_experiment_json(
        output_dir,
        3,
        ImageFormat.TIFF,
        primary_image_dimensions={
            Axes.ROUND: 9,
            Axes.CH: 3,
            Axes.ZPLANE: 34,
        },
        aux_name_to_dimensions={
            'nuclei': {
                Axes.ROUND: 9,
                Axes.CH: 1,
                Axes.ZPLANE: 34,
            },

        },
        primary_tile_fetcher=flad.AllenPrimaryTileFetcher(input_dir),
        default_shape=SHAPE
    )


# In[147]:


# # once analyzed and saved to disk, you can reload the IntensityTable like this

# it000 = starfish.IntensityTable.load("/home/brianl/Desktop/temp/CZI_test_01/"+
#                               "fov_000"+"01_.ncdf")
# it001 = starfish.IntensityTable.load("/home/brianl/Desktop/temp/CZI_test_01/"+
#                               "fov_001"+"01_.ncdf")
# it002 = starfish.IntensityTable.load("/home/brianl/Desktop/temp/CZI_test_01/"+
#                               "fov_002"+"01_.ncdf")

# # and single FOVs

# f1=experiment["fov_001"]
# f2=experiment["fov_002"]
# f0=experiment["fov_000"]


# In[120]:


cf = plt.gcf()
    


# In[121]:


cf.savefig("starfish.png")

