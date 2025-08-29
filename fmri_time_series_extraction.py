import os
import numpy as np
import pandas as pd
import nilearn.image as nimg
from nilearn import datasets
import logging


class BrainAtlas:
    def __init__(self, atlas_path=None, atlas_name=None):
        logging.info("Initializing brain_atlas class.")
        if atlas_path:
            self.atlas_img = nimg.load_img(atlas_path)
        elif atlas_name:
            self.atlas_img = self.fetch_atlas(atlas_name)
        else:
            raise ValueError("Either an atlas path or an atlas name must be provided.")
    
    def fetch(self, atlas_name):
        logging.info(f"Fetching atlas: {atlas_name}")
        atlas_fetchers = {
            "craddock": lambda: datasets.fetch_atlas_craddock_2012(
              data_dir=None, 
              url="http://cluster_roi.projects.nitrc.org/Parcellations/craddock_2011_parcellations.tar.gz", 
              resume=True, 
              verbose=1, 
              homogeneity=None, 
              grp_mean=True
            )
        }
        if atlas_name in atlas_fetchers:
            atlas = atlas_fetchers[atlas_name]()
            return nimg.index_img(atlas['tcorr_mean'], 42)
        else:
            raise ValueError(f"Atlas '{atlas_name}' not recognized.")

    def fetch_atlas(self, atlas_name):
        logging.info(f"Fetching atlas: {atlas_name}")
        if atlas_name == "craddock":
            datasets.fetch_atlas_craddock_2012(
                data_dir=None, 
                url="http://cluster_roi.projects.nitrc.org/Parcellations/craddock_2011_parcellations.tar.gz"
                resumme = True, 
                verbose = 1, 
                homogeneity = None, 
                grp_mean = True
            )
            