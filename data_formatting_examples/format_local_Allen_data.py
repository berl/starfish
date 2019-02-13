import argparse
import io
import json
import os
import glob
import re
import zipfile
from typing import Mapping, Tuple, Union

import numpy as np
import requests
from skimage.io import imread
from slicedimage import ImageFormat

from starfish import Codebook
from starfish.experiment.builder import FetchedTile, TileFetcher
from starfish.experiment.builder import write_experiment_json
from starfish.types import Coordinates, Features, Indices, Number
from starfish.util.argparse import FsExistsType

SHAPE = 2048, 2048
PIXEL_SIZE = 1e-7
Z_SPACING = 3e-7
TC_STRING = "TileConfig_405.txt"
CH_STRINGS = ["Bandpass405", "Bandpass488", "Bandpass561", "Bandpass640"]

class AllenTile(FetchedTile):
    def __init__(self, file_path, tileconfig):
        self.file_path = file_path
        self.tileconfig  = tileconfig
    @property
    def shape(self) -> Tuple[int, ...]:
        return SHAPE

    @property
    def coordinates(self) -> Mapping[Union[str, Coordinates], Union[Number, Tuple[Number, Number]]]:
        return {
            Coordinates.X: (self.tileconfig["xPixels"]*PIXEL_SIZE, (self.tileconfig["xPixels"]+SHAPE[0])*PIXEL_SIZE),
            Coordinates.Y: (self.tileconfig["yPixels"]*PIXEL_SIZE, (self.tileconfig["yPixels"]+SHAPE[1])*PIXEL_SIZE),
            Coordinates.Z: (self.tileconfig["z"]*Z_SPACING, (self.tileconfig["z"]+1)*Z_SPACING)
        }

    @property
    def format(self) -> ImageFormat:
        return ImageFormat.NUMPY

    def tile_data(self) -> np.ndarray:
        return imread(self.file_path)


class AllenPrimaryTileFetcher(TileFetcher):
    def __init__(self, input_dir, get_DAPI_only = False):
        self.get_DAPI_only = get_DAPI_only
        self.round_folders=None
        self.tileconfig_list=None
        self.tiles=None
        self.channel_list=None
        self.z_list = None


        self.input_dir = input_dir
        self.discover_rounds()
        self.discover_channels()
        self.discover_tiles()
        self.discover_zs()

    def get_tile(self, fov: int, r: int, ch: int, z: int) -> FetchedTile:
        path_to_file = os.path.join(self.input_dir, self.round_folders[r], self.tiles[fov], 
                                        self.tiles[fov]+"_"+self.channel_list[ch]+"_"+self.z_list[z]+ ".tif")
        tileconfig=[tc for tc in self.tileconfig_list[r] if tc["file"][:13] in path_to_file][0]
        tileconfig.update({"z":z})
        return AllenTile(path_to_file, tileconfig)

    def discover_rounds(self):
        """ 
            identify rounds in this data set as separate directories

            round order is assumed to follow and increasing round index 
            with increasing folder number (as determined by `sorted`)
            also collect the tileConfig information for each round
        """
        print(self)
        self.round_folders = sorted( [ folder for folder in glob.glob(os.path.join(self.input_dir,"mFISH__*")) if (os.path.isdir(folder) and os.path.isdir(os.path.join(folder,"summary"))) ] )
        self.tileconfig_list =[]
        [self.tileconfig_list.append(read_tile_config(os.path.join(round_folder,TC_STRING))) for round_folder in self.round_folders]
        print(self.round_folders)

    def discover_channels(self):
        """
            identify the different channels, producing a list
            in this case, these are hardcoded as constants
        """
        if self.get_DAPI_only:
            self.channel_list = [CH_STRINGS[0]]

        else:
            self.channel_list = CH_STRINGS[1:]

    def discover_tiles(self):
        #get tile names from round 0 
        self.tiles=sorted([os.path.basename(tile) for tile in glob.glob(os.path.join(self.round_folders[0],"*")) if (os.path.isdir(tile) and "summary" not in tile)])
    

    def discover_zs(self):
        """
            use the 405 channel name to identify the files for different z positions
        """
        round0tile0string = os.path.join(self.round_folders[0], self.tiles[0])

        z_files = sorted(glob.glob(os.path.join(round0tile0string,"*"+CH_STRINGS[0]+"*.tif")))
        self.z_list = sorted([z_file[-7:-4] for z_file in z_files])
 

class AllenAuxTileFetcher(TileFetcher):
    def __init__(self, path):
        self.path = path

    def get_tile(self, fov: int, r: int, ch: int, z: int) -> FetchedTile:
        
        return AllenTile(self.path)


def download(input_dir, url):
    print("Not Implemented.  These methods access local data only ...")



def write_json(res, output_path):
    json_doc = json.dumps(res, indent=4)
    print(json_doc)
    print("Writing to: {}".format(output_path))
    with open(output_path, "w") as outfile:
        json.dump(res, outfile, indent=4)


def format_data(input_dir, output_dir):
    if not input_dir.endswith("/"):
        input_dir += "/"

    if not output_dir.endswith("/"):
        output_dir += "/"

    def add_codebook(experiment_json_doc):
        experiment_json_doc['codebook'] = "codebook.json"

        return experiment_json_doc

    write_experiment_json(
        output_dir,
        1,
        {
            Indices.ROUND: 2,
            Indices.CH: 4,
            Indices.Z: 34,
        },
        {
            'nuclei': {
                Indices.ROUND: 1,
                Indices.CH: 1,
                Indices.Z: 1,
            },
            'dots': {
                Indices.ROUND: 1,
                Indices.CH: 1,
                Indices.Z: 1,
            }
        },
        primary_tile_fetcher=AllenPrimaryTileFetcher(input_dir),
        aux_tile_fetcher={
            'nuclei': AllenAuxTileFetcher(os.path.join(input_dir, "DO", "c1.TIF")),
            'dots': AllenAuxTileFetcher(os.path.join(input_dir, "DO", "c2.TIF")),
        },
        postprocess_func=add_codebook,
        default_shape=SHAPE
    )

    codebook_array = [
        {
            Features.CODEWORD: [
                {Indices.ROUND.value: 0, Indices.CH.value: 3, Features.CODE_VALUE: 1},
                {Indices.ROUND.value: 1, Indices.CH.value: 3, Features.CODE_VALUE: 1},
                {Indices.ROUND.value: 2, Indices.CH.value: 1, Features.CODE_VALUE: 1},
                {Indices.ROUND.value: 3, Indices.CH.value: 2, Features.CODE_VALUE: 1}
            ],
            Features.TARGET: "ACTB_human"
        },
        {
            Features.CODEWORD: [
                {Indices.ROUND.value: 0, Indices.CH.value: 3, Features.CODE_VALUE: 1},
                {Indices.ROUND.value: 1, Indices.CH.value: 1, Features.CODE_VALUE: 1},
                {Indices.ROUND.value: 2, Indices.CH.value: 1, Features.CODE_VALUE: 1},
                {Indices.ROUND.value: 3, Indices.CH.value: 2, Features.CODE_VALUE: 1}
            ],
            Features.TARGET: "ACTB_mouse"
        },
    ]
    codebook = Codebook.from_code_array(codebook_array)
    codebook_json_filename = "codebook.json"
    codebook.to_json(os.path.join(output_dir, codebook_json_filename))



def read_tile_config(tileconfigpath):
    """
    read in tile configuration  from a tileconfig file

    tileconfig files are saved by FIJI grid-collection stitching (among other sources), 
    tileconfig contains 2d grid coordinate indices ('xInteger' and 'yInteger' here) and 
    float coordinates for the upper left corner of each tile
     
    Args:
        tileconfigpath: location of a text file with rows delimited by ';'
    Returns:
        tileList: list of tileconfig dictionaries (i.e. elements of the list are rows of the tileconfig file)
    """
    delim=';'
    a = np.genfromtxt(tileconfigpath, delimiter=delim, skip_header=4, usecols=(0,2), dtype ='str')
    tileList = []
    for i,f in enumerate(a):
        fileStringi = f[0]
        coordinateStrings = re.split(r'[(|,|)]',f[1])
        ni =[ float(coordinateStrings[ii]) for ii in [1,2]]
        tileI = {'file':fileStringi, 
                 'xInteger':int(float(fileStringi.split('_')[1])),
                'yInteger':int(float(fileStringi.split('_')[2])),
                'xPixels':ni[0],
                'yPixels':ni[1]}
        tileList.append(tileI)
    return tileList


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=FsExistsType())
    parser.add_argument("output_dir", type=FsExistsType())
    parser.add_argument("--d", help="Download data", type=bool)

    args = parser.parse_args()

    format_data(args.input_dir, args.output_dir, args.d)
