import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle
from time import time
from PIL import Image
from pathlib import Path
from collections import defaultdict
from skimage.metrics import mean_squared_error
import glob
import os
import sys

from .png_preprocessing import PNG_Preprocessor
from .png_svg_converter import Raster2SVG_Converter
from animate_logos_main_adapted.src.preprocessing.svg_truncation import truncate_svg_path, truncate_svgs_folder


class PNGtoSVGConv:
    def __init__(self, inputImagePath, outputImagePath, quantatizePath, svgOutputPath, vtracerPath):
        self.inputImagePath = Path(inputImagePath)
        self.outputImagePath = Path(outputImagePath)
        self.quantatizePath = Path(quantatizePath)
        self.vtracerPath = Path(vtracerPath)
        self.svgOutputPath = Path(svgOutputPath)
        self.fileMode = self.inputImagePath.is_file()

        self.conversion()

    def conversion(self):
        if self.fileMode:
            print('-----------------------')
            print('Increase image resolution')
            new_resolution_image_path = PNG_Preprocessor.increase_image_resolution_path(
                input_image_path=str(self.inputImagePath),
                new_width=False,
                increase_ratio=4,
                write_to_file=True,
                output_folder=str(self.outputImagePath.parents[0]),
                output_filename=str(self.outputImagePath.stem)
            )
            print('Quantatize image')
            quantized_image = PNG_Preprocessor.quantize_image_from_path(input_image_path=str(self.outputImagePath),
                                                                        write_to_file=True,
                                                                        output_folder=str(
                                                                            self.quantatizePath.parents[0]),
                                                                        output_filename=str(
                                                                            self.quantatizePath.stem),
                                                                        error_threshold=0.005,
                                                                        show_intermediate_results=False)
            print('Convert to SVG')
            converter = Raster2SVG_Converter(str(self.vtracerPath.resolve()))
            converter.convert_raster2svg(
                input_image_path=str(self.quantatizePath),
                output_folder=str(self.svgOutputPath.parents[0]),
                output_filename=str(self.svgOutputPath.stem))
            truncated_svg_string = truncate_svg_path(str(self.svgOutputPath),
                                                     output_folder=str(
                                                         self.svgOutputPath.parents[0]),
                                                     output_filename=str(
                                                         self.svgOutputPath.stem),
                                                     threshold=0.99)
        else:
            print('Input needs to be file')
