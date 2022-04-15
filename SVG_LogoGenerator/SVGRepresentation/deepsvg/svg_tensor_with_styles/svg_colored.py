from __future__ import annotations
from deepsvg.svglib.geom import *
from xml.dom import expatbuilder
import torch
from typing import List, Union
import IPython.display as ipd
import cairosvg
from PIL import Image
import io
import os
from moviepy.editor import ImageClip, concatenate_videoclips, ipython_display
import math
import random
import networkx as nx
import re

Num = Union[int, float]

from deepsvg.svglib.svg import SVG
from get_style_attributes import get_local_style_attributes_single_path
from deepsvg.svglib.svg_command import SVGCommandBezier
from deepsvg.svglib.svg_path import SVGPath, Filling, Orientation
from deepsvg.svglib.svg_primitive import SVGPathGroup, SVGRectangle, SVGCircle, SVGEllipse, SVGLine, SVGPolyline, SVGPolygon
from deepsvg.svglib.geom import union_bbox


class SVG_colored(SVG):
    
    
    """
    SVG_colored object is advance version of SVG object from DeepSVG. SVG_colored has 2 major advantages : 1) it encodes stylistic features of a svg file to tensor. 2) encoded svg tensor is structured as a sequence of path tensors instead of a single whole tensor as in original SVG object from DeepSVG.    
    
    
    Parameters
    ----------
    It takes no parameters while initiated.
    
    
    Attributes
    ----------
    
    
    
    svg_top_marker : string
        String needed to reconstruct SVG file from tensor. Placed at the top of SVG file.
        
        
    svg_bottom_marker : string
        String needed to reconstruct SVG file from tensor. Placed at the bottom of SVG file.
        
    svg_path : string
        File path of a svg file
        
    SVG_object : SVG Object
        
        SVG object from DeepSVG
    
    styles_df : pandas.DataFrame Object
        
        A dataframe contains style information for each path of a parsed svg file
        
    path_separator_tensor : torch.Tensor
        
        A tensor used to separate path tensors of svg. It has shape (1,17) and all values are 333.33.
        17 because : 14 (dimension from DeepSVG tensor representation) + 3 (stylistic features, specifically 3 columns of "fill" RGB values)


    """
    
    
    def __init__ (self):
        
        self.svg_top_marker = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0.0 0.0 500.0 500.0" height="200px" width="200px">'
        self.svg_bottom_marker = '</svg>'
        self.svg_path = None
        self.SVG_object = None
        self.styles_df = None
        self.path_separator_tensor = torch.Tensor( [333.33] * (17)).unsqueeze(0)    
                                                    
                                                    # separates paths. 17 = 14(deepsvg) + 3 style columns
                
                                                    
        

    def load_svg_colored(self, svg_path):
        
        
        
        
        """
        Load SVG file from path and set self.attributes
        ----------
        svg_path : string
            Path of svg file

        Returns
        -------
        None
        
       
        """
        
        
        setattr(self, "svg_path", svg_path)
        setattr(self, "SVG_object", SVG.load_svg(svg_path))
        setattr(self, "styles_df", get_local_style_attributes_single_path(svg_path))

        


    
    @staticmethod
    def rgb2hex(r,g,b):
        
        """ Transforms RGB values to hexidecimal color code """
        
        
        
        return "#{:02x}{:02x}{:02x}".format(r,g,b)
        
        
        
    @staticmethod
    def from_styles_df_to_tensor (df):
        
        """ Input is a DataFrame containing  filename, class, fill, stroke, stroke_width, opacity, stroke_opacity for each path. So far only 'fill' values are encoded to tensor. """
    
        tensor_list = []


        for index, row in df.iterrows():
            if row["fill"]:
                fill_color_hex = row['fill'].lstrip('#')
                tensor_list.append(list(float(int(fill_color_hex[i:i+2], 16)) for i in (0, 2, 4)))
            else:
                fill_color_hex = "#000000"
                tensor_list.append(list(float(int(fill_color_hex[i:i+2], 16)) for i in (0, 2, 4)))

        return torch.Tensor(tensor_list)
    
    
        
    def to_tensor_full(self):
        
        """ Encodes SVG_colored object to stylistic tensor represenation """ 

        
        outline_tensor = self.SVG_object.to_tensor()
        style_tensor = self.from_styles_df_to_tensor(self.styles_df)
        
        list_of_tensors = []
        for i,j in enumerate(self.SVG_object):
            path_style_tensor = style_tensor[i,:].repeat(j.to_tensor().shape[0],1)
            path_outline_tensor = j.to_tensor()
            


            cat_style_and_outline = torch.cat((path_outline_tensor, path_style_tensor),1)
            cat_style_and_outline_and_separator = torch.cat((cat_style_and_outline,self.path_separator_tensor),0)
            
            list_of_tensors.append(cat_style_and_outline_and_separator)
        
        outcome_tensor = torch.cat(list_of_tensors)

        
        return outcome_tensor
    
    
    def from_tensor_to_svg_string(self, tensor):
        
        """ Decodes stylistic tensor representation back to SVG string """
        
        svg_str = ""
        sep_indices = (tensor == self.path_separator_tensor).nonzero(as_tuple=True)[0]
        sep_indices_list = sorted(list(set(sep_indices.tolist())))
        sep_indices_list.insert(0,0)
        svg_str = svg_str + self.svg_top_marker
        
        
        
        for index, value in enumerate(sep_indices_list):

            if index == 0:
                begin_index = 0
                end_index = sep_indices_list[index+1]
            else:
                try:
                    begin_index = sep_indices_list[index] + 1
                    end_index = sep_indices_list[index + 1]
                except:
                    del begin_index
                    del end_index
            try:

                path_tensor = tensor[begin_index:end_index,:]

                path_style_tensor = path_tensor[:,14:]

                fill_tensor = torch.mean(path_style_tensor, dim = 0)

                
                
                fill_color = self.rgb2hex(int(fill_tensor[0]), int(fill_tensor[1]), int(fill_tensor[2]))

                path_string = SVGPath.from_tensor(path_tensor[:,:14]).to_str()
  
                
                
                style_string = f'fill="{fill_color}"'

                
                path_string_with_style = re.sub('fill="none"',
                                                style_string,
                                                path_string)

                svg_str = svg_str + path_string_with_style
 
            except: 
                pass 

        svg_str = svg_str + self.svg_bottom_marker
        
        return svg_str
    

    