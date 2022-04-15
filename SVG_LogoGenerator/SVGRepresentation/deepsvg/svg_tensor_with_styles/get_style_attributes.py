


from get_style_attributes_utils import transform_to_hex, parse_svg

import pandas as pd
from xml.dom import minidom
import os
import time

pd.options.mode.chained_assignment = None  # default='warn'





def get_local_style_attributes_single_path(path):
    
    """ Generate dataframe containing local style attributes single SVG path.

    Args:
        path (str): Path of SVG.

    Returns:
        pd.DataFrame: Dataframe containing filename, , class, fill, stroke, stroke_width, opacity, stroke_opacity.

    """
    return pd.DataFrame.from_records(_get_local_style_attributes_single_path(path))
    
    
    
def _get_local_style_attributes_single_path(path):
    
    filename = os.path.splitext(path)[-2].split("/")[-1]
          
    try:
        _, attributes = parse_svg(path)
    except:
        print(f'{file}: Attributes not defined.')
    for i, attr in enumerate(attributes):

        fill = ''
        stroke = ''
        stroke_width = ''
        opacity = ''
        stroke_opacity = ''
        class_ = ''
        if 'style' in attr:
            a = attr['style']
            if a.find('fill') != -1:
                fill = a.split('fill:', 1)[-1].split(';', 1)[0]
            if a.find('stroke') != -1:
                stroke = a.split('stroke:', 1)[-1].split(';', 1)[0]
            if a.find('stroke-width') != -1:
                stroke_width = a.split('stroke-width:', 1)[-1].split(';', 1)[0]
            if a.find('opacity') != -1:
                opacity = a.split('opacity:', 1)[-1].split(';', 1)[0]
            if a.find('stroke-opacity') != -1:
                stroke_opacity = a.split('stroke-opacity:', 1)[-1].split(';', 1)[0]
        else:

            if 'fill' in attr:
                fill = attr['fill']

            if 'stroke' in attr:
                stroke = attr['stroke']


            if 'stroke-width' in attr:
                stroke_width = attr['stroke-width']




            if 'opacity' in attr:
                opacity = attr['opacity']




            if 'stroke-opacity' in attr:
                stroke_opacity = attr['stroke-opacity']




        if 'class' in attr:
            class_ = attr['class']

        # transform None and RGB to hex
        if '#' not in fill and fill != '':
            fill = transform_to_hex(fill)
        if '#' not in stroke and stroke != '':
            try:
                stroke = transform_to_hex(stroke)
            except:
                pass

#                 Cannot handle colors defined with linearGradient
#                 if 'url' in fill:
#                     fill = ''

        yield dict(filename=filename, class_=class_, fill=fill,
                   stroke=stroke, stroke_width=stroke_width, opacity=opacity, stroke_opacity=stroke_opacity)    
    





