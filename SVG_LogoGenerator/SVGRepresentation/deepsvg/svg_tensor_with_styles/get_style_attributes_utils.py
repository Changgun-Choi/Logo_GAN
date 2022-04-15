from svgpathtools import svg2paths
import pandas as pd
import numpy as np
from xml.dom import minidom

pd.options.mode.chained_assignment = None  # default='warn'




def parse_svg(file):
    """ Parse a SVG file.

    Args:
        file (str): Path of SVG file.

    Returns:
        list, list: List of path objects, list of dictionaries containing the attributes of each path.

    """
    paths, attrs = svg2paths(file)
    return paths, attrs






def _combine_columns(df, col_name):
    col = np.where(~df[f"{col_name}_y"].astype(str).isin(["", "nan"]),
                   df[f"{col_name}_y"], df[f"{col_name}_x"])
    return col


def transform_to_hex(rgb):
    """ Transform RGB to hex.

    Args:
        rgb (str): RGB code.

    Returns:
        str: Hex code.

    """
    if rgb == 'none':
        return '#000000'
    if 'rgb' in rgb:
        rgb = rgb.replace('rgb(', '').replace(')', '')
        if '%' in rgb:
            rgb = rgb.replace('%', '')
            rgb_list = rgb.split(',')
            r_value, g_value, b_value = [int(float(i) / 100 * 255) for i in rgb_list]
        else:
            rgb_list = rgb.split(',')
            r_value, g_value, b_value = [int(float(i)) for i in rgb_list]
        return '#%02x%02x%02x' % (r_value, g_value, b_value)
