import pickle
import os
from os import listdir
from os.path import isfile, join
from xml.dom import minidom
from pathlib import Path
from matplotlib import image
from datetime import datetime
from shutil import copyfile
from skimage.metrics import mean_squared_error
from animate_logos_main_adapted.src.utils import logger
from animate_logos_main_adapted.src.data.svg_to_png import convert_svgs_in_folder




    
    
# auxiliary function. It is used inside of a class already

def _get_number_of_paths_for_truncation(ordered_relevance_scores_list, coverage_percent):
    """
    args:
    ordered_relevance_scores_list - list of mse values
    coverage_percent - is a float mse threshold
    d between 0 and 1(e.g: 0.60, 0.999) 
    
    return:
    a number of paths to keep """
    
    sum_list = sum(ordered_relevance_scores_list)
    elements_sum = 0
    
    for i,number in enumerate(ordered_relevance_scores_list):
        elements_sum = elements_sum + number
        if elements_sum > coverage_percent*sum_list:
            print(f"number of path to truncate: {i + 1} out of {len(ordered_relevance_scores_list)}")
        
            return i + 1
        
        
        
        
        
        
        
        
class Selector_singe_Logo ():

    """ Selector class for path relevance ordering. """

    def __init__(self,  logo_path,
                 dir_path_selection='./PNG_SVG_Conversion/path_selection',
                 dir_truncated_svgs='./PNG_SVG_Conversion/truncated_svgs',
                 dir_selected_paths='./PNG_SVG_Conversion/selected_paths',
                 dir_decomposed_svgs='./PNG_SVG_Conversion/decomposed_svgs',
                 threshold = 0.5,
                 output_filename = False):
        """
        Args:
            dir_svgs (str): Directory containing SVGs to be sorted.
            dir_path_selection (str): Directory of logo folders containing PNGs of deleted paths.
            dir_truncated_svgs (str): Directory containing truncated SVGs to most relevant paths.
            dir_selected_paths (str): Directory containing decomposed SVGs selected by relevance ordering.
            dir_decomposed_svgs (str): Directory containing decomposed SVGs of all paths.

        """
        self.dir_path_selection = dir_path_selection
        self.dir_truncated_svgs = dir_truncated_svgs
        self.dir_selected_paths = dir_selected_paths
        self.dir_decomposed_svgs = dir_decomposed_svgs
        self.threshold = threshold
        self.logo_path = logo_path
        self.logo_filename = os.path.split(logo_path)[-1][:-4]
        if output_filename:
            self.logo_filename = output_filename[:-4]
        
        
        
        
        
        

    @staticmethod
    def get_elements(doc):
        """ Retrieve all animation relevant elements from SVG.

        Args:
            doc (xml.dom.minidom.Document): XML minidom document from which to retrieve elements.

        Returns:
            list (xml.dom.minidom.Element): List of all elements in document

        """
        
        return doc.getElementsByTagName('path') + doc.getElementsByTagName('circle') + doc.getElementsByTagName(
            'ellipse') + doc.getElementsByTagName('line') + doc.getElementsByTagName(
            'polygon') + doc.getElementsByTagName('polyline') + doc.getElementsByTagName(
            'rect') + doc.getElementsByTagName('text')
    
    
    
    

    def delete_paths(self):
        """ Function to iteratively delete single paths in an SVG and save remaining logo as PNG
        to Selector.dir_path_selection. Requires directory Selector.dir_decomposed_svgs.

        Args:
            logo (str): Name of logo (without file type ending).

        """
        logo = self.logo_filename
        Path(f'{self.dir_path_selection}/{logo}').mkdir(parents=True, exist_ok=True)
        
        doc = minidom.parse(self.logo_path)
        nb_original_elements = len(self.get_elements(doc))
        with open(f'{self.dir_path_selection}/{logo}/original.svg', 'wb') as file:
            file.write(doc.toprettyxml(encoding='iso-8859-1'))
        doc.unlink()
        for i in range(nb_original_elements):
            doc = minidom.parse(f'{self.dir_path_selection}/{logo}/original.svg')
            elements = self.get_elements(doc)
            path = elements[i]
            parent = path.parentNode
            parent.removeChild(path)
            with open(f'{self.dir_path_selection}/{logo}/without_id_{i}.svg', 'wb') as file:
                file.write(doc.toprettyxml(encoding='iso-8859-1'))
            doc.unlink()
        convert_svgs_in_folder(f'{self.dir_path_selection}/{logo}')
        
        
    

        
        
        
        
        
        
        

    def delete_paths_in_logos(self, logos):
        """ Iterate over list of logos to apply deletion of paths.

        Args:
            logos (list (str)): List of logos (without file type ending).

        """
        start = datetime.now()
        n_logos = len(logos)
        for i, logo in enumerate(logos):
            if i % 20 == 0:
                logger.info(f'Current logo {i+1}/{n_logos}: {logo}')
            self.delete_paths(logo)
        logger.info(f'Time: {datetime.now() - start}')

        
        
    @staticmethod
    def sort_by_relevance(path_selection_folder, excluded_paths, coverage_percent, nr_paths_trunc = 20) :
        """ Sort paths in an SVG by relevance. Relevance of the path is measured by the MSE between the
        original logo and the logo resulting when deleting the path.
        The higher the MSE, the more relevant the given path.

        Args:
            path_selection_folder (str): Path to folder containing PNGs of the original logo and of the resulting logos
            when deleting each path.
            excluded_paths (list (int)): List of animation IDs that should not be considered as relevant. These paths
            will be assigned a relevance score of -1.
            nr_paths_trunc (int): Number of paths that should be kept as the most relevant ones.

        Returns:
            list (int), list(int), list (int), list (int): List of animation IDs sorted by relevance (descending),
            sorted list of MSE scores (descending), list of MSE scores of paths that were missed, list of animation IDs
            of paths that were misses due to exclusion.

        """
        nr_paths = len([name for name in os.listdir(path_selection_folder)
                        if os.path.isfile(os.path.join(path_selection_folder, name))]) - 1
        relevance_scores = []
        missed_scores, missed_paths = [], []
        img_origin = image.imread(os.path.join(path_selection_folder, "original.png"))
        logo = path_selection_folder.split('/')[-1]
        counter = 0
        for i in range(nr_paths):
            img_reduced = image.imread(os.path.join(path_selection_folder, "without_id_{}.png".format(i)))
            try:
                decomposed_id = f'{logo}_{i}'
                if decomposed_id in excluded_paths:
                    missed_mse = mean_squared_error(img_origin, img_reduced)
                    missed_scores.append(missed_mse)
                    missed_paths.append(decomposed_id)
                    logger.warning(f'No embedding for path {decomposed_id}, actual MSE would be: {missed_mse}')
                    mse = 0
                else:
                    try:
                        mse = mean_squared_error(img_origin, img_reduced)
                        
                    except:
                        try:
                            mse = mean_squared_error(img_origin, img_reduced[:,:,:3])  
                        except ValueError as e:
                            logger.warning(f'Could not calculate MSE for path {logo}_{i} '
                               f'- Error message: {e}')
                            
            except ValueError as e:
                logger.warning(f'Could not calculate MSE for path {logo}_{i} '
                               f'- Error message: {e}')
                counter += 1
                mse = 0
            relevance_scores.append(mse)
            
        
        relevance_score_ordering = list(range(nr_paths))
        relevance_score_ordering.sort(key=lambda x: relevance_scores[x], reverse=True)
        
        if coverage_percent < 0:
            relevance_score_ordering = relevance_score_ordering[0:nr_paths_trunc]
        else:
            
            # each path id is key and respective mse error is value
            path_mse_dict = dict()
            for path_id , mse in enumerate(relevance_scores):
                path_mse_dict[path_id] = mse
                
            # sort keys by values in descending order of mse error (path importance)
            path_mse_dict_sorted = {k: v for k, v in sorted(path_mse_dict.items(), key=lambda item: item[1],reverse= True)}

            # n - is a number of paths to keep according to sum mse covered parameter
            n = _get_number_of_paths_for_truncation(path_mse_dict_sorted.values(),coverage_percent)

            # take top n keys from sorted dictionary as relevant
            relevance_score_ordering = list(path_mse_dict_sorted.keys())[:n]


        missed_relevant_scores, missed_relevant_paths = list(), list()
        for i in range(len(missed_scores)):
            score = missed_scores[i]
            if score >= relevance_scores[relevance_score_ordering[-1]]:
                missed_relevant_scores.append(score)
                missed_relevant_paths.append(missed_paths[i])
        if len(missed_relevant_scores) > 0:
            logger.warning(f'Number of missed relevant paths due to embedding: {len(missed_relevant_scores)}')
        if counter > 0:
            logger.warning(f'Could not calculate MSE for {counter}/{nr_paths} paths')
        relevance_score_ordering = [id_ for id_ in relevance_score_ordering if relevance_scores[id_] != -1]
        return relevance_score_ordering, relevance_scores, missed_relevant_scores, missed_relevant_paths, n,len(relevance_scores),coverage_percent 

    
    
    
    
    
    
    def select_paths(self, excluded_paths):
        """ Iterate over a directory of SVG files and select relevant paths. Selected paths and original
        SVGs will be saved to Selector.dir_selected_paths/logo. Requires directory Selector.dir_path_selection.

        Args:
            svgs_folder (str): Directory containing SVG files from which to select relevant paths.
            excluded_paths (list (int)): List of animation IDs that should not be considered as relevant. These paths
            will be assigned a relevance score of -1.

        Returns:
            list (int): List of missed paths.

        """
        Path(self.dir_selected_paths).mkdir(parents=True, exist_ok=True)
        logo = self.logo_filename
        start = datetime.now()
        missed_scores, missed_paths = list(), list()

        sorted_ids, sorted_mses, missed_relevant_scores, missed_relevant_paths = \
            self.sort_by_relevance(f'{self.dir_path_selection}/{logo}', excluded_paths)
        missed_scores.append(len(missed_relevant_scores))
        missed_paths.extend(missed_relevant_paths)
        copyfile(f'{svgs_folder}/{logo}.svg', f'{self.dir_selected_paths}/{logo}_path_full.svg')
        for j, id_ in enumerate(sorted_ids):
            copyfile(f'{self.dir_decomposed_svgs}/{logo}_{id_}.svg',
                     f'{self.dir_selected_paths}/{logo}_path_{j}.svg')
        logger.info(f'Total number of missed paths: {sum(missed_scores)}')
        logger.info(f'Time: {datetime.now() - start}')
        return missed_paths
    
    
    
    

    def truncate_svgs(self, svgs_folder, logos=None, excluded_paths=list(), nr_paths_trunc= 20):
        """ Truncate SVGs to most relevant paths and save them to Selector.dir_truncated_svgs. Requires directory
        Selector.dir_path_selection.

        Args:
            svgs_folder (str): Directory containing SVG files from which to select relevant paths.
            logos (list): List of logos to be truncated.
            excluded_paths (list (int)): List of animation IDs that should not be considered as relevant. These paths
            will be assigned a relevance score of -1.
            nr_paths_trunc (int): Number of paths that should be kept as the most relevant ones.

        """
        number_of_total_paths = 0
        number_of_kept_paths = 0
        Path(self.dir_truncated_svgs).mkdir(parents=True, exist_ok=True)
        start = datetime.now()
        logos = [f[:-4] for f in listdir(svgs_folder) if isfile(join(svgs_folder, f))] if logos is None else logos
        for i, logo in enumerate(logos):
            print(logo)
            if i % 20 == 0:
                logger.info(f'Current logo {i}/{len(logos)}: {logo}')
            sorted_ids, _, _, _ , kept_paths, total_paths, coverage_percent  = self.sort_by_relevance(f'{self.dir_path_selection}/{logo}',
                                                         excluded_paths, self.threshold, nr_paths_trunc)
            


            try:
                number_of_kept_paths = number_of_kept_paths + kept_paths
                number_of_total_paths = number_of_total_paths + total_paths
            except:
                pass
            doc = minidom.parse(f'{svgs_folder}/{logo}.svg')
            original_elements = self.get_elements(doc)
            nb_original_elements = len(original_elements)
            for j in range(nb_original_elements):
                if j not in sorted_ids:
                    path = original_elements[j]
                    parent = path.parentNode
                    parent.removeChild(path)
                    
                with open(f'{self.dir_truncated_svgs}/{logo}_truncated.svg', 'wb') as file:
                    file.write(doc.toprettyxml(encoding='iso-8859-1'))

            doc.unlink()
            


        logger.info(f'Time: {datetime.now() - start}')
        print(f"Kept {number_of_kept_paths} out of total {number_of_total_paths} to cover {coverage_percent*100} percent of MSE ")

        
        
        
        
    def truncate_svgs_output_string(self, logos=None, excluded_paths=list(), nr_paths_trunc= 20):
        """ Truncate SVG to most relevant paths and save. Requires directory Selector.dir_path_selection.

        Args:
            excluded_paths (list (int)): List of animation IDs that should not be considered as relevant. These paths
            will be assigned a relevance score of -1.
            nr_paths_trunc (int): Number of paths that should be kept as the most relevant ones.

        """
        number_of_total_paths = 0
        number_of_kept_paths = 0
        Path(self.dir_truncated_svgs).mkdir(parents=True, exist_ok=True)
        start = datetime.now()
        logo = self.logo_filename
        print(logo)


        sorted_ids, _, _, _ , kept_paths, total_paths, coverage_percent  = self.sort_by_relevance(f'{self.dir_path_selection}/{logo}',
                                                     excluded_paths, self.threshold, nr_paths_trunc)
        try:
            number_of_kept_paths = number_of_kept_paths + kept_paths
            number_of_total_paths = number_of_total_paths + total_paths
        except:
            pass
        doc = minidom.parse(self.logo_path)
        original_elements = self.get_elements(doc)
        nb_original_elements = len(original_elements)
        for j in range(nb_original_elements):
            if j not in sorted_ids:
                path = original_elements[j]
                parent = path.parentNode
                parent.removeChild(path)
            with open(f'{self.dir_truncated_svgs}/{logo}_truncated.svg', 'wb') as file:
                file.write(doc.toprettyxml(encoding='iso-8859-1'))
        s = doc.toxml()



        return s
        print("-"*1000)
        doc.unlink()



        logger.info(f'Time: {datetime.now() - start}')
        print(f"Kept {number_of_kept_paths} out of total {number_of_total_paths} to cover {coverage_percent*100} percent of MSE ")


        
        
        
        
        
def truncate_svg_path (logo_path,
                        threshold = 0.5,
                        output_folder = False,
                        output_filename = False):

    """
    Truncates minor paths and keeps the largest pathes according to MSE error
    ----------
    logo_path : string

        Input image path of a SVG file

    output_folder: string 
        If False, writes to ./SVG_LogoGenerator/PNG_SVG_Conversion/truncated_svgs. Otherwise, writes to the specified output_folder

    output_filename: string
        If False, writes file as "{input_filename}_truncated.svg". Otherwise, writes filename as specified output_filename

    error_threshold : float between 0 and 1
        1 - MSE error. If  error_threshold = 0.99, it means, that the MSE error between the original image and the image 
with reduced colors is less than 1%


    Returns
    -------
    Truncated SVG string


    """    
    
    
    if output_folder:
        if output_filename:
            sel = Selector_singe_Logo (
                                            logo_path,
                                            dir_truncated_svgs = output_folder,
                                            output_filename = output_filename,
                                            threshold = threshold)
        else:
           
            sel = Selector_singe_Logo (
                                            logo_path,
                                            dir_truncated_svgs = output_folder,
                                            output_filename = False,
                                            threshold = threshold) 
    else:
        if output_filename:
            
            sel = Selector_singe_Logo (
                                            logo_path,
                                            output_filename = output_filename,
                                            threshold = threshold)             
        else:
            
            sel = Selector_singe_Logo (logo_path, threshold = threshold)
            


        


    sel.delete_paths()

    output_string = sel.truncate_svgs_output_string()
    return output_string


def truncate_svgs_folder (
                            input_folder,
                            output_folder,
                            threshold = 0.5
    
                          ):
    """ Applies truncate_svg_path() for all files in given folder. Returns nothing
    
    """
    
    for filename in os.listdir(input_folder):
        try:
            outcome_svg_string =  truncate_svg_path(
                                                        logo_path = input_folder + "/" + filename,
                                                        threshold = threshold,
                                                        output_folder = output_folder,
                                                     ) 
        except:
            
            print(f"Could not process {filename}")



