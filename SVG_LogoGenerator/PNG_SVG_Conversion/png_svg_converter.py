import subprocess
import os


        
class Raster2SVG_Converter ():

        
        
    """
    'Vtracer' library is used to convert PNGs to SVGs.

    'Vtracer' is a command line app and usage pattern is:
    

    YOUR/LOCAL/PATH/.cargo/bin/vtracer --input YOUR/LOCAL/PATH/input.png --output LOCAL/OUTCOME/PATH/output.svg #
    
    
    more about Vtracer and usage available on github https://github.com/visioncortex/vtracer
    
    
    Parameters
    ----------
    vtracer_path : string
        Your local to "Vtracer" library. See Readme.md in PNG_SVG_Conversion folder for instructions.    
    
    
    Attributes
    ----------
    
    
    
    vtracer_path : string
        Your local to "Vtracer" library. See Readme.md in PNG_SVG_Conversion folder for instructions.
    
    command : list
        List of command line arguments to run using Python Subprocess module



    """    
    
    
    
    
    def __init__(self, vtracer_path):
        self.vtracer_path = vtracer_path
        self.command = [self.vtracer_path, '--input',  '--output' ] 
        

        

    def convert_raster2svg (  self,
                              input_image_path,
                              output_folder = False,
                              output_filename = False):

        
        
        """
        Reads png file from LOCAL path and writes to LOCAL svg file 

        ----------
        input_image_path : string
        
            LOCAL input image path of a png file
            
        output_folder: string 
        
            If False, writes to the input folder. Otherwise, writes to the specified output_folder.
            
        output_filename: string
        
            If False, writes file as "{input_filename}.svg". Otherwise, writes filename as specified output_filename
        

            

        Returns
        -------
        Nothing
        
       
        """        
        
        
        
        
        
      
        
        
        input_extension = "." + input_image_path.split("/")[-1].split(".")[-1]
        input_filename = input_image_path.split("/")[-1].split(".")[-2]
        input_folder = input_image_path[:-len(input_image_path.split("/")[-1])]
        
        
        
        file_command=[]
        
        ## in this order 
        
        file_command.append(self.command[0])
        file_command.append(self.command[1])
        file_command.insert(2, input_image_path)
        file_command.append(self.command[2])
        


        if output_folder:

            if output_filename:

                file_command.insert(4, output_folder + "/" + output_filename + ".svg" )
                
            else:
                
                file_command.insert(4, output_folder + "/" + input_filename + ".svg" )

        else:
            
            if output_filename:
                
                file_command.insert(4, input_folder + "/" + output_filename + ".svg" )
                
            else:
                
                file_command.insert(4, input_folder + "/" + output_filename + ".svg" )

        print(file_command)
        subprocess.run(file_command, capture_output = True)


        
        del file_command

    def convert_raster2svg_folder(  self,
                                    input_images_folder,
                                    output_folder
                                 ):
    
        """ The same as convert_raster2svg() but for all files in folder"""
        
        for filename in os.listdir(input_images_folder):
                    
            self.convert_raster2svg ( 
                                        input_image_path = input_images_folder + "/" + filename,
                                        output_folder = output_folder,
                                        output_filename = False
                                    )
                                     

        
    