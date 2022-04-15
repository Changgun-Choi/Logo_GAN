# import libraries
import logging
import cairosvg
from pathlib import Path
from tqdm import tqdm

def transformPng(inputPath, outputPath, logPath):
    # define variables for script
    basePath = Path(inputPath)
    targetPath = Path(outputPath)
    logPath = Path(logPath)
    targetPath.mkdir(exist_ok=True)
    listFiles = basePath.glob('*.svg')
    logging.basicConfig(filename=str(logPath),
                        filemode='a', format='%(name)s - %(levelname)s - %(message)s')
    listFiles = list(listFiles)
    # iterate over files in SVGLogo
    for file in tqdm(listFiles, total=len(listFiles)):
        fileName = str(file)
        try:
            cairosvg.svg2png(url=fileName,
                             write_to=f'{str(targetPath)}/{fileName.split("/")[-1].split(".")[0]}.png')
        except:
            logging.error(f'{file} not transformable')

    print('Successfull')
