# import libraries
import os
import logging
import cairosvg
# define variables for script
basePath = 'Data/SVGLogo'
targetPath = 'Data/PNGFolder'
listFiles = os.listdir(basePath)
logging.basicConfig(filename='FeatureCreation/Logging/pngConversion.log', filemode='a', format='%(name)s - %(levelname)s - %(message)s')
iteration = 0
# iterate over files in SVGLogo
for file in listFiles:
    splitFileName = file.split('.')[:-1]
    futureFileName = '.'.join(splitFileName)
    try:
        cairosvg.svg2png(url=f'{basePath}/{file}', write_to=f'{targetPath}/{futureFileName}.png')
    except:
        logging.error(f'{file} not transformable')
    iteration += 1
    print(f'{iteration} from {len(listFiles)}')
print('Successfull')
