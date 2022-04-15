# package
import os
import re
basePath = 'Data/SVGLogo'
listSVGFiles = os.listdir(basePath)
iteration = 0
for file in listSVGFiles:
    correctedFileName = re.sub(r'<|>|:|"|\/|\\|\||\?|\*', '', file)
    os.rename(f'{basePath}/{file}', f'{basePath}/{correctedFileName}')
    iteration += 1
    print(f'File {iteration} from {len(listSVGFiles)}')
