"""
The SVG Web Scraper script downloads and stores SVG logos based on an input dataframe.
A company name is used to search on google together with the words 'logo' and 'svg'.
For that it uses the selenium package to simulate a browser to overcome blocks from
the google servers.
As a source for the Scraping only 'wikipedia' and 'wikimedia' domains are accepted
and used for the SVG File storage.
To check whether the logo is really a SVG file, the last URL (logoImage) before the
download is validated with a regex command.
###################################################################################
imported 3rd-party Packages: Pandas (Version 1.3.0), selenium (Version 4.0.0)

Input: Kaggle dataset with company names
(https://www.kaggle.com/peopledatalabssf/free-7-million-company-dataset).
Stored in a folder called 'Data' with the original file name 'companies_sorted.csv'

Output: Store an SVG File with the found company logos in the folder SVGLogo
"""
# load necessary libraries
import os
import pandas as pd
import re
import urllib
from pathlib import Path
from urllib.request import urlretrieve
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options


class wikipediaScraper:
    
    def __init__(self, datasetPath, webdriverPath, logPath, destPath):
        self.dataPath = Path(datasetPath)
        self.webdriverPath = Path(webdriverPath)
        self.logPath = Path(logPath)
        self.destPath = destPath

    def scraper(self):

        # Load data frame of company
        companyData = pd.read_csv(self.dataPath)
        companyNames = companyData['name']


        # iterate over name of
        for companyName in companyNames:
            print(companyName)
            searchLink = f'https://www.google.com/search?q={urllib.parse.quote(companyName)}+logo+svg&source=lnms&tbm=isch&sa=X&ved=2ahUKEwjP6Of-7snzAhUQhP0HHR-oB7gQ_AUoAXoECAEQAw&biw=1200&bih=813'
            # set up selenium web driver with a headless browser
            chromeOptions = Options()
            chromeOptions.add_argument("--headless")
            driver = webdriver.Chrome(str(self.webdriverPath), options=chromeOptions)
            driver.get(searchLink)
            # extract all google search links containing wikipedia and wikimedia
            imageResults = driver.find_elements(By.CLASS_NAME, 'fxgdke')
            imageLinks = [result for result in imageResults if re.search(r'\bwikipedia\b|\bwikimedia\b', result.text)]
            # check if list is empty or not. if empty, write into error log file
            if imageLinks:
                imagePage = imageLinks[0]
                imagePage.click()
                driver.switch_to.window(driver.window_handles[-1])
                imageSiteUrl = driver.current_url
                # try downloading from wikipedia web page
                try:
                    driver.find_element(By.CLASS_NAME, 'internal').click()
                    driver.switch_to.window(driver.window_handles[-1])
                    logoImage = driver.current_url
                    companyName = re.sub(r'<|>|:|"|\/|\\|\||\?|\*', '', companyName)
                    if re.search('svg', logoImage):
                        urlretrieve(logoImage, f'{str(self.destPath)}/{companyName}.svg')
                    else:
                        raise ValueError('Not a svg File')
                # catch exception for try block of image download
                except Exception as e:
                    f = open(str(self.logPath), "a")
                    f.write(f'{companyName}; File can not be loaded\n')
                    f.close()
            else:
                f = open(str(self.logPath), "a")
                f.write(f'{companyName}; No Wikipedia Page for company\n')
                f.close()
            # close webdriver
            driver.quit()
            numberLogos = len([name for name in os.listdir(str(self.destPath)) if name != '.DS_STORE'])
            print(f'{numberLogos} from {len(companyNames)} logos downloaded')
            # check if downloads are fulfilled

        print('Successfully stored logos')
