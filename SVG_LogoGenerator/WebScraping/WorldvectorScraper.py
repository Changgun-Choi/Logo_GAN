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
from shutil import SpecialFileError
import string
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool
from pathlib import Path
from urllib.request import urlretrieve
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options


class worldvectorScraper:
    def __init__(self, webdriverPath, destPath):
        self.driverPath = Path(webdriverPath)
        self.destPath = Path(destPath)
        self.baseUrl = 'https://worldvectorlogo.com/alphabetical'
        self.logoList = list(string.ascii_lowercase) + list(range(10))



    def urlDownload(self, url):
        try:
            urlretrieve(url, f'{str(self.destPath)}/{url.split("/")[-1]}')
        except:
            pass

    def scraper(self):
        for logoListElement in tqdm(self.logoList):
            print(logoListElement)
            endPage = False
            page = 1
            while not endPage:
                searchLink = f'{self.baseUrl}/{logoListElement}/{page}'
                chromeOptions = Options()
                chromeOptions.add_argument("--headless")
                driver = webdriver.Chrome(
                    str(self.driverPath), options=chromeOptions)
                driver.get(searchLink)
                imageResults = driver.find_elements(By.CLASS_NAME, 'logo__img')
                if imageResults:
                    svgLogos = [result.get_attribute("src") for result in imageResults]
                    pool = ThreadPool(100)
                    result = pool.starmap(self.urlDownload, zip(svgLogos))
                    page += 1
                else:
                    endPage = True
                driver.close()
        print('Successfully stored logos')
