'''
 Browser automation for accessing https://www.image-map.net/
 and retrieving the bottom left and top right bounding box coordinates
 Theser are used to calculate other coordinates

 Input: None
 Output: Boundingbox coordinates - bottom_left, bottom_right, top_right, top_left

 When selecting the bounding box, first select the bottom_left point and then the
 top right
 
'''

from selenium import webdriver
from selenium.webdriver.common.keys import Keys  

import time

from boundingBox import getBoundingBoxCoords

from tkinter.filedialog import askopenfilename
from tkinter import *

import re

def extractBoundingBoxCoords(File):
    # open browser
    browser = webdriver.Chrome()  
    browser.get('https://www.image-map.net/') 


    #print("Loaded")
    # upload the currently labeling image 
    upload_image_button = browser.find_element_by_id('image-mapper-file')
    upload_image_button.send_keys(File)

    # wait for drawing bounding box and clicking "Show me the code!"
    while(True):
        time.sleep(1)
        coord_text = browser.find_element_by_id('modal-code-result').get_attribute('value')

        if(len(coord_text) != 0):
            break

    # get the cordinates of bottom left and top right
    pattern = "coords=\"([\d,]*)\""
    two_coords = re.search(pattern, coord_text).group(1)

    # close the browser
    browser.close()
    #print(two_coords)

    # get the remaining cordinates
    coords = getBoundingBoxCoords(two_coords)
    print('[scrapeCoordinates.py]: ',coords)
    return coords

if __name__ == "__main__":
	root = Tk()
	img_file = askopenfilename(parent=root, initialdir='frames',title='Choose an image.')
	extractBoundingBoxCoords(img_file)
