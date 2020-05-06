'''
    Get bounding box cordinates from bottom left and top right cordinates

    https://www.image-map.net/
'''
import re

def getBoundingBoxCoords(two_coords):
    while(True):

        #cord = re.findall('\d+', input("Enter bottom left and top right cordinates(eg: 384,393,500,200): "))
        cord = re.findall('\d+', two_coords)

        b_left = '{} {} '.format(cord[0], cord[1])
        t_right = '{} {} '.format(cord[2], cord[3])

        b_right = '{} {} '.format(cord[2], cord[1])
        t_left = '{} {} '.format(cord[0], cord[3])

        coords = ''.join([t_left, b_right])#([b_left, b_right, t_right, t_left])
        print('[boundingBox.py]', coords)

        return coords
