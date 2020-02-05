#Based on the tutorial found at https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5

import torch, torchvision
print(torch.__version__)

# Some basic setup
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import os
import numpy as np
import cv2
import json
import csv
import itertools
import random
import collections

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from pysilcam.config import PySilcamSettings
from pysilcam.process import extract_roi


def extract_pixels(im, bbox):
    ''' given a binary image (im) and bounding box (bbox), this will return all activated pixel coordinates in x and y

    returns:
      all_points_x, all_points_y
    '''
    

    roi = im[bbox[0]:bbox[2], bbox[1]:bbox[3], 0] # bbox[row, column]
    cv2.imshow("visualizer", roi)
    cv2.waitKey(0)    
    coloumns = bbox[3] - bbox[1]
    rows = bbox[2] - bbox[0]
    all_points_x, all_points_y = [], []
    #print(im.shape)
    #print(roi.shape)
    #print('({}, {})'.format(rows, coloumns))
    #print('iterating')

    """
    contours,_=cv2.findContours(roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(type(contours))
    print(contours)
    """
    for r in range(rows):
        #print('(r: {})'.format(r))
        for c in range(coloumns):
            #print('(c: {})'.format(c))
            if roi[r,c] == 255:
                all_points_x.append(bbox[1] + c)
                all_points_y.append(bbox[0] + r)

    return all_points_x, all_points_y


def read_stats(directory = ''):
    csv_file = os.path.join(directory+'/proc/', 'copepods-STATS.csv')
    json_file = os.path.join('/home/sondreab/Desktop/test', "via_region_data.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)
    stats = collections.defaultdict(dict)
    with open(csv_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for rows in reader:
            #for key in rows.keys():
            stats[rows['export name'].split('-')[0]][rows['particle index']] = rows
    
    return stats

def build_annotaion_dictionary(directory = ''):
    stats = read_stats(directory=directory)

    config_file = os.path.join(directory,'config.ini')
    settings = PySilcamSettings(config_file)

    height = 2050
    width = 2448
    
    dataset = []
    print('Image:')
    for image, particle in stats.items():
        print('\t{}'.format(image))
        record = {}
        objects = []
        ##TODO: determine current_image path and file extension
        record["file_name"] = image + '.bmp'
        record["image_id"] = image
        record["height"] = height
        record["width"] = width
        print('\tParticle:')
        for index, fields in particle.items():
            print('\t{}'.format(index))
            probabilities = np.array([float(fields['probability_oil']), float(fields['probability_bubble']), float(fields['probability_faecal_pellets']), 
                                   float(fields['probability_copepod']), float(fields['probability_diatom_chain']), float(fields['probability_oily_gas']), 
                                   float(fields['probability_other'])])
            #print(probabilities)
            class_probability = np.amax(probabilities)
            class_id = np.argmax(probabilities)

            #if the class is other (void) or it is lower than the desired threshold of confidence, skip adding the object
            if ((class_id == 6) or (class_probability < settings.Process.threshold)):
                continue
            
            minr, minc, maxr, maxc = fields['minr'], fields['minc'], fields['maxr'], fields['maxc']
            bbox = [int(float(minr)), int(float(minc)), int(float(maxr)), int(float(maxc))]
            im = cv2.imread(os.path.join(directory+'/export/'+image+'-SEG.bmp'))
            px, py = extract_pixels(im, bbox)
            #print(px)
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            #poly = list(itertools.chain.from_iterable(poly))
            obj = {
                "bbox": bbox,
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": class_id,
                "iscrowd": 0
            }
            objects.append(obj)
        record["annotations"] = objects
        dataset.append(record)
    return dataset

DIRECTORY = '/home/sondreab/Desktop/data_test'

build_annotaion_dictionary(DIRECTORY)

#build_annotaion_dictionary(directory=DIRECTORY)
#testBalloondata()

#trainBalloonModel()



# another equivalent way is to use trainer.test