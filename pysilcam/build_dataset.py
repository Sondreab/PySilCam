#Based on the tutorial found at https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5

# import some common libraries
import os
import numpy as np
import cv2
import json
import csv
import itertools
import random
import collections

from pysilcam.config import PySilcamSettings
from pysilcam.process import extract_roi, extract_pixels


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
    
    record = {}
    dataset = []
    for image in stats.values():
        previous_image = current_image
        current_image = stats['export name'].split('-')[0]

        probabilities = array([row['probability_oil'], row['probability_bubble'], row['probability_faecal_pellets'], 
                         row['probability_copepod'], row['probability_diatom_chain'], row['probability_oily_gas'], row['probability_other']])
        class_probability = np.amax(probabilities)
        class_id = np.argmax(probabilities)

        #if the class is other (void) or it is lower than the desired threshold of confidence, skip adding the object
        if ((class_id == 6) or (class_probability < settings.Process.threshold)):
            continue

        #check if we're evaluating particle in a new image
        if current_image != previous_image: 
            new_image = True 
        
        #if we are starting a new image,add record to the dataset dictionary and reset the current record
        if (new_image): 
            if record:
                for keys,values in record.items():
                    print(keys)
                    print(values)
                dataset.append(record)
            record = {}
            objects = []
            ##TODO: determine current_image path and file extension
            record["file_name"] = current_image + '.bmp'
            record["image_id"] = idx
            record["height"] = height
            record["width"] = width
        
        bbox = [row['min_r'], row['min_c'], row['max_r'], row['max_c']]
        px, py = extract_pixels(os.path.join(directory,'/export/'+current_image+'-SEG.bmp'), bbox)
        poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
        poly = list(itertools.chain.from_iterable(poly))

        obj = {
            "bbox": bbox,
            "bbox_mode": BoxMode.XYXY_ABS,
            "segmentation": [poly],
            "category_id": class_id,
            "iscrowd": 0
        }
        objects.append(obj)
    return dataset

DIRECTORY = '/home/sondreab/Desktop/data_test'

read_stats(DIRECTORY)

#build_annotaion_dictionary(directory=DIRECTORY)
#testBalloondata()

#trainBalloonModel()



# another equivalent way is to use trainer.test