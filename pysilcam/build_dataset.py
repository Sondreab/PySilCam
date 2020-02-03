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

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader


from pysilcam.config import PySilcamSettings
from pysilcam.process import extract_roi, extract_pixels



def simpleTest():
    #wget http://images.cocodataset.org/val2017/000000439715.jpg -O input.jpg
    test = cv2.imread("./input.jpg")
    print(test.shape)
    im = cv2.imread("./workstation.jpg")

    #cv2.imshow('input',im)
    #cv2.waitKey(0)


    cfg = get_cfg()
    cfg.merge_from_file("../detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can either use the https://dl.fbaipublicfiles.... url, or use the following shorthand
    cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
    predictor = DefaultPredictor(cfg)
    outputs = predictor(im)

    # look at the outputs. See https://detectron2.readthedocs.io/tutorials/models.html#model-output-format for specification
    outputs["instances"].pred_classes
    outputs["instances"].pred_boxes

    # We can use `Visualizer` to draw the predictions on the image.
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    pred = cv2.resize(v.get_image()[:, :, ::-1], (1280,960))
    cv2.imshow("visualizer", pred)
    cv2.waitKey(0)
    cv2.imwrite('./workstation-seg.jpg', pred)

def read_stats(directory = ''):
    csv_file = os.path.join(directory,'/proc/copepods-STATS.csv')
    with open(csv_file, newline='') as csvfile:
        stats = csv.DictReader(csvfile)
    return stats

def build_annotaion_dictionary(directory = ''):
    stats = read_stats(directory=directory)

    config_file = os.path.join(directory,'config.ini')
    settings = PySilcamSettings(config_file)

    height = 2050
    width = 2448
    
    record = {}
    dataset = []
    current_image = ''
    previous_image = ''
    idx = 0
    for row in stats:
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
                dataset.append(record)
            record = {}
            objects = []
            ##TODO: determine current_image path and file extension
            record['file_name'] = current_image + '.bmp'
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

# write a function that loads the dataset into detectron2's standard format
def get_balloon_dicts(img_dir):
    json_file = os.path.join(img_dir, "via_region_data.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns.values()):
        record = {}
        
        filename = os.path.join(img_dir, v["filename"])
        height, width = cv2.imread(filename).shape[:2]
        
        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
      
        annotations = v["regions"]
        objs = []
        for _, anno in annotations.items():
            assert not anno["region_attributes"]
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = list(itertools.chain.from_iterable(poly))

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
                "iscrowd": 0
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

def registerBalloondata():
    for d in ["train", "val"]:
        DatasetCatalog.register("balloon_" + d, lambda d=d: get_balloon_dicts("balloon/" + d))
        MetadataCatalog.get("balloon_" + d).set(thing_classes=["balloon"])

def testBalloondata():
    registerBalloondata()
    balloon_metadata = MetadataCatalog.get("balloon_train")
    dataset_dicts = get_balloon_dicts("balloon/train")
    for d in random.sample(dataset_dicts, 3):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=balloon_metadata, scale=0.5)
        vis = visualizer.draw_dataset_dict(d)
        im = vis.get_image()[:, :, ::-1]
        cv2.imshow('visualized sample', vis.get_image()[:, :, ::-1])
        cv2.waitKey(0)
        cv2.imwrite('./balloon-seg.jpg', im)
        

def trainBalloonModel():
    cfg = get_cfg()
    cfg.merge_from_file("../detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.DATASETS.TRAIN = ("balloon_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"  # initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough, but you can certainly train longer
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()

    evaluator = COCOEvaluator("balloon_val", cfg, False, output_dir="./output/")
    val_loader = build_detection_test_loader(cfg, "balloon_val")
    inference_on_dataset(trainer.model, val_loader, evaluator)

DIRECTORY = ''

simpleTest()
#testBalloondata()

#trainBalloonModel()



# another equivalent way is to use trainer.test