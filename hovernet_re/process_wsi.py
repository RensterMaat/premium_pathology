import sys
import cv2
import torch
import geojson
import openslide
import numpy as np
import xml.etree.ElementTree as ET
from pytorch_lightning import LightningModule
from monai.transforms import Lambdad, Compose, ToTensord
from monai.data import CacheDataset
from torch.utils.data import DataLoader
from util import format_output, output2annotations, normalize_staining
from pytorch_lightning import Trainer
from pathlib import Path

sys.path.insert(1, '/hpc/dla_patho/premium/wliu/hover_net')
from models.hovernet.net_desc import HoVerNet


class Processor(LightningModule):
    def __init__(self, slide_path, annotation_path):
        super().__init__()

        # setup hovernet
        self.net = HoVerNet(nr_types=6,mode='fast')
        ckpt = torch.load('/hpc/dla_patho/premium/rens/premium_pathology/weights/hovernet_fast_pannuke_type_tf2pytorch.tar')
        self.net.load_state_dict(ckpt['desc'], strict=True)

        # open slide
        self.slide = openslide.OpenSlide(str(slide_path))
        dimensions = np.array(self.slide.level_dimensions)

        # load annotation
        with open(annotation_path, 'r') as f:
            xml_file = ET.parse(f)

        # extract polygons from annotation
        self.rois = []
        for annotation in xml_file.getroot()[0]:
            roi = []
            for coordinate in annotation[0]:
                roi.append([coordinate.attrib['X'], coordinate.attrib['Y']])
            self.rois.append(np.array(roi).astype(float).astype(int))

        # create mask from annotation
        self.mask = np.zeros(np.flip(self.slide.dimensions))
        cv2.fillPoly(self.mask, self.rois, 1)
        self.mask = self.mask.astype(bool)

        # create and filter list of origins which will be inferenced
        xx = np.arange(0,self.slide.dimensions[0], 164)[:-1]
        yy = np.arange(0,self.slide.dimensions[1], 164)[:-1]

        self.origins = np.array([(x,y) for y in yy for x in xx])
        filtered_origins = self.origins[[self.mask[y,x] for x,y in self.origins]]

        # self.input is the input to the dataset
        self.input = [{'origin':origin, 'image':origin} for origin in filtered_origins]

        # output classes
        self.classes = {
            0 : "nolabe",
            1 : "neopla",
            2 : "inflam",
            3 : "connec",
            4 : "necros",
            5 : "no-neo",
        }

        # list for storing outputted annotations
        self.output = []

    def forward(self, x):
        return self.net(x)

    def test_dataloader(self):
        ds = CacheDataset(
            self.input,
            transform=Compose([
                Lambdad(
                    keys='image', 
                    func=lambda x: np.array(processor.slide.read_region(
                        (x[0] - 46, x[1] - 46),
                        0,
                        (256,256)
                    ).convert('RGB'))
                ),
                Lambdad(
                    keys='image',
                    func=lambda x: normalize_staining(x)[0]
                ),
                Lambdad(
                    keys='image', 
                    func = lambda x: x.transpose(2,0,1)
                ),
                ToTensord(keys='image')
            ]),
            cache_rate=0.01
        )

        return DataLoader(ds, batch_size=16)

    def test_step(self, batch, batch_ix):
        origins, images, _ = batch.values()
        output = self.net(images)

        for k in output.keys():
            output[k] = output[k].detach().cpu()

        formatted_output = format_output(output)
        annotations = [output2annotations(pred) for pred in formatted_output]
        
        for origin, annotation in zip(origins, annotations):
            for instance in annotation.values():
                dict_data = {}

                cc = (origin.cpu().numpy() + instance['contour']).tolist()
                cc.append(cc[0])

                dict_data["type"]="Feature"
                dict_data["id"]="PathCellObject"
                dict_data["geometry"]={
                    "type":"Polygon",
                    "coordinates":[cc]
                }
                dict_data["properties"]={
                    "isLocked":"false",
                    "measurements":[],
                    "classification": {
                        "name": self.classes[instance['type']]
                    }
                }

                self.output.append(dict_data)

    def save(self, save_path):
        with open(save_path, 'w') as file:
            geojson.dump(self.output, file)

r = Path('/hpc/dla_patho/premium/pathology/')

slide_dir = r / 'metastasis/isala'
annotation_dir = r / 'annotations/isala/core'
output_dir = r / 'hovernet_output/isala/core'

for annotation_file in list(annotation_dir.iterdir()):
    try:
        print(annotation_file.stem)
        slide_file = slide_dir / (annotation_file.stem + '.ndpi')


        processor = Processor(
            slide_file,
            annotation_file
        )

        trainer = Trainer(gpus=1)
        trainer.test(processor)

        save_path = output_dir / (annotation_file.stem + '.json')

        processor.save(save_path)
    except Exception as e:
        print(e)
