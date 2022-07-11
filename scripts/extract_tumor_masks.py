import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image
from skimage.morphology import binary_dilation, area_closing, binary_erosion
from tqdm import tqdm
from scipy.ndimage import binary_fill_holes, zoom
from joblib import Parallel, delayed


r= Path('/hpc/dla_patho/premium')

overlays = sorted(list(Path('/hpc/dla_patho/premium/wliu/output_20220520_mst/tif').iterdir()))

ix=0

def extract_patches(overlay_path):

    Image.MAX_IMAGE_PIXELS = None

    # os.makedirs(r / 'patches' / overlay_path.stem, exist_ok=True)

    image = np.array(Image.open(overlay_path))
    neoplasm = np.all(image == (255,0,0), axis=-1).astype(int)
    # neoplasm = neoplasm[1536:1536+512,1536:1536+512]

    img = neoplasm
    for step in range(40):
        img = binary_dilation(img)

    img = area_closing(img)
    for step in range(80):
        img = binary_erosion(img)

    img = area_closing(img)

    img = zoom(img, 1/8, order=0)

    Image.fromarray(img).save(r / 'rens' / 'output' / 'tumor_masks' / (overlay_path.stem + '.tif'))

Parallel(n_jobs=-1)(
    delayed(extract_patches)(overlay) 
    for overlay in tqdm(overlays)
)