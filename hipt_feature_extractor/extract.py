import os
import torch
import yaml
import sys
import h5py
import argparse
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from pathlib import Path
from openslide import OpenSlide

with open(Path(__file__).parent / "config.yaml", "r") as f:
    config = yaml.safe_load(f)

sys.path.append(str(Path(config["hipt_repository_path"]) / "HIPT_4K"))

from hipt_4k import HIPT_4K


class PatchDataset(Dataset):
    def __init__(self, patch_coordinates, transform):
        self.patch_coordinates = patch_coordinates
        self.transform = transform

    def __getitem__(self, ix):
        coords = self.patch_coordinates[ix]
        img = self.transform(coords)
        return img

    def __len__(self):
        return self.patch_coordinates.shape[0]


class HIPTFeatureExtractor(pl.LightningModule):
    def __init__(self, slide_path, patches_path, output_dir=None, patch_size=4096):
        super().__init__()

        self.case_id = slide_path.stem
        self.slide = OpenSlide(str(slide_path))
        self.patch_coordinates = h5py.File(patches_path, "r")["coords"]

        self.output_dir = output_dir
        self.patch_size = patch_size
        self.model = HIPT_4K()
        self.embeddings = []

    def forward(self, x):
        return self.model(x)

    def test_step(self, batch, batch_idx):
        out = self(batch)
        self.embeddings.append(out.detach().cpu())

    def test_epoch_end(self, *args):
        all_embeddings = torch.concatenate(self.embeddings)
        torch.save(all_embeddings, Path(self.output_dir) / (self.case_id + ".pt"))

    def setup(self, stage="test"):
        transform = transforms.Compose(
            [
                transforms.Lambda(
                    lambda coords: np.array(
                        self.slide.read_region(
                            coords, 1, (self.patch_size, self.patch_size)
                        )
                    )[:, :, :-1]
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=0.5, std=0.5),
            ]
        )

        self.dataset = PatchDataset(self.patch_coordinates, transform)

    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=1, shuffle=False, num_workers=12)


def extract_patches(slide_dir, patch_dir, output_dir):
    for slide_path in list(Path(slide_dir).iterdir()):
        case_id = slide_path.stem
        print(case_id)
        patches_path = Path(patch_dir) / (case_id + ".h5")

        if not patches_path.exists():
            print(f"No patch coordinate file found for {case_id}")
            continue

        fe = HIPTFeatureExtractor(slide_path, patches_path, output_dir)

        trainer = pl.Trainer(accelerator="gpu")
        trainer.test(fe)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--slide_dir",
    type=str,
    default=config["slide_dir"],
    help="Directory containing WSI slides for extraction.",
)
parser.add_argument(
    "--preprocess_dir",
    type=str,
    default=config["preprocess_dir"],
    help="Directory containing the preprocessing results of the WSIs to process. Should have the same structure as wsi_dir, except for the last layer which should contain the masks, patches and stitches subfolders",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default=config["feature_dir"],
    help="Directory for saving the output feature vectors.",
)


if __name__ == "__main__":
    args = parser.parse_args()

    for root, _, files in os.walk(args.slide_dir):
        sub_dir = root.replace(args.slide_dir, "")
        if sub_dir and sub_dir[0] == "/":
            sub_dir = sub_dir[1:]

        patches_dir = Path(args.preprocess_dir) / sub_dir / "patches"
        if files:
            print("Extracting features from slides in:", root)
            print("Using patch coordinates in:", patches_dir)
            extract_patches(root, patches_dir, args.output_dir)
