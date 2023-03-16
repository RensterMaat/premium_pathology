import torch
import h5py
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from hipt_model_utils import get_vit256
from pathlib import Path
from openslide import OpenSlide


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
    def __init__(self, slide_path):
        super().__init__()
        self.case_id = slide_path.stem
        self.slide = OpenSlide(slide_path)

        self.model = get_vit256(CHECKPOINT_PATH)

        self.embeddings = []

    def forward(self, x):
        return self.model(x)

    def test_step(self, batch, batch_idx):
        out = self(batch)
        self.embeddings.append(out.detach().cpu())

    def test_epoch_end(self, *args):
        all_embeddings = torch.concatenate(self.embeddings)
        torch.save(all_embeddings, FEATURE_DIR / (self.case_id + ".pt"))

    def setup(self, stage="test"):
        patches_path = (
            PATCH_DIR / "metastasis" / center / "patches" / (self.case_id + ".h5")
        )
        patch_coordinates = h5py.File(patches_path, "r")["coords"]

        transform = transforms.Compose(
            [
                transforms.Lambda(
                    lambda coords: np.array(
                        self.slide.read_region(coords, 1, (PATCH_SIZE, PATCH_SIZE))
                    )[:, :, :-1]
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=0.5, std=0.5),
            ]
        )

        self.dataset = PatchDataset(patch_coordinates, transform)

    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=64, shuffle=False, num_workers=24)


PATHOLOGY_ROOT = Path("/mnt/hpc/pathology")

WSI_DIR = PATHOLOGY_ROOT / "metastasis"
PATCH_DIR = PATHOLOGY_ROOT / "hipt_preprocessed"
FEATURE_DIR = PATHOLOGY_ROOT / "hipt_features/primary_vs_metastasis"

CHECKPOINT_PATH = (
    "/home/rens/repos/premium_pathology/hipt/checkpoints/vit256_small_dino.pth"
)

PATCH_SIZE = 256

center = "radboud"

for slide_path in sorted(list((WSI_DIR / center).iterdir()))[19:]:
    print(slide_path.stem)

    patches_path = (
        PATCH_DIR / "metastasis" / center / "patches" / (slide_path.stem + ".h5")
    )

    if not patches_path.exists():
        continue

    fe = HIPTFeatureExtractor(slide_path)
    trainer = pl.Trainer(accelerator="gpu")

    trainer.test(fe)
