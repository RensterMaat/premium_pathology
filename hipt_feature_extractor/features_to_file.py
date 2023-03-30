import torch
import pandas as pd
from pathlib import Path

# feature_dir = Path('/mnt/hpc/pathology/hipt_features/primary_vs_metastasis_4k')
feature_dir = Path("/mnt/hpc/pathology/clam_features/primary_vs_metastasis/pt_files")
dmtr = pd.read_csv("/mnt/c/Users/user/data/tables/dmtr.csv").set_index("id")

# primary_slides = [file.stem for c in Path('/mnt/hpc/pathology/hipt_preprocessed/4096/primary/').iterdir() for file in (c / 'patches').iterdir()]
# metastasis_slides = [file.stem for c in Path('/mnt/hpc/pathology/hipt_preprocessed/4096/metastasis/').iterdir() for file in (c / 'patches').iterdir()]

primary_slides = [
    file.stem
    for c in Path("/mnt/hpc/pathology/clam_preprocessed/primary/").iterdir()
    for file in (c / "patches").iterdir()
]
metastasis_slides = [
    file.stem
    for c in Path("/mnt/hpc/pathology/clam_preprocessed/metastasis/").iterdir()
    for file in (c / "patches").iterdir()
]

labels = pd.DataFrame(columns=["patient", "center", "primary", "metastasis"])

for path in feature_dir.iterdir():
    stem = path.stem

    if "VU" in stem:
        patient = stem[:11]
        center = "vumc"
    elif "MAX" in stem:
        patient = stem[:7].replace("-", "_")
        center = "maxima"
    elif "LU" in stem:
        patient = "PREM" + stem.split("PREM")[1][:8].replace("-", "_")
        center = "lumc"
    elif "MS" in stem:
        patient = stem[:11].replace("-", "_")
        center = "mst"
    elif "AM" in stem:
        patient = stem[:11].replace("-", "_")
        center = "amphia"
    elif "IS" in stem:
        patient = stem[:11].replace("-", "_")
        center = "isala"
    elif "IM" in stem:
        patient = stem[:6]
        center = "umcu"
    elif "UNI" in stem:
        patient = stem[:7].replace("-", "_")
        center = "umcu"
    elif "RA" in stem:
        patient = stem[:11].replace("-", "_")
        center = "radboud"
    elif "M-" in stem[:2]:
        patient = stem[:5].replace("-", "_")
        center = "umcu"
    elif "ZU" in stem:
        patient = stem[:11]
        center = "zuyderland"
    else:
        patient = float("nan")
        center = float("nan")
        print(stem)

    primary = stem in primary_slides
    metastasis = stem in metastasis_slides

    labels.loc[path] = [patient, center, primary, metastasis]
 labels = labels.join(on="patient", other=dmtr[["dcb", "response", "typbraf0n"]])

labels.to_csv(
    "/home/rens/repos/premium_pathology/hipt_feature_extractor/data/labels_clam.csv"
)

import h5py
import numpy as np
from tqdm import tqdm

features = pd.DataFrame(columns=[f"feature{x}" for x in range(1024)] + ["patient", "y"])

rows = []
for ix, row in tqdm(list(labels.iterrows())):
    f = torch.load(ix)
    f = pd.DataFrame(f, columns=[f"feature{x}" for x in range(1024)])
    f["patient"] = row["patient"]
    f["slide"] = ix.stem

    try:
        coords_file = (
            Path("/mnt/hpc/pathology/clam_preprocessed")
            / ("primary" if row["primary"] else "metastasis")
            / row["center"]
            / "patches"
            / (ix.stem + ".h5")
        )
        coords = h5py.File(coords_file, "r")

        f["x"] = np.array(coords["coords"])[:, 0]
        f["y"] = np.array(coords["coords"])[:, 1]
    except:
        print(ix)

    rows.append(f)

features = pd.concat(rows)
features.to_csv(
    "/home/rens/repos/premium_pathology/hipt_feature_extractor/data/features_clam.csv"
)
