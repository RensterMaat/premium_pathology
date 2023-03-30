import yaml
import torch
import pandas as pd
import h5py
import numpy as np
from tqdm.notebook import tqdm
from pathlib import Path


with open(Path(__file__).parent / "config.yaml", "r") as f:
    config = yaml.safe_load(f)


def extract_labels():
    feature_dir = Path(config["feature_dir"])
    dmtr = pd.read_csv("/mnt/c/Users/user/data/tables/dmtr.csv").set_index("id")

    primary_slides = [
        file.stem
        for c in (Path(config["preprocess_dir"]) / "primary").iterdir()
        for file in (c / "patches").iterdir()
    ]
    metastasis_slides = [
        file.stem
        for c in (Path(config["preprocess_dir"]) / "metastasis").iterdir()
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

    labels.to_csv(Path(__file__).parent / "data" / "labels.csv")

    return labels


def extract_features(labels):
    features = pd.DataFrame(
        columns=[f"feature{x}" for x in range(192)] + ["patient", "y"]
    )

    rows = []
    for ix, row in tqdm(list(labels.iterrows())):
        f = torch.load(ix)
        f = pd.DataFrame(f, columns=[f"feature{x}" for x in range(192)])
        f["patient"] = row["patient"]
        f["slide"] = ix.stem

        try:
            coords_file = (
                Path(config["preprocess_dir"])
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
    features.to_csv(Path(__file__).parent / "data" / "features.csv")


if __name__ == "__main__":
    labels = extract_labels()
    extract_features(labels)
