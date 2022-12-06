import pytorch_lightning as pl
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import random
import matplotlib.pyplot as plt
from torchvision.models import densenet121
from monai.data import CacheDataset, DataLoader
from monai.transforms import Lambdad, ToTensord, ScaleIntensityd, Compose
from PIL import Image
from torchmetrics import AUROC
from sklearn.metrics import roc_auc_score, roc_curve


class Index():
    def __init__(self, path='/home/rens/hpc/rens/output/patches/index.csv'):
        self.index = pd.read_csv(path).set_index('path')
        self.index = self.index[~self.index.dcb.isna()]

    def get(self, **kwargs):
        subset = self.index
        for key, value in kwargs.items():
            if isinstance(value, list):
                subset = subset[subset[key].isin(value)]
            else:
                subset = subset[subset[key] == value]
        return subset


class Dataset(pl.LightningDataModule):
    def __init__(
        self, dev_set=None, test_set=None, 
        train_transform=None, val_transform=None, test_transform=None, 
        target='dcb'
    ):
        super().__init__()
        self.dev_set = dev_set
        self.test_set = test_set

        default_transform = self.test_transform=Compose([
            Lambdad(['image'], 
                lambda x: np.array(Image.open(x)).transpose(2,0,1).astype(np.float32)
            ),
            ScaleIntensityd(['image']),
            ToTensord(keys=['image']),
        ])

        if train_transform is None:
            self.train_transform = default_transform
        else:
            self.train_transform = train_transform

        if val_transform is None:
            self.val_transform = default_transform
        else:
            self.val_transform = val_transform

        if test_transform is None:
            self.test_transform = default_transform
        else:
            self.test_transform = test_transform

        self.target = target

    def setup(self, stage=None):
        if self.dev_set is not None:
            patients = self.dev_set.patient.unique()
            random.shuffle(patients)
            split_ix = int(len(patients) * 0.8)

            train_patients = {pt for pt in patients[:split_ix]}
            val_patients = {pt for pt in patients[split_ix:]}

            assert train_patients.isdisjoint(val_patients)

            self.train_input = self.df_to_input(
                self.dev_set[self.dev_set.patient.isin(train_patients)]
            )
            self.val_input = self.df_to_input(
                self.dev_set[self.dev_set.patient.isin(val_patients)]
            )

        if self.test_set is not None:
            self.test_input = self.df_to_input(self.test_set)

        if self.dev_set is not None and self.test_set is not None:
            test_patients = {pt for pt in self.test_set.patient}

            assert test_patients.isdisjoint(train_patients)
            assert test_patients.isdisjoint(val_patients)

    def make_dataloader(self, input, transform, shuffle=False):
        ds = CacheDataset(input, transform=transform, cache_rate=0)
        dl = DataLoader(ds, batch_size=16, shuffle=shuffle, num_workers=12)
        
        return dl

    def train_dataloader(self):
        return self.make_dataloader(self.train_input, self.train_transform, shuffle=True)

    def val_dataloader(self):
        return self.make_dataloader(self.val_input, self.val_transform,)

    def test_dataloader(self):
        return self.make_dataloader(self.test_input, self.test_transform,)

    def df_to_input(self, df):
            return [
                {'path':str(ix),
                'image':str(ix),
                'label':int(r[self.target])}
                for ix, r in df.iterrows()
            ]


class Model(pl.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()

        self.model = densenet121(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad=False
        self.model.classifier = nn.Linear(1024,1)

        self.results = pd.DataFrame(columns=['pred'])

        self.lr = lr

    def forward(self, x):
        out = self.model(x)
        return torch.sigmoid(out)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        # lr_scheduler = LinearWarmupCosineAnnealingLR(
        #     optimizer,
        #     warmup_epochs=3,
        #     max_epochs=40,
        #     eta_min=1e-5,
        #     warmup_start_lr=1e-5
        # )
        return {
            'optimizer':optimizer,
            # 'lr_scheduler':{
            #     'scheduler':lr_scheduler,
            #     'name':'learning_rate',
            #     'interval':'epoch',
            #     'frequency':1
            # }
        }

    def training_step(self, batch, batch_idx):
        x,y = batch['image'], batch['label'] 
        y_hat = self(x)

        loss = nn.BCELoss()(y_hat, y.unsqueeze(1).float())

        self.logger.experiment.add_scalars('loss',{'train':loss.detach()},self.global_step)

        batch_dictionary = {
            'loss':loss
        }

        return batch_dictionary

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.logger.experiment.add_scalar(
            'avg_loss_train',
            avg_loss,
            self.current_epoch
        )

        # lr = model.optimizers().state_dict()['param_groups'][0]['lr']
        # self.logger.experiment.add_scalar(
        #     'learning_rate',
        #     lr,
        #     self.current_epoch
        # )

    def validation_step(self, batch, batch_idx):
        x,y = batch['image'], batch['label'] 
        y_hat = self(x)

        loss = nn.BCELoss()(y_hat, y.unsqueeze(1).float())
        self.logger.experiment.add_scalars('loss',{'val':loss.detach()},self.global_step)

        # auc = AUROC()(y_hat, y.unsqueeze(1).int())
        # self.logger.experiment.add_scalars('auc',{'val':auc},self.global_step)

        batch_dictionary = {
            'loss':loss,
            'y':y,
            'y_hat':y_hat
        }

        return batch_dictionary

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.logger.experiment.add_scalar(
            'avg_loss_val',
            avg_loss,
            self.current_epoch
        )

        auc = AUROC(pos_label=1)(
            torch.concat([x['y_hat'].squeeze() for x in outputs]),
            torch.concat([x['y'] for x in outputs])
        )
        self.logger.experiment.add_scalar(
            'auc_val',
            auc,
            self.current_epoch
        )

    def test_step(self, batch, batch_idx):
        paths, images = batch['path'], batch['image']
        y_hat = self(images)
        
        for p, f in zip(paths, y_hat.cpu().detach().numpy()):
            self.results.loc[p] = f

    def reset_results(self):
        self.results = pd.DataFrame(columns=['pred'])


def roc(df):
    df['x'] = [int(ix.split('x')[-1].split('_')[0]) for ix in df.index]
    df['y'] = [int(ix.split('y')[-1].split('.')[0]) for ix in df.index]

    df['patient'] = [ix.split('/')[-1].split('_')[0].replace('-','_') for ix in df.index]

    dmtr = pd.read_csv('/home/rens/repos/PREMIUM/code/radiomics_paper/dmtr.csv').set_index('id')

    dcb = []
    for patient in df.patient:
        try:
            dcb.append(dmtr.loc[patient, 'dcb'])
        except:
            dcb.append(float('nan'))
    df['dcb'] = dcb

    plt.figure(figsize=(12,12))

    # patch level
    fpr, tpr, _ = roc_curve(df.dcb, df.pred)
    auc = roc_auc_score(df.dcb, df.pred)
    plt.plot(fpr, tpr, label=f'Patch level AUC={auc:.3f}')

    # slide level
    slide_level_df = df.groupby('patient')[['pred','dcb']].mean()
    fpr, tpr, _ = roc_curve(slide_level_df.dcb, slide_level_df.pred)
    auc = roc_auc_score(slide_level_df.dcb, slide_level_df.pred)
    plt.plot(fpr, tpr, label=f'Slide level AUC={auc:.3f}')

    plt.plot([0,1],[0,1], linestyle='--', c='gray')

    plt.xlabel('1 - specificity')
    plt.ylabel('sensitivity')
    plt.legend()
    plt.show()
