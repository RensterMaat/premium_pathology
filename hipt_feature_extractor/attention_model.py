import torch
import pandas as pd
import torch.nn as nn
import pytorch_lightning as pl
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from monai.data import CacheDataset


class DINO_4k_feature_dataset(Dataset):
    def __init__(self, target):
        features = pd.read_csv('/home/rens/repos/premium_pathology/hipt_feature_extractor/data/features.csv').set_index('Unnamed: 0')
        labels = pd.read_csv('/home/rens/repos/premium_pathology/hipt_feature_extractor/data/labels.csv').set_index('Unnamed: 0')
        labels.index = [Path(ix).stem for ix in labels.index]

        self.target = target

        self.data = features.join(labels[target], on='slide')
        self.data[target] = self.data[target].astype(int)

        self.slides = self.data.slide.unique()
        
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.float()[0])
        ])

    def __len__(self):
        return len(self.slides)
    
    def __getitem__(self, ix):
        slide = self.slides[ix]

        slide_data = self.data[self.data.slide == slide]

        features = slide_data[[f'feature{f}' for f in range(192)]].to_numpy()
        x = self.transforms(features)

        y = torch.tensor(slide_data[self.target][0]).float()

        return x, y
    

class CLAM_feature_dataset(Dataset):
    def __init__(self, target):
        self.features_root = Path('/hpc/dla_patho/premium/pathology/clam_features/primary_vs_metastasis/pt_files')
        self.labels = pd.read_csv('/hpc/dla_patho/premium/rens/premium_pathology/hipt_feature_extractor/data/labels_clam.csv').set_index('Unnamed: 0')
        self.labels.index = [Path(ix).stem for ix in self.labels.index]

        self.target = target
        
        self.transforms = transforms.Compose([
            # transforms.ToTensor(),
            transforms.Lambda(lambda x: x.float())
        ])

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, ix):
        x = torch.load(self.features_root / (self.labels.index[ix] + '.pt'))
        x = self.transforms(x)

        y = self.labels.iloc[ix][self.target]
        y = torch.tensor(y).float()

        return x, y
          
        
class GatedAttentionLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0):
        super().__init__()
        
        self.attention_a = [
            nn.Linear(input_dim,output_dim),
            nn.Tanh()
        ]
        self.attention_b = [
            nn.Linear(input_dim,output_dim),
            nn.Sigmoid()
        ]

        self.attention_c = [
            nn.Linear(output_dim, 1),
            nn.Softmax(dim=1)
        ]

        if dropout:
            self.attention_a.append(nn.Dropout(dropout))
            self.attention_b.append(nn.Dropout(dropout))

        self.attention_a = nn.Sequential(*self.attention_a) 
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Sequential(*self.attention_c)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        out = self.attention_c(A)

        return out
    
class AttentionModel(pl.LightningModule):
    def __init__(self, dataset, dropout=0):
        super().__init__()

        self.dataset = dataset

        self.attention_layer = GatedAttentionLayer(1024, 128, dropout)
        self.classifier = nn.Sequential(*[
            nn.Linear(1024, 1),
            # nn.ReLU(),
            # nn.Linear(128,128),
            # nn.ReLU(),
            # nn.Linear(128,1),
            nn.Sigmoid()
        ])

        self.criterion = nn.BCELoss()

    def forward(self, x):
        A = self.attention_layer(x).transpose(1,2)
        M = torch.matmul(A, x)
        out = self.classifier(M)

        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = self.criterion(y_hat.squeeze(), y.squeeze())

        self.log('train_loss', loss)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = self.criterion(y_hat.squeeze(), y.squeeze())

        self.log('val_loss', loss)

        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-3)
        return optimizer
    
    def setup(self, **kwargs):
        n = len(self.dataset)
        train_n = int(n*0.8)

        self.train_dataset, self.val_dataset = random_split(self.dataset, [train_n, n-train_n])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size = 1, num_workers=18)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size = 1, num_workers=18)
        
import wandb
from pytorch_lightning.loggers import WandbLogger


if __name__ == '__main__':
    dataset = CLAM_feature_dataset(target='primary')
    model = AttentionModel(dataset)

    logger = WandbLogger(project='pathology', name='debug_attention_on_clam_features')

    trainer = pl.Trainer(
        accelerator='gpu',
        accumulate_grad_batches=16,
        logger=logger
        # overfit_batches=10
    )

    trainer.fit(model)

    wandb.finish()

