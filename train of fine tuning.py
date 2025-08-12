import os
import sys
import numpy as np
import torch
import pytorch_lightning as pl
from dataset import ISICLoader
from metrics import iou_score, dice_score, dice_tversky_loss
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from models.AC_MambaSeg import AC_MambaSeg
from collections import OrderedDict

IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    SAVE_DIR = ""
else:
    SAVE_DIR = ""
os.makedirs(SAVE_DIR, exist_ok=True)


DATA_PATH = '/content/dataset.npz'
data = np.load(DATA_PATH)
x, y = data["images"], data["masks"]
x_trainval, x_test, y_trainval, y_test = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True)
x_train, x_val, y_train, y_val = train_test_split(x_trainval, y_trainval, test_size=0.1, random_state=42, shuffle=True)



class Segmentor(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model


    def forward(self, x):
        return self.model(x)


    def _step(self, batch):
        image, y_true = batch
        y_pred = self.model(image)
        loss = dice_tversky_loss(y_pred, y_true)
        dice = dice_score(y_pred, y_true)
        iou = iou_score(y_pred, y_true)
        return loss, dice, iou


    def training_step(self, batch, batch_idx):
        loss, dice, iou = self._step(batch)
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_dice", dice, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_iou", iou, on_step=True, on_epoch=True, prog_bar=True)
        return loss


    def validation_step(self, batch, batch_idx):
        loss, dice, iou = self._step(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_dice", dice, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_iou", iou, on_step=False, on_epoch=True, prog_bar=True)
        return {"val_loss": loss, "val_dice": dice, "val_iou": iou}


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-5, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5, verbose=True
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_dice"}

model = AC_MambaSeg().cuda()
ckpt_path = ""
ckpt = torch.load(ckpt_path, map_location='cuda')


new_state_dict = OrderedDict()
for k, v in ckpt['state_dict'].items():
    new_k = k.replace("model.", "") if k.startswith("model.") else k
    new_state_dict[new_k] = v


model.load_state_dict(new_state_dict)

train_loader = DataLoader(
    ISICLoader(x_train, y_train),
    batch_size=4, pin_memory=True, shuffle=True, num_workers=2, drop_last=True, prefetch_factor=8
)
val_loader = DataLoader(
    ISICLoader(x_val, y_val, typeData="val"),
    batch_size=1, num_workers=2, prefetch_factor=16
)


checkpoint_cb = pl.callbacks.ModelCheckpoint(
    dirpath=SAVE_DIR,
    filename="ckpt-{val_dice:.4f}",
    monitor="val_dice",
    mode="max",
    save_top_k=1,
    verbose=True,
    save_weights_only=False
)


early_stop_cb = pl.callbacks.EarlyStopping(
    monitor="val_dice",
    mode="max",
    patience=7,
    verbose=True,
    min_delta=1e-4
)


progress_bar = pl.callbacks.TQDMProgressBar()


trainer = pl.Trainer(
    default_root_dir=SAVE_DIR,
    logger=True,
    benchmark=True,
    max_epochs=10,
    precision=16,
    enable_progress_bar=True,
    callbacks=[checkpoint_cb, early_stop_cb, progress_bar],
    log_every_n_steps=1,
    num_sanity_val_steps=0,
)



segmentor = Segmentor(model)
trainer.fit(segmentor, train_loader, val_loader)



test_save_path = os.path.join(SAVE_DIR, "data.npz")
np.savez(test_save_path, image=x_test, mask=y_test)
print(f" Test data saved to {test_save_path}")


final_model_path = os.path.join(SAVE_DIR, "model.ckpt")
trainer.save_checkpoint(final_model_path)
print(f" Full model saved to {final_model_path}")


