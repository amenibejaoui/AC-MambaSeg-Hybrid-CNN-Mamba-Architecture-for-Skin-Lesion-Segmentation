import os
import numpy as np
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import random


from dataset import ISICLoader
from metrics import iou_score, dice_score
from torch.utils.data import DataLoader
from models.AC_MambaSeg import AC_MambaSeg


# --- 1. Paths ---
DATA_PATH = '/content/drive/MyDrive/event/test_data.npz'
CHECKPOINT_PATH = '/content/drive/MyDrive/event/ckpt-val_dice=0.9674.ckpt'
SAVE_DIR = '/content/drive/MyDrive/event/predictions'
os.makedirs(SAVE_DIR, exist_ok=True)


# --- 2. Load model ---
model = AC_MambaSeg()


class Segmentor(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model


    def forward(self, x):
        return self.model(x)


    def test_step(self, batch, batch_idx):
        image, y_true = batch
        y_pred = self.model(image)
        dice = dice_score(y_pred, y_true)
        iou = iou_score(y_pred, y_true)
        self.log_dict({"test_dice": dice, "test_iou": iou}, prog_bar=True)
        return {"y_pred": y_pred, "y_true": y_true, "image": image}


# --- 3. Load test data ---
data = np.load(DATA_PATH)
x_test, y_test = data["image"], data["mask"]
test_loader = DataLoader(ISICLoader(x_test, y_test, typeData="test"), batch_size=1, num_workers=2, prefetch_factor=8)


# --- 4. Load checkpoint ---
segmentor = Segmentor.load_from_checkpoint(CHECKPOINT_PATH, model=model)
segmentor.eval()
segmentor.freeze()


# --- 5. Trainer ---
trainer = pl.Trainer(
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1
)


# --- 6. Run test (logs Dice & IoU) ---
trainer.test(segmentor, dataloaders=test_loader)


# --- 7. Manual prediction & collection for saving/visualization ---
all_preds, all_gts, all_images = [], [], []


for batch in test_loader:
    with torch.no_grad():
        img, gt = batch
        img_device = img.cuda() if torch.cuda.is_available() else img
        pred = segmentor(img_device)
        pred_bin = (pred > 0.5).float().cpu().squeeze(0).squeeze(0).numpy()
        all_preds.append(pred_bin)
        all_gts.append(gt.squeeze(0).numpy())
        all_images.append(img.squeeze(0).cpu().numpy())


# --- 8. Save predictions to .npz ---
np.savez_compressed(
    os.path.join(SAVE_DIR, "predictions.npz"),
    preds=np.array(all_preds),
    gts=np.array(all_gts),
    images=np.array(all_images)
)
print(f"✅ Predictions saved at {SAVE_DIR}/predictions.npz")
