import os
import sys
import numpy as np
import torch
import pytorch_lightning as pl
from dataset import ISICLoader
from metrics import iou_score, dice_score, dice_tversky_loss
from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import DataLoader
from models.AC_MambaSeg import AC_MambaSeg



DO_CV = True  #change à False si tu veux revenir au training normal



IN_COLAB = 'google.colab' in sys.modules


if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    SAVE_DIR = ""
else:
    SAVE_DIR = ""



os.makedirs(SAVE_DIR, exist_ok=True)


DATA_PATH = ''  
print(f"Chargement des données depuis: {DATA_PATH}")
data = np.load(DATA_PATH)
x, y = data["images"], data["masks"]
print(f"Dataset complet: x={x.shape}, y={y.shape}")


x_trainval, x_test, y_trainval, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, shuffle=True
)
print(f"TrainVal: {x_trainval.shape} | Test hold-out: {x_test.shape}")



test_save_path = os.path.join(SAVE_DIR, "test_data.npz")
np.savez(test_save_path, image=x_test, mask=y_test)
print(f" Test data saved to {test_save_path}")



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
        self.log_dict({"loss": loss, "train_dice": dice, "train_iou": iou}, prog_bar=True)
        return loss


    def validation_step(self, batch, batch_idx):
        loss, dice, iou = self._step(batch)
        self.log_dict({"val_loss": loss, "val_dice": dice, "val_iou": iou}, prog_bar=True)
        return {"val_loss": loss, "val_dice": dice, "val_iou": iou}


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5, verbose=True
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_dice"}





if DO_CV:
    print("\n============================")
    print("Lancement Cross-Validation K=5")
    print("============================\n")


    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_scores = []


    for fold, (train_idx, val_idx) in enumerate(kf.split(x_trainval), start=1):
        print(f"\n--- Fold {fold}/5 ---")
        fold_dir = os.path.join(SAVE_DIR, f"fold{fold}")
        os.makedirs(fold_dir, exist_ok=True)


        
        x_train = x_trainval[train_idx]
        y_train = y_trainval[train_idx]
        x_val = x_trainval[val_idx]
        y_val = y_trainval[val_idx]
        print(f"Fold {fold}: train={x_train.shape}, val={x_val.shape}")


   
        train_loader = DataLoader(
            ISICLoader(x_train, y_train),
            batch_size=4,
            pin_memory=True,
            shuffle=True,
            num_workers=2,
            drop_last=True,
            prefetch_factor=8,
        )
        val_loader = DataLoader(
            ISICLoader(x_val, y_val, typeData="val"),
            batch_size=1,
            num_workers=2,
            prefetch_factor=16,
        )


       
        model = AC_MambaSeg().cuda()
        segmentor = Segmentor(model)


        checkpoint_cb = pl.callbacks.ModelCheckpoint(
            dirpath=fold_dir,
            filename=f"fold{fold}-ckpt-{{val_dice:.4f}}",
            monitor="val_dice",
            mode="max",
            save_top_k=1,
            verbose=True,
            save_weights_only=False,
        )
        progress_bar = pl.callbacks.TQDMProgressBar()


       
        trainer = pl.Trainer(
            benchmark=True,
            max_epochs=20,
            precision=16,
            enable_progress_bar=True,
            callbacks=[checkpoint_cb, progress_bar],
            log_every_n_steps=1,
            num_sanity_val_steps=0,
            default_root_dir=fold_dir,
        )


        trainer.fit(segmentor, train_loader, val_loader)


       
        metrics = trainer.validate(segmentor, val_loader, verbose=False)
        best_dice = None
        if checkpoint_cb.best_model_score is not None:
            try:
                best_dice = float(checkpoint_cb.best_model_score.cpu().item())
            except Exception:
                best_dice = float(checkpoint_cb.best_model_score)


        if best_dice is None and len(metrics) > 0 and "val_dice" in metrics[0]:
            best_dice = float(metrics[0]["val_dice"])


        fold_scores.append(best_dice)
        print(f"Fold {fold} terminé. Best Dice = {best_dice}")


 
    valid_scores = [s for s in fold_scores if s is not None]
    if len(valid_scores) > 0:
        mean_dice = float(np.mean(valid_scores))
        std_dice = float(np.std(valid_scores))
        print("\n=== Résultats Cross-Validation ===")
        for i, s in enumerate(fold_scores, start=1):
            print(f"Fold {i}: Dice={s}")
        print(f"Moyenne Dice: {mean_dice:.4f} | Std: {std_dice:.4f}")
    else:
        print("\n Aucun score de fold valide récupéré.")


    
    sys.exit(0)





x_train, x_val, y_train, y_val = train_test_split(
    x_trainval, y_trainval, test_size=0.1, random_state=42, shuffle=True
)



model = AC_MambaSeg().cuda()
train_loader = DataLoader(
    ISICLoader(x_train, y_train),
    batch_size=4,
    pin_memory=True,
    shuffle=True,
    num_workers=2,
    drop_last=True,
    prefetch_factor=8,
)
val_loader = DataLoader(
    ISICLoader(x_val, y_val, typeData="val"),
    batch_size=1,
    num_workers=2,
    prefetch_factor=16,
)


test_loader = DataLoader(
    ISICLoader(x_test, y_test, typeData="test"),
    batch_size=1,
    num_workers=2,
)



checkpoint_cb = pl.callbacks.ModelCheckpoint(
    dirpath=SAVE_DIR,
    filename="ckpt-{val_dice:.4f}",
    monitor="val_dice",
    mode="max",
    save_top_k=1,
    verbose=True,
    save_weights_only=False,
)


progress_bar = pl.callbacks.TQDMProgressBar()



trainer = pl.Trainer(
    benchmark=True,
    max_epochs=20,
    precision=16,
    enable_progress_bar=True,
    callbacks=[checkpoint_cb, progress_bar],
    log_every_n_steps=1,
    num_sanity_val_steps=0,
)



segmentor = Segmentor(model)
trainer.fit(segmentor, train_loader, val_loader)


print("\n=== Test hold-out ===")
trainer.test(segmentor, dataloaders=test_loader, verbose=True)


final_model_path = os.path.join(SAVE_DIR, "final_model.ckpt")
trainer.save_checkpoint(final_model_path)
print(f" Full model saved to {final_model_path}")

