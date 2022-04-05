import argparse
import torch
from pytorch_lightning.callbacks import EarlyStopping

from src.model import DiseaseClassification
import pytorch_lightning as pl

AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 256 if AVAIL_GPUS else 64


def main(args):
    model = DiseaseClassification(args.dataset_path, BATCH_SIZE)
    trainer = pl.Trainer(
        gpus=AVAIL_GPUS,
        max_epochs=1,
        # stop training based on custom metric
        callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=3)]
    )
    trainer.fit(model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default="/Users/hasnainmamdani/Downloads/ml_eng_test_task2_data/")
    args = parser.parse_args()
    main(args)
