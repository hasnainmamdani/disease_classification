import torch
import pytorch_lightning as pl
from torch import nn, Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from src.dataset import SpeciesDataset, SUPPORTED_SPECIES, SPECIES_DUCK, SPECIES_SNAKE, SPECIES_HUMAN, DUCK_NUM_DISEASE, \
    HUMAN_NUM_DISEASE, SNAKE_NUM_DISEASE, GENE_DIMS
from pytorch_lightning.trainer.supporters import CombinedLoader
from src.merics import CustomMetric


class DiseaseClassification(pl.LightningModule):
    """
    Implementation of the disease classification task for duck, human and snake species
    Args:
        data_dir (string): path where dataset is stored.
        batch_size (int): batch size for model training.
        hidden_size (int): Number of units in the hidden layers of the neural network.
        learning_rate (float): learning rate for optimizer.
    """
    def __init__(self, data_dir: str = None, batch_size: int = 64, hidden_size: int = 512, learning_rate: float = 2e-4):
        super().__init__()

        # Set our init args as class attributes
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        self.duck_num_classes = DUCK_NUM_DISEASE
        self.human_num_classes = HUMAN_NUM_DISEASE
        self.snake_num_classes = SNAKE_NUM_DISEASE
        self.dims = GENE_DIMS

        # a simple MLP
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.dims[0] * self.dims[1], hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # create separate predictor heads for each species
        self.head_duck = nn.Linear(hidden_size, self.duck_num_classes)
        self.head_human = nn.Linear(hidden_size, self.human_num_classes)
        self.head_snake = nn.Linear(hidden_size, self.snake_num_classes)
        self.sigmoid = nn.Sigmoid()

        self.valid_loss = CustomMetric()

    def forward(self, x: Tensor, species: str):
        assert species in SUPPORTED_SPECIES, "Invalid species. Must be in: " + str(SUPPORTED_SPECIES)
        x = self.model(x)

        # choose the last layer based on the species
        if species == SPECIES_DUCK:
            x = self.head_duck(x)
        elif species == SPECIES_HUMAN:
            x = self.head_human(x)
        else:  # snake
            x = self.head_snake(x)

        x = self.sigmoid(x)
        return x

    def training_step(self, batch, batch_idx):
        batch_duck = batch["duck_train_dl"]
        batch_human = batch["human_train_dl"]
        batch_snake = batch["snake_train_dl"]

        X_duck, Y_duck = batch_duck
        X_human, Y_human = batch_human
        X_snake, Y_snake = batch_snake

        preds_duck = self(X_duck, SPECIES_DUCK)
        preds_human = self(X_human, SPECIES_HUMAN)
        preds_snake = self(X_snake, SPECIES_SNAKE)

        loss_duck = F.binary_cross_entropy(preds_duck, Y_duck)
        loss_human = F.binary_cross_entropy(preds_human, Y_human)
        loss_snake = F.binary_cross_entropy(preds_snake, Y_snake)

        loss = loss_duck + loss_human + loss_snake

        self.log("train_loss", loss)

        return loss

        # TODO: At the moment, batches are alternating between the duck, human, and snake datasets but backprop is being
        # done on a batch of combination of all of them at once. Discuss how best to backprop one species at a time (as
        # suggested in the specs) with the PytorchLightning API. It's more straightforward to do it with just PyTorch
        # as the dataset iterator and backprops logic are well exposed.
        # TBD: the iteration is stopped after the longest loader (the one with most batches) is done,
        # while cycling through the shorter loaders.
        # TBD: hyperparameter tuning, doesn't seem required in this task

    def evaluate(self, batch, stage=None):
        batch_duck = batch["duck_" + stage + "_dl"]
        batch_human = batch["human_" + stage + "_dl"]
        batch_snake = batch["snake_" + stage + "_dl"]

        X_duck, Y_duck = batch_duck
        X_human, Y_human = batch_human
        X_snake, Y_snake = batch_snake

        preds_duck = self(X_duck, SPECIES_DUCK)
        preds_human = self(X_human, SPECIES_HUMAN)
        preds_snake = self(X_snake, SPECIES_SNAKE)

        self.valid_loss(torch.cat((preds_duck, preds_human, preds_snake), 1), torch.cat((Y_duck, Y_human, Y_snake), 1).long())

        if stage:
            self.log(f"{stage}_loss", self.valid_loss, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "tests")  # seems not required in this assignment

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    ####################
    # DATA RELATED HOOKS
    ####################

    def prepare_data(self):
        self.duck_dataset = SpeciesDataset(self.data_dir, SPECIES_DUCK)
        self.human_dataset = SpeciesDataset(self.data_dir, SPECIES_HUMAN)
        self.snake_dataset = SpeciesDataset(self.data_dir, SPECIES_SNAKE)

    def setup(self, stage=None):
        self.duck_train, self.duck_val, self.duck_test = random_split(self.duck_dataset, [14000, 3000, 3000])
        self.human_train, self.human_val, self.human_test = random_split(self.human_dataset, [4000, 3000, 3000])
        self.snake_train, self.snake_val, self.snake_test = random_split(self.snake_dataset, [44000, 3000, 3000])

    def train_dataloader(self):
        duck_train_dl = DataLoader(self.duck_train, batch_size=self.batch_size)
        human_train_dl = DataLoader(self.human_train, batch_size=self.batch_size)
        snake_train_dl = DataLoader(self.snake_train, batch_size=self.batch_size)

        return {"duck_train_dl": duck_train_dl, "human_train_dl": human_train_dl, "snake_train_dl": snake_train_dl}

    def val_dataloader(self):
        duck_val_dl = DataLoader(self.duck_val, batch_size=self.batch_size)
        human_val_dl = DataLoader(self.human_val, batch_size=self.batch_size)
        snake_val_dl = DataLoader(self.snake_val, batch_size=self.batch_size)

        return CombinedLoader({"duck_val_dl": duck_val_dl, "human_val_dl": human_val_dl, "snake_val_dl": snake_val_dl})

    def test_dataloader(self):
        duck_test_dl = DataLoader(self.duck_test, batch_size=self.batch_size)
        human_test_dl = DataLoader(self.human_test, batch_size=self.batch_size)
        snake_test_dl = DataLoader(self.snake_test, batch_size=self.batch_size)

        return CombinedLoader({"duck_test_dl": duck_test_dl, "human_test_dl": human_test_dl, "snake_test_dl": snake_test_dl})
