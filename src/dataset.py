import numpy as np
from torch.utils.data import Dataset

# This could be programmatically loaded from the provided json, but since there are only 3 species to focus at the moment,
# it's good to have a clear visual of the species in the code. Similar idea applied in model.py.
SPECIES_DUCK = "duck"
SPECIES_HUMAN = "human"
SPECIES_SNAKE = "snake"
SUPPORTED_SPECIES = [SPECIES_DUCK, SPECIES_HUMAN, SPECIES_SNAKE]

# number of diseases for each specie and input gene matrix shape
DUCK_NUM_DISEASE = 20
HUMAN_NUM_DISEASE = 10
SNAKE_NUM_DISEASE = 5
GENE_DIMS = (1000, 4)


class SpeciesDataset(Dataset):
    """
    A generic Dataset to load the genome sequences and the corresponding diseases of each species.

    Args:
        path:
            Path of the dataset.
        species:
            Species name to load the corresponding dataset. Supported names : ['duck', 'human', 'snake']
    """

    def __init__(self, path: str, species: str):

        print(path)
        assert species in SUPPORTED_SPECIES, "Invalid species. Must be in: " + str(SUPPORTED_SPECIES)
        self.X = np.load(path + species + "_input_data.npy").astype('float32')
        self.Y = np.load(path + species + "_target_data.npy").astype('float32')

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return [self.X[idx], self.Y[idx]]
