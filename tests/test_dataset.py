from src.dataset import SPECIES_DUCK, SPECIES_SNAKE, SPECIES_HUMAN, DUCK_NUM_DISEASE, GENE_DIMS, SNAKE_NUM_DISEASE, \
    HUMAN_NUM_DISEASE, SpeciesDataset


class TestDataset(object):

    #  Some data for testing is created in the path "test/test_data/". It's basically (first) 10 samples of data of
    #  each species.
    def test_duck_data_correctly_loaded(self):
        dataset = SpeciesDataset(path="tests/test_data/", species=SPECIES_DUCK)
        assert len(dataset) == 10
        item = next(iter(dataset))
        assert item[0].shape == GENE_DIMS
        assert item[1].shape == (DUCK_NUM_DISEASE,)

    def test_human_data_correctly_loaded(self):
        dataset = SpeciesDataset(path="tests/test_data/", species=SPECIES_HUMAN)
        assert len(dataset) == 10
        item = next(iter(dataset))
        assert item[0].shape == GENE_DIMS
        assert item[1].shape == (HUMAN_NUM_DISEASE,)

    def test_snake_data_correctly_loaded(self):
        dataset = SpeciesDataset(path="tests/test_data/", species=SPECIES_SNAKE)
        assert len(dataset) == 10
        item = next(iter(dataset))
        assert item[0].shape == GENE_DIMS
        assert item[1].shape == (SNAKE_NUM_DISEASE,)
