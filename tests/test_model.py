import pytest
import torch
from src.dataset import SPECIES_DUCK, SPECIES_SNAKE, SPECIES_HUMAN, DUCK_NUM_DISEASE, GENE_DIMS, SNAKE_NUM_DISEASE, HUMAN_NUM_DISEASE
from src.model import DiseaseClassification


# Testing some basics of the model, mostly the 'pre-train' aspects (https://www.jeremyjordan.me/testing-ml/)
class TestModel(object):

    def setup_method(self):
        """Called before every method to setup any state."""
        self.model = DiseaseClassification()
        self.batch_size = 5
        self.x = torch.randint(0, 1+1, size=(self.batch_size, GENE_DIMS[0], GENE_DIMS[1]), dtype=torch.float32)

    def teardown_method(self):
        """Called after every method to teardown any state."""
        del self.model
        del self.batch_size
        del self.x

    def test_correct_predictor_head_used(self):
        """Check if the output layer is applied in accordance to species type."""

        x_ = self.model.forward(self.x, SPECIES_DUCK)
        assert x_.shape == (self.batch_size, DUCK_NUM_DISEASE)

        x_ = self.model.forward(self.x, SPECIES_HUMAN)
        assert x_.shape == (self.batch_size, HUMAN_NUM_DISEASE)

        x_ = self.model.forward(self.x, SPECIES_SNAKE)
        assert x_.shape == (self.batch_size, SNAKE_NUM_DISEASE)

    def test_species_validity(self):
        with pytest.raises(AssertionError):
            self.model.forward(self.x, "cat")  # cat is invalid

    def test_output_range(self):
        """ mode outputs should be in [0, 1] """
        x_ = self.model.forward(self.x, SPECIES_DUCK)
        assert (x_ <= 1).all() and (x_ >= 0).all()

        x_ = self.model.forward(self.x, SPECIES_HUMAN)
        assert (x_ <= 1).all() and (x_ >= 0).all()

        x_ = self.model.forward(self.x, SPECIES_SNAKE)
        assert (x_ <= 1).all() and (x_ >= 0).all()
