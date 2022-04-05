This is the deliverable of Task 2 that implements a fully functional ML package around a multi-task prediction task.

The goal is to build a model that can predict probability of certain diseases from a genome sequence.
The dataset consists of three different types of genome sequences corresponding to three different species – human, duck, and snake. For each species, there is a different set of diseases that they could have – i.e. for each species, a set of probabilities for all possible diseases an organism of that species could have is predicted.

A multi-task prediction model is achieved by using a neural network model with multiple predictor heads.

### Instructions

1. Create a source environment, and install the required dependencies as specified in requirements.txt.  This project is tested on python 3.8.
2. Specify the path of dataset in arguments (see run.py)
3. Execute `python -m src.run` at the root. Model will start training for one epoch and the state will be stored automatically by the PyTorchLightning framework in the folder `lightning_logs` under root after the training epoch ends.
4. Test cases are under tests/ and can be run with Pytest at the root.
