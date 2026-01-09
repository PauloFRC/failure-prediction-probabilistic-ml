# failure-prediction-probabilistic-ml

## Dataset

The dataset is **not included** in this repository.

Please download it from the following link:

- https://zenodo.org/records/8196385/files/BGL.zip?download=1

After downloading, extract the files and place them in the `data/` directory.

## Installation

This project requires conda. Setup the environment and setup the project running:

`make`

## Project Structure

- The main notebooks (parsing.ipynb, VAE.ipynb, GP.ipynb) are in src/notebooks
- src/dataloaders has vectorization for sliding window and bert
- util has utility functions, including bayesian optimization


