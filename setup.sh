#!/bin/bash

# This script sets up the conda environment for GraphWaveNet

# Name of the conda environment
ENV_NAME=graph_wavenet

# Create a new conda environment with Python 3.8
echo "Creating a new conda environment named $ENV_NAME with Python 3.8"
conda create --name $ENV_NAME python=3.8 -y

# Activate the conda environment
echo "Activating the conda environment: $ENV_NAME"
source activate $ENV_NAME

# Install the required dependencies from requirements.txt
echo "Installing dependencies..."
while read requirement; do conda install --yes $requirement || pip install $requirement; done < requirements.txt

echo "Setup is complete. The conda environment '$ENV_NAME' is ready to use."
echo "To activate the conda environment, run 'conda activate $ENV_NAME'."
