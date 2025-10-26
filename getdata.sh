#!/bin/sh
# Script to download MNIST dataset

# Create data directory if it doesn't exist
mkdir -p data

# Base URL for MNIST dataset (AWS S3 mirror)
BASE_URL="http://ossci-datasets.s3.amazonaws.com/mnist/"

# Download the MNIST files
echo "Downloading MNIST dataset..."
wget ${BASE_URL}train-images-idx3-ubyte.gz -P ./data/
wget ${BASE_URL}train-labels-idx1-ubyte.gz -P ./data/
wget ${BASE_URL}t10k-images-idx3-ubyte.gz -P ./data/
wget ${BASE_URL}t10k-labels-idx1-ubyte.gz -P ./data/

# Unzip the files
echo "Extracting files..."
gunzip ./data/*.gz

echo "MNIST dataset downloaded successfully!"