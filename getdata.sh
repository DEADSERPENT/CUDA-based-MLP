# #!/bin/sh
# mkdir data

# wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz -P ./data/
# wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz -P ./data/
# wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz -P ./data/
# wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz -P ./data/

# gunzip ./data/train-images-idx3-ubyte.gz
# gunzip ./data/train-labels-idx1-ubyte.gz
# gunzip ./data/t10k-images-idx3-ubyte.gz
# gunzip ./data/t10k-labels-idx1-ubyte.gz


#!/bin/sh
# Create a data directory if it doesn't exist
mkdir -p data

# Base URL for the new dataset location
BASE_URL="http://ossci-datasets.s3.amazonaws.com/mnist/"

# Download the files
wget ${BASE_URL}train-images-idx3-ubyte.gz -P ./data/
wget ${BASE_URL}train-labels-idx1-ubyte.gz -P ./data/
wget ${BASE_URL}t10k-images-idx3-ubyte.gz -P ./data/
wget ${BASE_URL}t10k-labels-idx1-ubyte.gz -P ./data/

# Unzip the files
gunzip ./data/*.gz