#!/usr/bin/env python3
"""
Generate test input for CUDA inference from MNIST test dataset.
This script reads MNIST test images and outputs them in a format suitable
for the infer binary (784 space-separated normalized floats).
"""

import struct
import sys
import os

def read_mnist_images(filename):
    """Read MNIST image file and return images as list of lists."""
    with open(filename, 'rb') as f:
        # Read header
        magic = struct.unpack('>I', f.read(4))[0]
        num_images = struct.unpack('>I', f.read(4))[0]
        num_rows = struct.unpack('>I', f.read(4))[0]
        num_cols = struct.unpack('>I', f.read(4))[0]

        if magic != 2051:
            raise ValueError(f"Invalid magic number: {magic}")

        # Read images
        images = []
        for _ in range(num_images):
            image = []
            for _ in range(num_rows * num_cols):
                pixel = struct.unpack('B', f.read(1))[0]
                # Normalize to [0, 1]
                image.append(pixel / 255.0)
            images.append(image)

        return images

def read_mnist_labels(filename):
    """Read MNIST label file and return labels as list."""
    with open(filename, 'rb') as f:
        # Read header
        magic = struct.unpack('>I', f.read(4))[0]
        num_labels = struct.unpack('>I', f.read(4))[0]

        if magic != 2049:
            raise ValueError(f"Invalid magic number: {magic}")

        # Read labels
        labels = []
        for _ in range(num_labels):
            label = struct.unpack('B', f.read(1))[0]
            labels.append(label)

        return labels

def main():
    # Parse command-line arguments
    if len(sys.argv) < 2:
        print("Usage: python3 generate_test_input.py <image_index> [options]")
        print("  <image_index>  : Index of test image to use (0-9999)")
        print("  -v, --verbose  : Print image visualization and label")
        sys.exit(1)

    try:
        index = int(sys.argv[1])
    except ValueError:
        print(f"Error: Invalid index '{sys.argv[1]}'")
        sys.exit(1)

    verbose = '-v' in sys.argv or '--verbose' in sys.argv

    # Check for data files
    image_file = './data/t10k-images-idx3-ubyte'
    label_file = './data/t10k-labels-idx1-ubyte'

    if not os.path.exists(image_file):
        print(f"Error: {image_file} not found", file=sys.stderr)
        print("Run 'bash getdata.sh' to download MNIST data", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(label_file):
        print(f"Error: {label_file} not found", file=sys.stderr)
        print("Run 'bash getdata.sh' to download MNIST data", file=sys.stderr)
        sys.exit(1)

    # Read MNIST data
    images = read_mnist_images(image_file)
    labels = read_mnist_labels(label_file)

    if index < 0 or index >= len(images):
        print(f"Error: Index {index} out of range (0-{len(images)-1})", file=sys.stderr)
        sys.exit(1)

    # Get requested image and label
    image = images[index]
    label = labels[index]

    if verbose:
        print(f"Test image {index}, True label: {label}", file=sys.stderr)
        print("", file=sys.stderr)

        # Visualize the image
        for i in range(28):
            row = image[i*28:(i+1)*28]
            line = ""
            for pixel in row:
                if pixel > 0.5:
                    line += "██"
                elif pixel > 0.25:
                    line += "▓▓"
                elif pixel > 0.1:
                    line += "░░"
                else:
                    line += "  "
            print(line, file=sys.stderr)
        print("", file=sys.stderr)

    # Output the image as space-separated floats
    print(' '.join(f'{pixel:.6f}' for pixel in image))

if __name__ == '__main__':
    main()
