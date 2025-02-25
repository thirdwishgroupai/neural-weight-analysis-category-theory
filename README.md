
# HyperCategory Framework for Neural Weight Space Analysis

This project implements a novel framework that combines topological data analysis and category theory to analyze neural network weight spaces. In particular, it constructs a Vietoris–Rips complex over the weights of a designated layer, maps the resulting simplices into a learned context space using a Transformer-based encoder, and then applies a hypercategorical composition along with a contrastive regularizer. The framework not only trains a model on a given task (MNIST classification in this example) but also provides extensive visualization and textual summaries of the learned hypercategory structure.

## Table of Contents

- [Overview](#overview)
- [Theory and Motivation](#theory-and-motivation)
- [Features](#features)
- [Installation and Requirements](#installation-and-requirements)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Auto-Tuning and Hyperparameter Grid Search](#auto-tuning-and-hyperparameter-grid-search)
  - [Visualizations and Analysis](#visualizations-and-analysis)
- [Code Structure](#code-structure)
  - [SimpleCNN](#simplecnn)
  - [Differentiable Vietoris–Rips Module](#differentiable-vietoris–rips-module)
  - [Outstanding Mapping Module](#outstanding-mapping-module)
  - [HyperCategory Framework](#hypercategory-framework)
  - [Topological Regularizer](#topological-regularizer)
  - [Visualization Functions](#visualization-functions)
  - [Hypercategory Summary](#hypercategory-summary)
- [Interpreting the Results](#interpreting-the-results)
- [Future Improvements](#future-improvements)
- [License](#license)
- [Author](#author)

## Overview

The **HyperCategory Framework** aims to bridge the gap between symbolic, categorical reasoning and deep learning by providing a structured analysis of a neural network's weight space. The project:
- Builds a Vietoris–Rips complex from the weights of a selected layer.
- Maps simplices (0–, 1–, and 2–simplices) into a learned context space using a Transformer-based encoder.
- Applies a hypercategorical composition law that fuses context embeddings.
- Uses an adaptive margin-based contrastive regularizer to enforce consistency among overlapping simplices.
- Performs layer–wise analysis so you can compare the structure of low–level (conv1) and higher–level (conv2) filters.
- Provides rich visualizations (t-SNE plots, graph representations, and filter images) along with a textual summary of the hypercategory structure.
- Saves model checkpoints, visualizations, and summary data to a local directory.

## Theory and Motivation

Deep neural networks are powerful but often function as "black boxes." By analyzing the structure of the weight space—using tools from topology (Vietoris–Rips complexes) and category theory (hypercategories, functors, and morphisms)—we can gain interpretability into how networks internally organize features. In this framework:

- **Vietoris–Rips Complexes:** Capture the geometric structure of the weight space by connecting filters (or their flattened representations) that are "close" in Euclidean distance.
- **Hypercategories:** Extend standard graphs by considering higher–order relationships (pairs, triplets) between filters.
- **Outstanding Mapping:** A Transformer-based encoder maps pooled weight vectors into a rich context space.
- **Topological Regularizer:** A contrastive loss enforces that similar filters (or filter–pairs) have similar context embeddings, with an adaptive margin determined by the data.

## Features

- **Layer–Wise Analysis:** Choose which convolutional layer (e.g., conv1 or conv2) to analyze.
- **Auto-Tuning:** Automatically performs grid search over key hyperparameters (ε, base margin, and adaptive factor) to optimize the topological loss.
- **Visualization:** 
  - **t-SNE of Context Embeddings:** Projects high-dimensional context embeddings into 2D.
  - **VR Complex Graph:** Visualizes the Vietoris–Rips complex as a graph, with nodes representing filters and edges for pairs.
  - **Filter Visualizations:** Displays the raw filter weights as images.
- **Hypercategory Summary:** Generates a textual summary that enumerates the objects (0–simplices), hyperedges (1– and 2–simplices), and provides a naive composition interpretation.
- **Data Saving:** Model checkpoints, visualizations, and hypercategory summaries are saved locally for later review.

## Installation and Requirements

Ensure you have Python 3.6+ and install the required packages. You can install dependencies using pip:

```bash
pip install torch torchvision matplotlib scikit-learn networkx
```

## Usage

### Training the Model

The main training loop is included in the script. It uses the MNIST dataset for demonstration. The model is trained with both a task loss (cross-entropy) and a topological (categorical) loss.

To run training:

```bash
python your_script.py
```

The code will automatically:
- Tune the hyperparameters for the topological regularizer.
- Train the model for a specified number of epochs.
- Save model checkpoints and visualizations to the `local_results` directory.

### Auto-Tuning and Hyperparameter Grid Search

The `hyperparameter_tuning` function performs a grid search over candidate values for:
- **Epsilon (ε):** VR threshold.
- **Base Margin:** For the topological regularizer.
- **Adaptive Factor:** Scales the adaptive margin based on average pairwise distance.

The best combination is selected based on the highest average categorical loss (to encourage strong discriminative structure).

### Visualizations and Analysis

After training, the following are produced:
- **t-SNE Plot of Context Embeddings:** Saved as `layer_conv1_context_embeddings.png` (or similar for conv2).
- **VR Complex Graph:** A spring layout graph of the VR complex (0– and 1–simplices) is saved; 2–simplices are stored in JSON.
- **Filter Images:** A grid of filter images from the analyzed layer is saved.
- **Hypercategory Summary:** A textual summary is saved as `layer_conv1_hypercategory.txt` (or similar).

These outputs help you interpret which filters are similar, how they form pairs or triplets, and how the hypercategory composes complex structures.

## Code Structure

### SimpleCNN

- Implements a basic CNN for MNIST.
- Contains two convolutional layers and three fully connected layers.
- Designed to allow layer-specific analysis (via the `layer_name` argument).

### Differentiable Vietoris–Rips Module

- Computes a soft VR complex from the weight vectors.
- Generates 0–simplices, 1–simplices, and if `vr_dim >= 2`, 2–simplices.

### Outstanding Mapping Module

- Uses a Transformer encoder to map pooled weight vectors into a rich context space.
- Provides more expressive representations than a simple MLP.

### HyperCategory Framework

- Integrates the base model, VR module, and outstanding mapping.
- Constructs the hypercategory by extracting weights from a selected layer, building the VR complex, mapping simplices to context embeddings, and performing hypercomposition.
- Adjusts classifier outputs based on the aggregated context.

### Topological Regularizer

- Applies an adaptive margin-based contrastive loss to enforce that overlapping simplices have similar context embeddings.
- The adaptive margin is computed from the average pairwise distance.

### Visualization Functions

- **`visualize_context_embeddings`:** Projects context embeddings into 2D using t-SNE and saves the plot.
- **`visualize_vr_complex`:** Constructs and saves a graph of the VR complex (nodes for 0–simplices, edges for 1–simplices) and stores 2–simplices as JSON.
- **`visualize_conv_filters`:** Displays convolutional filters in a grid format.

### Hypercategory Summary

- **`print_hypercategory_structure`:** Enumerates the 0–simplices, 1–simplices, and 2–simplices from the VR complex and writes a textual summary showing potential compositions.

## Interpreting the Results

- **t-SNE Plot:**  
  Clusters in the 2D projection indicate groups of filters with similar context embeddings.
- **VR Complex Graph:**  
  The graph shows how filters (0–simplices) connect into pairs (1–simplices). The JSON file lists 2–simplices, which can be interpreted as candidate compositions.
- **Filter Visualizations:**  
  The displayed filter images help you see the raw features learned by the layer.
- **Hypercategory Summary:**  
  This textual file details the number of objects (filters), 1–simplices (pairs), 2–simplices (triplets), and provides a naive explanation of compositional overlaps.

## Future Improvements

- **Layer-Specific Adaptive Tuning:**  
  Use adaptive or multi-objective loss weighting to better balance task and topological losses for each layer.
- **Enhanced Visualizations:**  
  Consider overlaying filter images on VR graph nodes or using more advanced graph layouts.
- **Scalability Enhancements:**  
  Optimize the VR module for larger networks (using GPU-accelerated routines or sparse representations).
- **Extending to Other Datasets:**  
  Adapt the framework to other datasets (e.g., CIFAR-10, ImageNet) with appropriate hyperparameter adjustments.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Author

**Juan Zambrano**  
Third Wish Group  
Email: [juan@thirdwishgroup.com](mailto:juan@thirdwishgroup.com)
