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

## Overview

The **HyperCategory Framework** aims to bridge the gap between symbolic, categorical reasoning and deep learning by providing a structured analysis of a neural network's weight space. The project:
- Builds a Vietoris–Rips complex from the weights of a selected layer.
- Maps simplices (0–, 1–, and 2–simplices) into a learned context space using a Transformer-based architecture.
- Applies a hypercategorical composition law that fuses context embeddings.
- Uses an adaptive margin-based contrastive regularizer to enforce consistency among overlapping simplices.
- Performs layer–wise analysis so you can compare the structure of low–level (conv1) and higher–level (conv2) filters.
- Provides rich visualizations (t-SNE plots, graph representations, and filter images) along with a textual summary of the hypercategory structure.

## Theory and Motivation

Deep neural networks are powerful but often function as "black boxes." By analyzing the structure of the weight space—using tools from topology (Vietoris–Rips complexes) and category theory (hypercategories, functors, and morphisms)—we can gain interpretability into how networks internally organize features. In this framework:

- **Vietoris–Rips Complexes:** Capture the geometric structure of the weight space by connecting filters (or their flattened representations) that are "close" in Euclidean distance.
- **Hypercategories:** Extend standard graphs by considering higher–order relationships (pairs, triplets) between filters.
- **Outstanding Mapping:** A Transformer–based encoder maps pooled weight vectors into a rich context space.
- **Topological Regularizer:** A contrastive loss enforces that similar filters (or filter combinations) have similar context embeddings, with an adaptive margin determined by the data.

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
