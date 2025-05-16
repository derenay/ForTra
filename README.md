# Automatic Tactical Formation Recognition

This project implements a deep learning model, `HierarchicalFormationTransformer`, to automatically classify tactical military unit formations (e.g., Line, Wedge, Column) based on the spatial coordinates and directional orientations of individual units. The model is designed to be invariant to the input order of units and leverages geometric biases within a Transformer architecture.

## Overview

The system takes data representing a group of units (e.g., tanks), each defined by its 2D coordinates, class type (though current examples primarily use 'tank'), and facing direction. It then outputs a predicted formation label for the entire group. The project includes scripts for synthetic data generation, model training, evaluation, and a simple Flask API server for inference.

## Key Features

* **Formation Classification:** Identifies 8 distinct tactical formations (Line, Wedge, Vee, Herringbone, Coil, Staggered Column, Column, Echelon).
* **Custom Transformer Model (`HierarchicalFormationTransformer`):**
    * Uses a multi-stage Transformer encoder architecture.
    * **Order Invariant:** Does *not* use absolute positional encoding, treating unit inputs as an unordered set.
    * **Geometric Biases:** Incorporates spatial bias (based on pairwise distances) and directional bias (based on pairwise relative orientations) directly into the self-attention mechanism to provide geometric context.
    * **Feature Fusion:** Combines embeddings from unit coordinates, class types, directions, and relative coordinates (to formation centroid).
    * **Attention Pooling:** Aggregates final unit representations into a single formation vector for classification.
    * Optional adapter modules.
* **Synthetic Data Generation:** Includes Python scripts to generate diverse training and validation data for various formations with controllable parameters (unit count, spacing, orientation, noise levels - though the current preference is for noise-free generation).
* **Training and Evaluation Scripts:**
    * `train.py`: For training the model, with internal configuration for hyperparameters, data paths, and saving the best model based on validation loss.
    * `val.py`: For evaluating a trained model using standard metrics (accuracy, classification report, confusion matrix).
* **Inference API:**
    * `server.py`: A Flask-based web server to load a trained model and provide predictions via a POST request to a `/predict` endpoint.

## Motivation

Automatic recognition of tactical formations is crucial for enhancing situational awareness in military operations, improving the realism of training simulations, enabling more sophisticated post-mission analysis, and potentially supporting decision-making in command and control (C2) systems. This project aims to develop a robust AI solution for this challenging geometric pattern recognition task.

## Model Architecture Highlights

The `HierarchicalFormationTransformer` processes a set of units as follows:

1.  **Input Embeddings:** Each unit's coordinates, class, direction, and coordinates relative to the formation's centroid are embedded.
2.  **Feature Fusion:** These embeddings are concatenated for each unit.
3.  **No Positional Encoding:** The fused embeddings are directly passed to the Transformer stages without absolute positional encoding to ensure order invariance.
4.  **Geometric Biases:**
    * **Spatial Bias:** Calculated as the negative Euclidean distance (`-dist`) between unit pairs.
    * **Directional Bias:** Calculated as the cosine of the relative angle difference (`cos(angle_diff)`) between unit pairs, after scaling input directions (0-1 normalized angles) to radians. The tensor shape is adjusted with `.squeeze(-1)` after this calculation.
5.  **Transformer Stages:** The data passes through multiple stages of `GraphormerTransformerLayer`s. Each layer uses `GraphormerMultiheadAttention`, where the spatial and directional biases are added to the attention scores.
6.  **Attention Pooling:** The final representations of all units are aggregated into a single vector using an attention-based pooling mechanism.
7.  **Classification:** This summary vector is fed to a final linear layer to predict formation probabilities.

## Technologies Used

* **Language:** Python 3.x
* **Core Libraries:**
    * PyTorch (for the deep learning model and training)
    * NumPy (for numerical operations, especially in data generation)
    * scikit-learn (for data splitting and evaluation metrics)
* **Data Handling:** `json` (for data loading/saving)
* **Visualization (Development):** Matplotlib
* **Web Server (Optional):** Flask (for `server.py`)
* **Version Control:** Git (recommended)


## Setup and Installation

1.  **Clone the repository (if applicable).**
2.  **Create a Python virtual environment (recommended):**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
3.  **Install dependencies:**
    Create a `requirements.txt` file with the following content (adjust versions as needed):
    ```
    torch
    numpy
    scikit-learn
    matplotlib
    flask
    flask-cors
    # pandas (if you use it for data exploration/loading elsewhere)
    ```
    Then run:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Ensure CUDA is set up correctly if using a GPU.**

## Data

* Data is expected in JSON format, as a list of dictionaries. Each dictionary represents a formation sample and should contain:
    * `"coordinates"`: A list of `[x, y]` coordinate pairs for each unit.
    * `"classes"`: A list of string class labels for each unit (e.g., `"tank"`).
    * `"directions"`: A list of floats (0-1, representing normalized angles) for each unit's direction.
    * `"formation"`: A string label for the ground-truth formation type.
* Synthetic data can be generated using the `generate_*.py` scripts. Modify parameters within these scripts as needed.
* Place your main training data (e.g., combined generated data) in `dataset/data.json` and validation data in `dataset/val.json`, or update the paths in the `config` dictionary within `train.py` and `val.py`.
* The `dataset.py` script handles:
    * Mapping string class/formation labels to integer indices.
    * (It was previously discussed to include canonical ordering here; if you are using a model *with* positional encoding, ensure canonical ordering is active. For the current PE-less model, this sorting by `dataset.py` provides consistency but isn't directly leveraged by PE.)
    * The `collate_fn` handles batching and padding.

## Training the Model

1.  **Configure Training:** Open `train.py`. All training and model hyperparameters are defined within the `config` dictionary in the `if __name__ == "__main__":` block. Adjust parameters like:
    * `data_file`, `save_dir`
    * Model parameters (`class_embed_dim`, `direction_dim`, `stage_dims`, `num_heads`, `num_layers`, `dropout_stages`, `max_len`, bias parameters if using learnable biases, etc.) to match your desired architecture (e.g., the "balanced" config or the one you found works "perfectly").
    * Training parameters (`epochs`, `batch_size`, `lr`, etc.).
2.  **Run Training:**
    ```bash
    python train.py
    ```
3.  The script will create a run directory under `trained_models/` (e.g., `trained_models/hft_balanced_run_YYYYMMDD_HHMMSS/`) and save the `best_model.pth` based on validation loss. Console output will show training progress.

## Evaluating the Model

1.  **Configure Evaluation:** Open `val.py`.
    * Update `DEFAULT_MODEL_PATH` to point to your trained `best_model.pth` file.
    * Update `DEFAULT_VAL_DATA_PATH` to the dataset you want to evaluate on.
    * Ensure `MODEL_CONFIG` matches the architecture of the loaded model *exactly*.
    * Ensure `CLASS_TO_IDX` and `FORMATION_TO_IDX` are consistent with your training and dataset.
2.  **Run Evaluation:**
    ```bash
    python val.py
    ```
3.  The script will output overall accuracy, a classification report, and a confusion matrix.

## Running the Inference API (Flask Server)

1.  **Configure Server:** Open `server.py`.
    * Update `DEFAULT_MODEL_PATH` to point to your trained `best_model.pth`.
    * Ensure `MODEL_CONFIG`, `CLASS_TO_IDX`, and `FORMATION_TO_IDX` are consistent.
    * Adjust CORS settings if your frontend is served from a different origin/port. If serving frontend and backend from the same Flask app, CORS might not be strictly needed.
2.  **Run Server:**
    ```bash
    python server.py
    ```
    The server will typically start on `http://127.0.0.1:5000/`.
3.  **Make Predictions:** Send a POST request to the `/predict` endpoint with a JSON payload containing `coordinates`, `classes`, and `directions` for a single formation.
    Example payload:
    ```json
    {
        "coordinates": [[0.5, 0.5], [0.5, 0.6], [0.5, 0.7]],
        "classes": ["tank", "tank", "tank"],
        "directions": [0.25, 0.25, 0.25]
    }
    ```
    The server will respond with the predicted formation name.

## Results (Example)

The model (without positional encoding, using fixed geometric biases) has shown promising results, achieving approximately **76-77% accuracy** on a validation set. It successfully learned to distinguish challenging cases like "Line" vs. "Column" and performed well on "Vee" and "Coil". Challenges remain in robustly distinguishing highly similar formations like "Staggered Column" from "Column" and "Wedge" from "Echelon" under all variations.

*(You should update this section with your latest/best findings and specific metrics for different configurations if you have them.)*

## Future Work & Improvements

* **Learnable Geometric Biases:** Fully implement and tune learnable biases for distances and directions (using binning and `nn.Embedding`) to allow the model more flexibility in learning geometric relationships.
* **Advanced Data Augmentation:** Implement more sophisticated data augmentation techniques during training (rotations, scaling, slight non-uniform deformations, more varied noise models).
* **Feature Engineering:** Explore adding explicit features that highlight subtle differences (e.g., metrics for "staggeredness," unit direction relative to the formation's principal axis).
* **Hyperparameter Optimization:** Conduct a more systematic search for optimal hyperparameters (model dimensions, layer counts, dropout, learning rate, bias bin counts).
* **Robustness Testing:** Evaluate against more diverse and challenging datasets, including data with higher noise levels or occlusions.
* **Comparison with GNNs:** Implement and compare against Graph Neural Network architectures designed for similar tasks.
* **Explainability (XAI):** Integrate methods to understand and visualize what features or unit relationships the model is focusing on for its predictions.

---
