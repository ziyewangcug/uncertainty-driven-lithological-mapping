Uncertainty-driven lithological mapping: from data uncertainty to model uncertainty (UQDL v1.0)

Overview
Traditional lithological mapping with remote sensing data and deep learning models often ignores the sources of uncertainty arising from:
Data uncertainty — due to noise, limited observation, sampling issues, and measurement errors.
Model uncertainty — due to stochasticity in model parameters, architectures, or optimization.

UQDL v1.0 investigates uncertainty quantification in lithological mapping using remote sensing data and a Bayesian U-Net model:
(1) Data uncertainty quantification via stochastic simulation of probabilistic labels and noisy input images.
(2) Model uncertainty quantification via a Bayesian U-Net fully convolutional neural network (U-Net FCN) incorporating Monte Carlo Dropout.
(3) Visualizing uncertainties using SHAP technology to enhance model explainability.

Installation
Python ≥ 3.7
TensorFlow ≥ 2.1

Key Scripts
1. unet_data_uncertainty_quantification.py
Quantifying data uncertainty through simulating noisy inputs and probabilistic labels.
Main features:
Generates multiple soft label realizations (generate_soft_labels()).
Trains U-Net models across NUM_ITERATIONS simulations.
Injects Gaussian noise to evaluate prediction stability.
Outputs:
soft_labels_collection.mat
uncertainty_predictions_cube.mat — a cube of predictions for uncertainty estimation.
training_log.csv — per-iteration training metrics.

2. unet_model_uncertainty_quantification.py
Quantifies model uncertainty using Monte Carlo dropout.
Main features:
Builds a dropout-regularized U-Net (build_model()).
Performs Monte Carlo sampling during inference (evaluate_model_mc_dropout()).
Outputs:
mc_pred_map.mat — multiple Monte Carlo-sampled predictions.
Training logs (training_metrics.csv)— per-iteration training metrics
Visualization of mean prediction and uncertainty map.

Case study data
14-bands ASTER data in VNIR-SWIR-TIR regions. 
image_train.tif — training image.
class_train.tif — training label.
image_test.tif — testing image.

Author & Contact
Author: Ziye Wang
Affiliation: China University of Geosciences
Contact: ziyewang@cug.edu.cn