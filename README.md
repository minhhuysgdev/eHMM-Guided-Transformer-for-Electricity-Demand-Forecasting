# Methodology

## 1. Overview

This study proposes a regime-guided forecasting framework for short-term electricity demand prediction. The main objective is to forecast future electricity consumption over a fixed prediction horizon using historical demand, weather variables, calendar features, and latent load-regime information.

The proposed framework combines an Ensemble Hidden Markov Model (e-HMM) with a Transformer encoder. The e-HMM is not used as a direct forecasting model. Instead, it is used as an unsupervised regime discovery module that estimates the probability of each latent demand regime at every time step. These regime probabilities are then concatenated with engineered time-series features and passed to a Transformer forecasting model.

In summary, the method follows the principle:

```text
e-HMM: latent load-regime discovery
Transformer: multi-step electricity demand forecasting
```

## 2. Problem Formulation

Let `y_t` denote the electricity demand at time step `t`. Given a historical input window of length `L`, the goal is to predict the next `H` future demand values:

```text
Input:  X_t = [x_{t-L+1}, x_{t-L+2}, ..., x_t]
Output: Y_t = [y_{t+1}, y_{t+2}, ..., y_{t+H}]
```

where:

```text
L = 168 hours
H = 24 hours
```

Each input vector `x_t` contains demand-related features, weather variables, calendar features, lag features, rolling statistics, and regime probabilities inferred by the e-HMM.

The forecasting task is therefore formulated as a supervised many-to-many time-series forecasting problem:

```text
f_theta(X_t) -> Y_t
```

where `f_theta` is the Transformer-based forecasting model with learnable parameters `theta`.

## 3. Dataset

The experiment uses the public electricity demand dataset:

```text
EDS-lab/electricity-demand
```

The dataset consists of three main files:

```text
demand.parquet
metadata.parquet
weather.parquet
```

The demand table contains smart-meter electricity consumption values with the columns `unique_id`, `timestamp`, and `y`. The metadata table provides meter-level information such as building type, location, and frequency. The weather table contains location-level meteorological variables such as temperature, humidity, apparent temperature, precipitation, rain, and snowfall.

The three tables are merged using:

```text
demand + metadata: unique_id
merged table + weather: location_id and timestamp
```

## 4. Data Preprocessing

### 4.1 Meter Selection

For the initial prototype, a subset of meters is selected to reduce computational cost. Meters are ranked based on data completeness, number of observations, and available duration. The MVP configuration uses one representative meter with high data availability.

### 4.2 Resampling

All demand observations are converted to an hourly frequency. If the original observations have a higher frequency, electricity demand is aggregated by summation within each hourly interval. Weather variables are aggregated using their hourly mean values.

### 4.3 Missing Value Handling

The data is sorted by `unique_id` and `timestamp`. Short gaps in electricity demand are filled using linear interpolation, while long missing intervals are removed. Missing weather values are filled using forward-fill and backward-fill operations within each meter sequence.

### 4.4 Feature Engineering

The feature engineering process constructs three groups of predictors.

First, calendar features are extracted from timestamps:

```text
hour
day_of_week
month
is_weekend
```

Cyclical encodings are used for periodic variables:

```text
hour_sin = sin(2 * pi * hour / 24)
hour_cos = cos(2 * pi * hour / 24)
dow_sin  = sin(2 * pi * day_of_week / 7)
dow_cos  = cos(2 * pi * day_of_week / 7)
```

Second, lag features are created to represent short-term and seasonal dependencies:

```text
y_lag_1
y_lag_24
y_lag_168
```

Third, rolling statistics are computed to summarize local and weekly demand patterns:

```text
rolling_mean_24
rolling_std_24
rolling_mean_168
rolling_std_168
```

Weather variables are also included when available:

```text
temperature_2m
relative_humidity_2m
apparent_temperature
precipitation
rain
snowfall
```

## 5. Train, Validation, and Test Split

Because the data is a time series, random splitting is not used. The dataset is split chronologically:

```text
Training set:   first 70%
Validation set: next 10%
Test set:       last 20%
```

This setup prevents future information from leaking into model training and better reflects real forecasting conditions.

## 6. Latent Regime Discovery Using e-HMM

### 6.1 HMM Input Features

The HMM module receives a smaller feature subset that reflects the latent load condition:

```text
y
y_lag_24
rolling_mean_24
rolling_std_24
temperature_2m
hour_sin
hour_cos
dow_sin
dow_cos
```

Let this HMM input vector be denoted as:

```text
z_t in R^F_hmm
```

where `F_hmm` is the number of HMM-specific input features.

### 6.2 Gaussian Hidden Markov Model

A Gaussian HMM is used to model latent demand regimes. The model assumes that each time step belongs to one hidden state:

```text
s_t in {1, 2, ..., K}
```

where `K = 3` in the MVP implementation. These states are interpreted as low-load, normal-load, and peak-load regimes after post-hoc mapping.

For each time step, the HMM estimates the posterior probability:

```text
P(s_t = k | z_t)
```

for each regime `k`.

Because HMM states are unsupervised, state indices do not have fixed semantic meanings. Therefore, states are mapped based on the mean demand associated with each state:

```text
lowest mean demand  -> low-load regime
middle mean demand  -> normal-load regime
highest mean demand -> peak-load regime
```

### 6.3 Ensemble HMM

Instead of relying on a single HMM, an ensemble of HMMs is trained to improve robustness. The ensemble consists of `M` Gaussian HMMs:

```text
M = 5
```

Each HMM is trained on bootstrap samples of contiguous time-series sub-sequences. Sampling contiguous windows preserves the temporal structure required for estimating the HMM transition matrix.

For model `m`, the posterior probability is:

```text
p_t^(m) = P_m(s_t | z_t)
```

The ensemble regime probability is computed by averaging the posterior probabilities:

```text
p_t = (1 / M) * sum_{m=1}^{M} p_t^(m)
```

The resulting regime vector is:

```text
p_t = [P_low, P_normal, P_peak]
```

These probabilities are appended to the forecasting dataset as:

```text
regime_low_prob
regime_normal_prob
regime_peak_prob
```

## 7. Regime-Guided Transformer Forecasting Model

### 7.1 Input Representation

At each time step, the base feature vector (demand, lags, rolling statistics, weather, and calendar features) is denoted as:

```text
x_t in R^F
```

The regime probability vector inferred by the e-HMM is:

```text
p_t in R^K
```

Unlike a feature-augmented approach that would simply concatenate `p_t` into the input, the proposed model keeps the two information sources separate. The Transformer encoder operates only on the base feature sequence:

```text
X_t = [x_{t-L+1}, ..., x_t]  in R^(L x F)
```

while the regime probability at the most recent observed step, `p_t in R^K`, is used to condition the encoded representation (Section 7.2). This separation ensures the regime signal acts as guidance rather than as one more correlated input feature.

### 7.2 Model Architecture

The proposed forecasting model is a regime-conditioned Transformer encoder with a multi-step regression head. The architecture consists of the following components:

```text
Base feature sequence            Regime probabilities p_t
-> Linear projection                    |
-> Positional encoding                  |
-> Transformer encoder                  |
-> Last-token pooling  --------> FiLM conditioning (gamma, beta)
-> MLP forecast head
-> 24-step demand forecast
```

The input projection maps the base feature dimension to the Transformer hidden dimension:

```text
R^F -> R^d_model
```

The positional encoding injects information about temporal order into the sequence representation. The Transformer encoder then learns dependencies across the historical window using multi-head self-attention.

After encoding the full input sequence, the representation of the last token is used as a summary of the available history:

```text
h_t = Encoder(X_t)[-1]   in R^d_model
```

### 7.3 Regime Conditioning (FiLM)

The regime probabilities modulate the encoded representation through a Feature-wise Linear Modulation (FiLM) mechanism. A small network maps the regime vector to per-channel scale and shift parameters:

```text
[gamma_t, beta_t] = MLP_regime(p_t)
```

The encoded representation is then affinely transformed:

```text
h'_t = (1 + gamma_t) * h_t + beta_t
```

The final layer of `MLP_regime` is initialized to zero, so at the start of training `gamma_t = 0` and `beta_t = 0`, making the model equivalent to an unconditioned Transformer. During training the model learns how each latent load regime should reshape the temporal representation.

The forecast head maps the conditioned representation to the prediction horizon:

```text
Y_hat_t = MLP(h'_t)   in R^24
```

This design makes the model genuinely regime-guided: the low-load, normal-load, and peak-load regimes discovered by the e-HMM directly steer the encoder output, rather than appearing only as additional input columns.

## 8. Training Objective

The model is trained using mean squared error between the predicted and true future demand values:

```text
Loss = (1 / H) * sum_{h=1}^{H} (y_{t+h} - y_hat_{t+h})^2
```

The optimizer is AdamW with weight decay. Early stopping is applied based on validation loss, and the best checkpoint is saved for final test evaluation. Gradient clipping is used to improve training stability.

## 9. Hyperparameter Optimization

To improve the performance of the proposed eHMM-Transformer, hyperparameter optimization is conducted using Optuna. The optimization process searches for the best model architecture and training configuration based on validation loss.

The objective function minimizes the validation mean squared error of the proposed model:

```text
minimize Validation MSE
```

The search space includes both architectural and optimization hyperparameters:

```text
d_model          in {64, 128, 256}
n_heads          in {4, 8}
num_layers       in [1, 4]
dim_feedforward  in {128, 256, 512}
dropout          in [0.05, 0.30]
learning_rate    in [1e-4, 3e-3]
weight_decay     in [1e-6, 1e-3]
batch_size       in {32, 64, 128}
```

During hyperparameter optimization, each trial is trained for a limited number of epochs to reduce computational cost. Early stopping is also applied within each trial. After the best configuration is selected, the final proposed model is retrained using the selected hyperparameters and evaluated on the held-out test set.

The hyperparameter optimization outputs are saved as:

```text
hpo_trials.csv
best_hyperparameters.csv
```

This procedure ensures that the proposed model is not evaluated using an arbitrary manually selected configuration, but rather with a validation-based optimized setting.

## 10. Baseline Models

The proposed eHMM-Transformer is compared with several baseline models:

```text
Naive Last Value
Seasonal Naive
LightGBM
LSTM
Vanilla Transformer
```

The Vanilla Transformer uses the same engineered demand, weather, and calendar features as the proposed model, but excludes the e-HMM regime probabilities. This comparison isolates the contribution of regime-guided features.

## 11. Ablation Study

An ablation study is conducted to evaluate the contribution of different regime representations:

```text
A1: Transformer only
A2: Transformer + hard HMM state
A3: Transformer + soft HMM probability
A4: Transformer + soft e-HMM probability
```

The purpose of this study is to test whether soft probabilistic regime information is more useful than hard regime labels, and whether an ensemble HMM improves forecasting performance compared with a single HMM.

## 12. Evaluation Metrics

Forecasting performance is evaluated using standard regression and time-series forecasting metrics:

```text
MAE
RMSE
MAPE
sMAPE
MASE
```

Mean Absolute Error (MAE) measures the average absolute prediction error. Root Mean Squared Error (RMSE) penalizes large forecasting errors more strongly. MAPE and sMAPE provide percentage-based error measures. MASE normalizes prediction error relative to a naive forecasting baseline, making it useful for time-series comparison.

The final outputs include:

```text
metrics.csv
ablation_table.csv
hpo_trials.csv
best_hyperparameters.csv
prediction_plot.png
regime_distribution.png
training_loss_curves.png
```

## 13. Expected Contribution

The main contribution of this method is the use of an ensemble HMM as a regime discovery module that guides a Transformer forecaster through feature-wise conditioning. Unlike conventional HMM-based forecasting approaches, the HMM is not used to directly predict future demand. Instead, it provides soft latent-regime probabilities that condition the Transformer's temporal representation on different demand states, such as low-load, normal-load, and peak-load periods.

Crucially, the regime probabilities are not concatenated as additional input features. They are injected through a FiLM conditioning mechanism that applies a regime-dependent affine transformation to the encoded representation. This makes the framework a genuine regime-guided Transformer, in which the discovered load regimes directly modulate the model's internal representation rather than acting as extra, largely redundant input columns.
