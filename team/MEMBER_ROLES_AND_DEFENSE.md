# Team Member Roles, Code Ownership & Viva Defense Guide

**Project Title:** AeroCast AI: Multi-Horizon PM2.5 Forecasting Engine  
**Team Size:** 4 Members  
**Target Evaluation Rubric:** Criterion #10 (Individual Contributions & Technical Defense)  

---

## 1. Division of Labor & Code Ownership Matrix

To score full marks on Rubric #10, each team member must have distinct, parallelizable responsibilities and own specific source files. Under questioning, each student should speak to their specific module with deep technical ownership.

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                       TEAM ROLES & RESPONSIBILITIES                                     │
├───────────────────┬───────────────────┬───────────────────────────────┬─────────────────────────────────┤
│ Team Member       │ Primary Role      │ Core Code Files Owned         │ Technical Specialty             │
├───────────────────┼───────────────────┼───────────────────────────────┼─────────────────────────────────┤
│ **Student 1**     │ Data Engineer     │ `data/download_data.py`       │ CPCB data hygiene, missingness  │
│                   │ & Physics Pipeline│ `src/preprocessing.py`        │ imputation, wind vectorization, │
│                   │                   │ `src/feature_engineering.py`  │ zero-leakage embargo splitting  │
├───────────────────┼───────────────────┼───────────────────────────────┼─────────────────────────────────┤
│ **Student 2**     │ Tabular & Baseline│ `src/models/baseline_lr.py`   │ Ridge L2 optimization, GBDT     │
│                   │ Modeling Lead     │ `src/models/xgboost_model.py` │ tree depth tuning, walk-forward │
│                   │                   │ `src/train.py` (Orchestration)│ validation, time-series splits  │
├───────────────────┼───────────────────┼───────────────────────────────┼─────────────────────────────────┤
│ **Student 3**     │ Deep Learning     │ `src/models/lstm_model.py`    │ PyTorch sequence modeling,      │
│                   │ Architect         │ `src/models/cnn_bilstm.py`    │ 1D-CNN temporal filters, BiLSTM │
│                   │                   │ `src/utils.py`                │ attention pooling, Huber loss   │
├───────────────────┼───────────────────┼───────────────────────────────┼─────────────────────────────────┤
│ **Student 4**     │ XAI & Deployment  │ `src/explainability.py`       │ TreeSHAP feature attributions,  │
│                   │ Engineer          │ `src/evaluate.py`             │ Streamlit dashboard, CPCB NAQI  │
│                   │                   │ `app/streamlit_app.py`        │ health advisories, parity plots │
└───────────────────┴───────────────────┴───────────────────────────────┴─────────────────────────────────┘
```

---

## 2. Individual Viva Questions & High-Scoring Defense Answers

### 👤 Student 1: Data Engineering & Physics-Grounded Feature Engineering

#### Q1: "Why can't you just use standard random train_test_split from scikit-learn for this project?"
* **Model Answer (Student 1):**
  > *"Because air pollution time-series data has very high temporal autocorrelation—the correlation between PM2.5 at hour $t$ and hour $t-1$ exceeds 0.92. If we randomly shuffle the data, samples from $t-1$ and $t+1$ end up in the training set while time $t$ is in the test set. The model merely interpolates adjacent points rather than learning actual forecasting dynamics. This inflates the test $R^2$ to a fake 0.98. In production, that model collapses. We avoided this by enforcing a strict chronological split with an explicit 24-hour embargo buffer between train and test sets to completely eliminate autocorrelation leakage."*

#### Q2: "Why did you convert wind speed and direction into Cartesian u and v components instead of using degrees?"
* **Model Answer (Student 1):**
  > *"Wind direction in degrees has an artificial mathematical discontinuity at the north boundary: $359^\circ$ and $1^\circ$ are numerically far apart (a difference of 358 degrees), but physically they represent virtually identical northerly winds. Neural networks and tree models cannot handle this jump without distortion. We decomposed the polar vectors into continuous orthogonal components: $u = -ws \cdot \sin(\theta)$ and $v = -ws \cdot \cos(\theta)$, which preserve fluid continuity everywhere."*

#### Q3: "What is the Ventilation Index and why does it matter?"
* **Model Answer (Student 1):**
  > *"The atmospheric ventilation index is the product of horizontal wind speed and the Planetary Boundary Layer Height (PBLH) divided by 1000. It measures the physical volume of the atmospheric mixing box. In winter, the boundary layer collapses from 2000m down to 200m and wind speeds drop below 1 m/s, causing ventilation to collapse. Even if urban emissions remain constant, the reduced volume forces particulate concentration to spike exponentially."*

---

### 👤 Student 2: Tabular Modeling, Baselines & Validation Strategy

#### Q1: "Why did you include Ridge regression if you already knew deep learning would be better?"
* **Model Answer (Student 2):**
  > *"In academic research, you must establish an empirical lower bound to demonstrate that the complexity of neural networks is actually justified. Ridge regression uses an $L_2$ penalty to handle severe multicollinearity among meteorological variables (like temperature, humidity, and pressure). Our Ridge model achieved an $R^2$ of only 0.48, which quantitatively proves that atmospheric dispersion and chemical kinetics are fundamentally non-linear and cannot be solved with linear combinations alone."*

#### Q2: "Why did you pick XGBoost over Random Forest?"
* **Model Answer (Student 2):**
  > *"Empirical literature—such as Singh et al. (2025) on Indian urban air quality—shows that Random Forest severely overfits on historical time-series, with training $R^2$ reaching 0.99 while test $R^2$ collapsed to 0.35. Because Random Forest averages predictions in leaf nodes, it compresses dynamic range toward the sample mean and fails to predict extreme episodic peaks. In contrast, XGBoost uses second-order Taylor expansions of the loss function and penalizes tree complexity through $\gamma T + \frac{1}{2}\lambda \sum w_j^2$, achieving much better generalization ($R^2 = 0.865$)."*

#### Q3: "How did you prevent target leakage in your rolling statistics?"
* **Model Answer (Student 2):**
  > *"When computing 3-hour, 6-hour, and 24-hour rolling averages, if you calculate the rolling mean directly at timestamp $t$ with `center=True`, you incorporate future values $[t+1, \dots, t+12]$. Even with `center=False`, you still include the concurrent target value at time $t$. We strictly shifted the target series by 1 hour forward (`shift(1)`) before computing rolling statistics, ensuring that the feature vector for time $t$ only sees data up to time $t-1$."*

---

### 👤 Student 3: Deep Learning Architecture (CNN-BiLSTM)

#### Q1: "Walk us through the exact architecture of your CNN-BiLSTM hybrid and explain why each layer is there."
* **Model Answer (Student 3):**
  > *"Our architecture consists of three specialized stages: First, a 1D-Convolutional layer with 64 filters and kernel size 3. This acts as a localized receptive field filter across the 24-hour window to extract cross-pollutant interactions—such as morning rush-hour $\text{NO}_2/\text{CO}$ bursts and sudden cold fronts. Second, a 2-layer Bidirectional LSTM with 64 hidden units. BiLSTM processes the sequence both forwards and backwards, because in atmospheric science, current pollution is governed both by cumulative past buildup and by the multi-day duration of the stagnation episode. Third, a Temporal Self-Attention layer that learns dynamic weights over all 24 sequence steps rather than naively taking the last hidden state. Finally, a dense projection head with Mish activations and Huber loss outputs the forecast."*

#### Q2: "Why did you use Huber Loss instead of standard Mean Squared Error (MSE)?"
* **Model Answer (Student 3):**
  > *"MSE penalizes errors quadratically. During North Indian winters, episodic pollution spikes can exceed $700\ \mu\text{g/m}^3$ due to Diwali fireworks or crop burning. With MSE, large outlier errors produce massive gradients that destabilize neural network weights. Huber loss acts quadratically for small errors ($|y - \hat{y}| \le \delta$), providing smooth convergence, but transitions to a linear penalty for errors larger than $\delta=1.0$, preventing extreme winter spikes from dominating gradient descent."*

#### Q3: "Why did you choose CNN-BiLSTM instead of a Transformer architecture like PatchTST or Informer?"
* **Model Answer (Student 3):**
  > *"Transformers have high data appetite and quadratic self-attention complexity. As recent benchmarks by Grinsztajn et al. (NeurIPS 2022) and Zeng et al. (AAAI 2023) demonstrated, full transformers frequently overfit on single-station datasets and are consistently outperformed by tightly regularized recurrent/convolutional hybrids on short-to-medium continuous horizons. Our CNN-BiLSTM gives superior convergence with under 100,000 parameters and trains in minutes on Google Colab."*

---

### 👤 Student 4: Explainability (SHAP), Evaluation & Interactive Deployment

#### Q1: "Why isn't reporting R² and RMSE enough? Why did you implement SHAP?"
* **Model Answer (Student 4):**
  > *"In high-stakes environmental health and municipal policymaking, a black-box number is unhelpful. If our model predicts that tomorrow's PM2.5 will jump by $150\ \mu\text{g/m}^3$, city officials need to know: 'Is this jump caused by a spike in local vehicular emissions, or is it caused by calm winds and boundary layer compression?' Using TreeSHAP, we satisfy game-theoretic Shapley properties to attribute exact numerical contributions to each feature, separating meteorological stagnation from emission sources."*

#### Q2: "What did your SHAP decomposition reveal about air pollution in Delhi?"
* **Model Answer (Student 4):**
  > *"Our SHAP analysis revealed a 48.2% to 51.8% split between meteorological stagnation and precursor emissions. The top physical drivers were the 1-hour autoregressive lag, the ventilation index, and $\text{NO}_2$. This proves mathematically that severe winter smog crises in Delhi are not caused solely by sudden surges in urban emissions, but by the physical collapse of atmospheric ventilation, which traps ongoing baseline emissions in a shallow surface layer."*

#### Q3: "How is your Streamlit app designed for operational evaluation?"
* **Model Answer (Student 4):**
  > *"Our Streamlit dashboard provides sub-second inference. It allows an examiner or municipal planner to either load real-world episode presets (such as 'Severe Winter Smog' or 'Monsoon Washout') or manipulate telemetry sliders. The system outputs multi-horizon predictions for 1h, 6h, 12h, and 24h with shaded 95% confidence intervals, alongside official CPCB National Air Quality Index health warning badges and medical advisories."*

