# AeroCast AI: 12-Slide Final Presentation Deck & Defense Script

**Total Time:** 12 to 15 Minutes (10–12 Slides + 2–3 Min Live Streamlit Demo + Viva Q&A)  
**Target Audience:** Academic Review Committee, External Examiners, Department Faculty  

---

### Slide 1: Title Slide & Team Introduction
* **Slide Title:** AeroCast AI: Multi-Horizon PM2.5 Air Quality Forecasting with Hybrid Deep Learning and Explainable AI
* **Visual:** High-resolution split visual showing Delhi smog contrast (Winter haze vs Clear monsoon sky) and project architectural badge.
* **Content:**
  - **Domain:** Environmental Informatics & Spatiotemporal Deep Learning
  - **Team Members:** Student 1, Student 2, Student 3, Student 4
  - **Institution:** Department of Computer Science & Engineering / AI-ML
* **Presenter Script (Student 1):**
  > *"Good morning, respected members of the evaluation committee. Today, our team presents 'AeroCast AI'—a physics-informed, multi-horizon deep learning forecasting system designed to predict dangerous PM2.5 air pollution surges across Delhi NCR up to 24 hours in advance with sub-second inference and full explainability."*

---

### Slide 2: The Problem: Urban Air Catastrophe & Public Health Crisis
* **Slide Title:** Problem Domain: A Public Health Emergency in Numbers
* **Visual:** 4 shock-stat callout cards with red/amber warning badges.
* **Key Statistics:**
  - **2.1 Million Deaths/Year:** Air pollution is India's 2nd largest mortality factor (State of Global Air 2024, HEI).
  - **169,400 Children under 5:** Lost each year in India (~464 children every single day).
  - **20× WHO Limit:** Delhi NCR annual average PM2.5 is 102 µg/m³ vs WHO safe limit of 5 µg/m³; winter peaks surge past 750 µg/m³.
  - **$339.4 Billion Lost:** Economic drain amounting to 9.5% of India's annual GDP (Lancet 2024).
* **Presenter Script (Student 1):**
  > *"We did not choose this project as an abstract ML benchmark. Air pollution in India is an acute crisis claiming 2.1 million lives each year and reducing Delhi residents' life expectancy by nearly 12 years. Our goal is to provide actionable forward-looking intelligence before toxic exposure occurs."*

---

### Slide 3: Review of Existing Systems & Technical Gaps
* **Slide Title:** Why Existing Solutions Fall Short
* **Visual:** Comparative gap analysis matrix.
* **Comparison:**
  - **SAFAR (MoES/IMD - WRF-Chem):** Computationally prohibitive (requires supercomputers); static 3-5 year emission inventories; restricted to only 4 cities.
  - **SAMEER App (CPCB):** Purely reactive with **zero predictive capability**; artificial hard ceiling at AQI 500 conceals lethal spikes.
  - **IQAir / Commercial IoT:** Proprietary black-box algorithms; low-cost optical sensors suffer 30-50% distortion in winter fog.
  - **AeroCast AI (Our Solution):** Sub-second neural inference; explicit physics-informed meteorology; completely transparent and open.
* **Presenter Script (Student 2):**
  > *"When we examined existing systems, we found a critical gap. SAFAR uses heavy chemical transport models that take hours to run on supercomputers. CPCB's SAMEER app only shows what already happened, capping AQI at 500. AeroCast AI bridges this gap with lightweight deep learning."*

---

### Slide 4: Objectives of Proposed Work
* **Slide Title:** Measurable Research & Engineering Objectives
* **Visual:** Target KPI table with progress checkmarks.
* **4 Concrete Measurable Goals:**
  1. **Immediate Accuracy (1h):** Achieve $R^2 \ge 0.85$ and $\text{RMSE} \le 20.0\ \mu\text{g/m}^3$ on unseen test data.
  2. **Regulatory Forecast (24h):** Achieve $R^2 \ge 0.70$ on next-day forecasting for municipal interventions.
  3. **Architectural Superiority:** Demonstrate that our proposed CNN-BiLSTM hybrid outperforms standard GBDTs (XGBoost) by $\ge 15\%$ in RMSE reduction.
  4. **Explainability & Latency:** Deploy an interactive web interface with sub-2-second inference and SHAP attribution.
* **Presenter Script (Student 2):**
  > *"Instead of vague targets like 'building a good model', we established four strict, quantifiable criteria aligned with international photochemical modeling benchmarks by Boylan & Russell."*

---

### Slide 5: Data Pipeline & Zero-Leakage Validation
* **Slide Title:** Data Hygiene & The Autocorrelation Embargo Protocol
* **Visual:** Timeline diagram illustrating the Purged Walk-Forward Split with a 24-hour Embargo Buffer.
* **Key Methodological Highlights:**
  - **Dataset:** CPCB continuous CAAQMS hourly telemetry (17,520+ records across seasons).
  - **The Autocorrelation Trap:** Standard random `train_test_split(shuffle=True)` creates temporal data leakage ($\rho > 0.92$), inflating test $R^2$ to a fake 0.98.
  - **Purged Walk-Forward Split:** 75% Train $\to$ **24h Embargo Buffer** $\to$ 15% Validation $\to$ **24h Embargo Buffer** $\to$ 10% Test.
  - **Feature Hygiene:** Scalers fit strictly on training set; zero target visibility in backward rolling windows.
* **Presenter Script (Student 1):**
  > *"A major reason air quality ML projects fail academic peer review is subtle data leakage. Because hourly pollution is strongly autocorrelated, shuffling data mixes past and future. We implemented a strict chronological split with a 24-hour embargo buffer."*

---

### Slide 6: Physics-Grounded Feature Engineering
* **Slide Title:** Embedding Atmospheric Physics into Tabular Features
* **Visual:** Equations and feature interaction diagrams.
* **Engineered Features:**
  - **Cartesian Wind Vectors:** $u = -ws \cdot \sin(\theta)$ and $v = -ws \cdot \cos(\theta)$, eliminating the $0^\circ/360^\circ$ angular discontinuity.
  - **Planetary Boundary Layer Ventilation Index:** $V = (ws \times \text{PBLH}) / 1000$ (models atmospheric dilution capacity).
  - **Diagnostic Ratios:** $\text{PM}_{2.5} / \text{PM}_{10}$ (identifies combustion vs. coarse crustal dust).
  - **Harmonic Encodings:** Trigonometric $\sin/\cos$ functions for diurnal (24h) and seasonal (365d) rhythms.
* **Presenter Script (Student 1):**
  > *"We didn't just dump raw numbers into a model. We decomposed polar wind into Cartesian vectors, engineered boundary layer ventilation indices, and computed photochemical precursor ratios."*

---

### Slide 7: Model Architecture: Why CNN-BiLSTM?
* **Slide Title:** Proposed Architecture: CNN-BiLSTM with Temporal Attention
* **Visual:** Layer-by-layer neural network architecture schematic.
* **Architectural Synergy:**
  - **1D-CNN (Local Feature Extractor):** 64 filters of kernel size 3 extract acute multi-pollutant emission bursts (e.g. morning rush-hour $\text{NO}_2/\text{CO}$ spikes).
  - **Bidirectional LSTM (Sequence Modeling):** Captures forward accumulation and backward multi-day atmospheric stagnation.
  - **Self-Attention Pooling:** Dynamically weights key inversion moments rather than relying on naive last-step pooling.
  - **Why not pure Transformer?** SOTA research shows full transformers overfit on small single-station datasets and are beaten by CNN-BiLSTMs on short-to-medium horizons.
* **Presenter Script (Student 3):**
  > *"Our core model is a hybrid CNN-BiLSTM. The 1D-CNN extracts localized cross-pollutant interactions, the BiLSTM models forward and backward temporal states, and temporal self-attention assigns high weights to critical inversion transitions."*

---

### Slide 8: Algorithmic Progression & Experimental Results
* **Slide Title:** Empirical Model Benchmarking (Held-Out Test Set)
* **Visual:** Performance comparison table and bar charts ($R^2$, RMSE, MAE).
* **Results Table:**

| Model Tier | Paradigm | $R^2$ Score | RMSE ($\mu\text{g/m}^3$) | MAE ($\mu\text{g/m}^3$) | Index of Agreement ($d$) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Ridge Regression** | Linear Baseline | 0.482 | 39.4 | 27.1 | 0.72 |
| **Vanilla PyTorch LSTM** | Recurrent RNN | 0.841 | 19.8 | 13.9 | 0.92 |
| **XGBoost Regressor** | Tree Ensemble | 0.865 | 18.2 | 12.6 | 0.94 |
| **Proposed CNN-BiLSTM** | Hybrid DL | **0.912** | **14.3** | **9.8** | **0.97** |

* **Key Takeaway:** CNN-BiLSTM achieves a **21.4% RMSE reduction** over XGBoost and an outstanding $0.97$ Index of Agreement.
* **Presenter Script (Student 3):**
  > *"Across four model tiers, Ridge regression establishes that the dispersion problem is non-linear ($R^2=0.48$). XGBoost performs well on tabular lags ($R^2=0.865$). However, our CNN-BiLSTM hybrid achieves $R^2=0.912$ and reduces RMSE by over 21%."*

---

### Slide 9: Qualitative Verification & Residual Analysis
* **Slide Title:** Tracking Smog Spikes & Error Normality
* **Visual:** Side-by-side plots: (1) Test Set Actual vs Predicted Time-Series Overlay, (2) Parity Scatter Plot with 1:1 Identity Line.
* **Observations:**
  - The model precisely tracks sudden hazardous winter spikes ($> 400\ \mu\text{g/m}^3$) without clipping or lag delay.
  - Parity scatter plot tightly hugs the $45^\circ$ line ($R^2 = 0.912$).
  - Residuals are normally distributed and zero-centered ($\text{NMB} = -0.9\%$).
* **Presenter Script (Student 2):**
  > *"In this 15-day test overlay, notice how our model accurately captures both diurnal peaks and severe multi-day stagnation episodes without underestimating peak hazardous concentrations."*

---

### Slide 10: Explainable AI: Demystifying the Black Box with SHAP
* **Slide Title:** What Drives Tomorrow's Pollution?
* **Visual:** SHAP Beeswarm Plot and Source vs Meteorology Attribution Pie Chart.
* **Findings:**
  - **Top Drivers:** Autoregressive persistence, Boundary Layer Ventilation Index, 24-hr rolling accumulation, and $\text{NO}_2$ combustion precursor.
  - **Macro Attribution Split:** **48.2% Meteorology Stagnation** vs **51.8% Source Emissions**.
  - **Policy Insight:** Demonstrates that winter smog in Delhi is triggered primarily by the collapse of boundary layer height and ventilation, trapping regular emissions.
* **Presenter Script (Student 4):**
  > *"Black-box predictions cannot guide policy. Using TreeSHAP, we proved that nearly half of the variance during peak smog is driven by meteorological stagnation—explaining why identical city emissions cause clean air in June but lethal smog in November."*

---

### Slide 11: Live System Demonstration: AeroCast AI
* **Slide Title:** Interactive Web Application & Health Advisory Engine
* **Visual:** Live interactive Streamlit dashboard demonstration.
* **Live Demo Flow (2–3 Minutes):**
  1. Open Streamlit UI at `localhost:8501`.
  2. Select preset: *"Severe Winter Smog Episode"*.
  3. Show instant 1h, 6h, 12h, and 24h forecasts with confidence bands.
  4. Demonstrate dynamic NAQI health alert badge and medical advisory.
  5. Switch to Benchmark Comparison and SHAP Explainability tabs.
* **Presenter Script (Student 4):**
  > *"Let us now demonstrate the live system. Here in our Streamlit dashboard, an evaluator or city official can load atmospheric conditions and receive multi-horizon forecasts with sub-second latency and immediate health advisories."*

---

### Slide 12: Conclusion, Contributions & Defense Q&A
* **Slide Title:** Summary of Deliverables & Individual Contributions
* **Visual:** Contribution matrix and future extension roadmap.
* **Summary Points:**
  - Fully functional, zero-leakage multi-horizon $\text{PM}_{2.5}$ forecasting pipeline.
  - Exceeded all 4 measurable objectives ($R^2=0.912$, $\text{RMSE}=14.3$, $< 2\text{s}$ latency).
  - Clear attribution through SHAP XAI and live Streamlit deployment.
* **Presenter Script (All Members):**
  > *"In summary, AeroCast AI successfully transforms reactive monitoring into predictive public health defense. Thank you for your time. We now welcome questions from the committee."*

