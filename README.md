# Hybrid Temporal Forecasting for Energy

This project focuses on forecasting energy consumption (electricity load) using time series analysis techniques. Specifically, this repository contains the Phase 1 implementation, which establishes a baseline predictive model using the **ARIMA** (AutoRegressive Integrated Moving Average) algorithm on hourly energy consumption data.

## 📁 Repository Structure

```text
├── dataset/
│   └── PJME_hourly.csv                     # Hourly energy consumption data
├── report/
│   └── Hybrid_temporal_forecaster_phase1.pdf # Phase 1 project report
├── Hybrid_Temporal_Forcasting (2).ipynb    # Main Jupyter Notebook with code and analysis
└── README.md                               # Project documentation
```

## 📊 Dataset

The dataset used is **PJME_hourly.csv**, representing the estimated energy consumption in Megawatts (MW) for the PJM East Region.

- **Frequency:** Hourly
- **Features:** 
  - `Datetime`: Timestamp of the record.
  - `PJME_MW`: Electricity load in Megawatts.

### Data Preprocessing
- Converted `Datetime` to explicit pandas datetime objects and set as the index.
- Sorted indices and handled duplicate timestamps (kept first occurrence).
- Enforced a strict hourly frequency (`asfreq('h')`).
- Handled missing values (`NaN`) by filling them with the median of the load.

## 🔬 Methodology

The Jupyter Notebook (`Hybrid_Temporal_Forcasting (2).ipynb`) covers the following steps:

1. **Exploratory Data Analysis (EDA):**
   - Plotted the overall electricity load time series.
   - Looked into the 2015 load pattern.
   - Analyzed average load distributions across different hours of the day and days of the week.
   
2. **Time Series Analysis:**
   - Evaluated stationarity by checking rolling mean and standard deviation limits.
   - Performed additive seasonal decomposition (period=24) to observe trend, seasonality, and residuals.
   
3. **Forecasting Model:**
   - **Train-Test Split:** Used 80% of the dataset sequentially for training and the remaining 20% for testing.
   - **Differencing:** Applied `.diff()` to achieve stationarity for certain analyses.
   - **Auto-ARIMA:** Executed on a subset containing the last 20,000 samples of the training data to efficiently search the hyperparameter space (using `pmdarima.auto_arima` with stepwise execution) and determine the optimal `(p, d, q)` order.
   - **ARIMA Formulation:** Fit the actual ARIMA model (`statsmodels.tsa.arima.model.ARIMA`) to the full training set using the best order found.

4. **Evaluation:**
   - Forecasted the steps required to align with the dimensions of the test set.
   - Evaluated the model against actual observed test data using **MAE** (Mean Absolute Error) and **RMSE** (Root Mean Squared Error).
   - Visualized the actual target variable (Train/Test sets) versus the ARIMA Forecast.

## 🚀 Getting Started

### Prerequisites

To run the notebook on your local machine, ensure you have Python installed along with the following libraries:

```bash
pip install pandas numpy matplotlib statsmodels pmdarima scikit-learn
```

### Running the Project

1. Clone or clone the repository to your local machine.
2. Ensure `PJME_hourly.csv` is correctly placed inside the `dataset/` directory. (Note: The notebook currently references `/content/PJME_hourly.csv`, which you may need to update to relative path `./dataset/PJME_hourly.csv`).
3. Open `Hybrid_Temporal_Forcasting (2).ipynb` in Jupyter Notebook or JupyterLab.
4. Execute the cells sequentially.

## 📌 Future Work (Phase 2+)
While Phase 1 implements an ARIMA model, future phases aim to incorporate "Hybrid" models by integrating deep learning components (like LSTMs, GRUs) or other advanced ensembling techniques to capture complex non-linear temporal dependencies and improve forecasting performance.
