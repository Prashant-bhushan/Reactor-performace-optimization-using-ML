# Reactor-performace-optimization-using-ML
# Reactor Performance Prediction and Optimization

This project provides a complete workflow for data-driven modeling, prediction, and optimization of chemical reactor performance (Batch, CSTR, and PFR) using machine learning techniques.

## 📁 Project Structure

- `BATCH.ipynb` – Batch reactor data generation, ML modeling, and optimization
- `CSTR.ipynb` / `CSTR1.ipynb` – CSTR data generation, ML modeling, and optimization
- `PFR.ipynb` – PFR data generation, ML modeling, and optimization
- `batch_reactor_data_set.csv`, `CSTR data set.csv`, `PFR data set.csv` – Generated datasets
- `catboost_model.pkl`, `best_CSTR_model.pkl`, `pfr_model.pkl` – Trained ML models
- `scaler.pkl` – Feature scaler for normalization
- `Batch_GUI.py`, `CSTR_GUI.py`, `PFR_GUI.py` – Streamlit GUIs for each reactor type

## 🚀 Features

- **Synthetic Data Generation**: Simulates realistic reactor datasets using kinetic equations for each reactor type.
- **Data Cleaning & EDA**: Outlier removal, statistical summaries, correlation analysis, and visualization.
- **Feature Engineering**: Calculates conversion and other relevant features.
- **Model Training & Comparison**: Trains and compares Random Forest, Gradient Boosting, SVR, Linear Regression, XGBoost, CatBoost, and Neural Network models.
- **Hyperparameter Tuning**: Uses GridSearchCV for optimal model selection.
- **Feature Importance Analysis**: Identifies key variables affecting conversion.
- **Residual Analysis**: Evaluates model prediction errors.
- **Optimization**: Uses Bayesian Optimization to find optimal operating conditions for maximum conversion.
- **Deployment**: Streamlit GUIs for user-friendly prediction and optimization.

## 🧪 Methodology

1. **Data Generation**: Synthetic datasets are generated for each reactor using domain-specific kinetic equations.
2. **Preprocessing**: Data is cleaned (missing values, outliers), scaled, and split into train/test sets.
3. **Modeling**: Multiple ML models are trained and evaluated; best models are selected based on R² and MSE.
4. **Optimization**: Bayesian Optimization is used to maximize conversion by tuning input variables.
5. **Deployment**: Trained models and scalers are saved and integrated into Streamlit GUIs for easy use.

## 📊 Results

- **Batch Reactor**: CatBoost achieved the best performance (R² ≈ 0.99).
- **CSTR**: Random Forest or Gradient Boosting typically performed best (R² > 0.95).
- **PFR**: Random Forest achieved high accuracy (R² > 0.95).
- Feature importance and residual plots are included for model interpretability.

## 🖥️ Usage

1. Clone the repository and install dependencies:
    ```sh
    pip install -r requirements.txt
    ```
2. Run the Jupyter notebooks (`BATCH.ipynb`, `CSTR.ipynb`, `PFR.ipynb`) to generate data, train models, and perform optimization.
3. Launch the Streamlit GUIs for interactive prediction:
    ```sh
    streamlit run Batch_GUI.py
    streamlit run CSTR_GUI.py
    streamlit run PFR_GUI.py
    ```

## 📚 References

- [CHE506 Reaction Engineering Laboratory](https://www.isca.in/rjcs/Archives/v5/i11/3.ISCA-RJCS-2015-137.pdf)
- [IJERA Research Paper](https://www.ijera.com/papers/Vol5_issue2/Part%20-%202/K502027478.pdf)
- [IRJET Research Paper (k0 and Ea values)](https://www.irjet.net/archives/V6/i3/IRJET-V6I31210.pdf)

---

**Developed by:** Prashant Bhushan  
**Course:** Final Year Project (FYP 2025)  
