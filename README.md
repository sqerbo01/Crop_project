# 🌾 Barley Climate Risk App

This project analyzes the impact of climate scenarios on barley yields across French departments and provides forward-looking risk insights through an interactive Streamlit application.

The app supports:

- Yield forecasting under multiple climate scenarios  
- Climate-only impact analysis  
- Department-level risk ranking  
- Water stress and climate sensitivity diagnostics  

---

## 📁 Project Structure

Crop_project/
│
├── app/
│   └── app.py
├── src/
├── data/                # ⚠️ You must place datasets here (not tracked)
├── models/
├── .streamlit/
├── requirements.txt
└── README.md

---

## ⚠️ Data Requirements (Important)

The datasets are **not included in the repository**.

You must manually place them in the `data/` folder with the following names:

data/
- barley_yield_from_1982.csv  
- climate_data_from_1982.parquet  

If the files are missing or misnamed, the app will fail at startup.

---

## 🧪 Environment Setup (macOS / Linux)

### 1. Clone the repository

git clone https://github.com/sqerbo01/Crop_project.git  
cd Crop_project

### 2. Create and activate virtual environment

python3 -m venv .venv  
source .venv/bin/activate  

You should see:

(.venv)

### 3. Install dependencies

pip install --upgrade pip  
pip install -r requirements.txt  

---

## 🤖 Train the Models (required once)

Before running the app, generate the model artifacts:

python -m src.train

This will create:

- models/model_with_year.joblib  
- models/model_no_year.joblib  

---

## 🚀 Run the Streamlit App

From the project root:

python -m streamlit run app/app.py

Then open the local URL shown in the terminal (usually):

http://localhost:8501

---

## 📊 App Overview

### Forecast
- Yield projections including time trend  
- Department or portfolio view  
- Economic value estimation  

### Climate-only
- Isolates pure climate signal  
- Removes technological/time trend  
- Useful for scenario stress testing  

### Risk
- Department ranking by projected downside  
- Severe-year yield indicator  
- Water stress index  
- Climate sensitivity metrics  

---

## 🧠 Key Methodological Notes

- Crop season defined as Sep–Jun crop year  
- Climate features aggregated seasonally  
- Rolling time-series validation used for training  
- Risk metrics compare future severe years vs historical baseline  
- Water stress index is a relative proxy, not a physical hydrological model  

---

## 🛠 Troubleshooting

### Streamlit cannot find `src`

Run with:

PYTHONPATH=. python -m streamlit run app/app.py

### Models missing

Make sure you ran:

python -m src.train

### Data loading errors

Verify datasets exist exactly at:

data/barley_yield_from_1982.csv  
data/climate_data_from_1982.parquet  

---

## 📌 Future Improvements

- Geospatial visualization  
- Probabilistic yield distributions  
- Irrigation scenario modeling  
- Multi-crop support  