# Smart Facility Pulse

**A multi-page Streamlit dashboard for facilities management — real-time comfort monitoring, pool quality tracking, and HVAC predictive maintenance — built as a portfolio proof-of-concept around the operational needs of a residential FM team.**

[Live Dashboard](https://smartfacilitymanagement.streamlit.app/)] 

---

## What this is

A multi-page operational dashboard that demonstrates how a building's BMS (Building Management System) data can be turned into a proactive command center for Facility Management teams. Three modules:

| Module | Purpose | Data | Logic |
|---|---|---|---|
| **A. Comfort & IAQ** | Real-time zone monitoring (temp, humidity, CO₂, PM2.5, CO) | Synthetic, calibrated to CU-BEMS commercial-building ranges | Rule-based thresholds (ASHRAE-aligned) |
| **B. Pool Quality** | Pool chemistry monitoring + service prediction | Synthetic, calibrated to CDC public-pool standards | Rule-based + linear trend extrapolation |
| **C. HVAC Predictive** | FCU fault detection + SHAP explainability | **Real LBNL public dataset** (DOI: 10.25984/1881324) | XGBoost classifier (Test AUC: 0.997) |

The HVAC module is the ML showcase. The model is trained on LBNL's published Fan Coil Unit fault dataset — the academic benchmark for HVAC fault detection research — and explained using SHAP on real test data.

## Reality of the data, and the model being used

- **Comfort and pool data are synthetic.** Calibrated to industry-standard ranges, but not real BMS readings. A real deployment would connect to a building's BMS feed.
- **The HVAC fault model achieves 0.997 AUC on LBNL test data** — exceptional, but on a simulated benchmark. Expected real-world AUC on noisy BMS data: ~0.85-0.92.
- **Building FCU fleet scoring uses a rule layer**, not the trained model directly. Why? Zone-level synthetic sensors don't contain FCU-internal signals (valve positions, internal airflows). The rule layer infers fault classes from observable symptoms; the actual model runs on the LBNL test set on a separate page (see "HVAC Performance Reference"). This separation is documented inside the dashboard itself.

## Live dashboard

[https://smartfacilitymanagement.streamlit.app/](#)

## Screenshots

<img width="1541" height="592" alt="image" src="https://github.com/user-attachments/assets/6fec2862-a536-4793-9a31-f9b36e6be809" />

<img width="1557" height="885" alt="image" src="https://github.com/user-attachments/assets/387ff28d-e14b-4e95-b684-f4da85e99752" />

<img width="1523" height="785" alt="image" src="https://github.com/user-attachments/assets/2eddf023-de65-4664-aaaf-fdded0e5d95e" />

<img width="1535" height="822" alt="image" src="https://github.com/user-attachments/assets/37d371b9-48e1-444e-a82b-daff3b7a61e4" />

## Project structure
```
smart-facility-pulse/
├── notebooks/                    # End-to-end notebooks (data + model)
│   ├── 01_synthetic_comfort_data.ipynb
│   ├── 02_synthetic_pool_data.ipynb
│   ├── 03_lbnl_fcu_eda.ipynb
│   ├── 04_hvac_fault_model.ipynb
│   └── 05_streamlit_app_builder.ipynb
├── app/                          # Streamlit dashboard (deployed)
│   ├── streamlit_app.py          # Page 1: Building Overview
│   ├── pages/
│   │   ├── 2_Comfort_Monitoring.py
│   │   ├── 3_Pool_Quality.py
│   │   ├── 4_HVAC_Predictive_Hybrid.py
│   │   └── 5_HVAC_Performance_Reference.py
│   └── utils/                    # Shared logic
│       ├── data_loader.py        # Cached data access
│       ├── alert_engine.py       # Threshold + symptom-to-fault rules
│       ├── sidebar.py            # Cross-page alert panel
│       └── theme.py              # Dark-mode design tokens
├── data/
│   ├── synthetic/                # Generated comfort + pool data
│   └── lbnl/                     # Processed LBNL FCU subset
└── models/                       # Trained XGBoost + SHAP artifacts
```
## Methodology highlights

### Why time-based train/test split (not random)
Each minute of FCU sensor data correlates strongly with the next. Random splits leak future information into training. We train on Jan-Sep 2018 LBNL data and test on Oct-Dec 2018, mirroring how the model would be deployed in production.

### Why hourly downsampling
LBNL data is 1-minute resolution (525,600 rows per file × 49 files = 25.7 M rows). In a realistic scenario, a real BMS would typically log data at 5-15 minute intervals. Creating an hourly mean balances signal preservation with performance and dataset size.

### Why filter to occupied hours only
Faults are only observable when the system is actively running (LBNL schedule: Mon-Fri 6am-6pm operate, otherwise setback). Including setback hours dilutes fault signatures with idle data and also increases the dataset's size. 

### Symptom-to-fault mapping (Building Fleet page)
The trained model expects FCU-internal sensors. Until a real BMS feed is connected, the building FCU fleet (81 zones) is scored using a rule layer derived from the model's learned feature importance (top SHAP features). Each rule maps a zone-level symptom pattern (e.g. "rising temp + normal CO₂") to the most likely LBNL fault class. Documented in-app with full rule table.

## Data sources

- **LBNL FCU Fault Detection Dataset** — Granderson J, Lin G, Chen Y, et al. *LBNL FDD Datasets.* DOE Open Energy Data Initiative, 2022. DOI: [10.25984/1881324](https://dx.doi.org/10.25984/1881324). Generated using HVACSIM+ (NIST). Used 18 of 49 cooling-relevant scenarios.
- **CU-BEMS** — Pipattanasomporn et al., *Smart building electricity consumption and indoor environmental sensor datasets.* Scientific Data, 2020. Used as the calibration reference for synthetic Module A ranges.
- **CDC pool guidance** — Used as the basis for Module B chemistry thresholds.
- **ASHRAE Standard 55 / 62.1** — Used for Module A comfort thresholds.

## Running locally

```bash
git clone https://github.com/<your-username>/smart-facility-pulse.git
cd smart-facility-pulse
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

## License

Code: MIT. LBNL dataset is public domain (US Department of Energy).

## Author

Built as a project to provide a solution to real FM teams. The author is a Data Analyst transitioning into Data Science, working in the real estate sector with a background in mechatronics engineering and deeply passionate about data.
