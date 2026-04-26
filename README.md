# Building Energy Performance Predictor

## Description of the App
This Streamlit application predicts total, cooling, and heating energy use intensity for an office-building design scenario using pretrained Explainable Boosting Machine (EBM) regression models.

## Model Description
Three separate EBM pipelines are packaged with the project:
- `models/ebm_eui.pkl`
- `models/ebm_cooling.pkl`
- `models/ebm_heating.pkl`

Each model uses min-max normalization followed by EBM regression for interpretable predictions.

## Explainability Features
- Global feature importance
- Partial dependence plots for each input feature
- EBM shape functions from `explain_global()`
- Local explanation from `explain_local()`
- Sensitivity mini-analysis with ±10% perturbations for EUI

## Deployment Instructions for Streamlit Cloud
1. Upload the `Mine app` folder to a GitHub repository.
2. In Streamlit Cloud, create a new app and point the entry file to `app.py`.
3. Streamlit Cloud will install dependencies from `requirements.txt` automatically.
4. Confirm that the `models/`, `data/`, and `utils/` folders are included in the repository.

## Credits
Developed by:
Alireza Oroomiei
Dr. Morteza Rahbar
Dr. Mohamadali Khanmohamadi
School of Architecture, Iran University of Science and Technology
