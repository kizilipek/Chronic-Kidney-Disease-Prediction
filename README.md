# Chronic-Kidney-Disease-Prediction
Predict whether a patient will progress in CKD staging given the patient's past longitudinal information.
## Requirements
The code requires Python 3.9 or later. The file [requirements.txt](requirements.txt) contains the full list of
required Python modules.
```bash
pip install -r requirements.txt
```

## Training and Evaluation
For running model for specific settings add_meds=True includes the medication data, add_static=True runs the second model with static patient data. 
```bash
python lstm.py --add-meds=True --add-static=False
```

For data only operations, you can call data_preprocess with necessary arguments: add_meds=True creates a dataset with biomarkers and medication data combined time level. Also it is possible to pass raw_data and processed data directories as arguments.
```bash
python data_preprocess.py --add-meds=True

```