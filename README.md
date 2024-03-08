# proto_framework
Basic ML frameworks for prototyping.  Current examples include 1) a binary classification pipeline
(RFC) for the sklearn breast cancer dataset, and 2) an XGBoost time-series analysis.

To run these scripts, you must create a `.env` file in the project root directory which will define
`PROJECT_ROOT`, namely the full path to the directory containing this `.env` file.

## For the RFC model

1) src/RFC/tune.py
- AUC Hyperparameter optimization for the RFC model to be trained.
- python src/modeling/tune.py --help


2) src/RFC/train.py
- Train a RFC model, with or without undersampling to address class imbalance.
- python src/modeling/train.py --help


3) src/RFC/predict.py
- Make predictions using the model trained in (2).
- python src/modeling/predict.py --help


## For the time-series model

1) src/time_series/explore_ts.ipynb
- Generate a time-series dataset with mutlii-periodic signal and noise
- Illustrate extracting frequencies in the periodic data with fourier analysis.
- View the final prediction results after having run steps (2) - (4)

2) src/RFC/tune.py
- Hyperparameter optimization for the XGB model to be trained (using oldest 50% of data).

3) src/time_series/train.py
- Train an XGBoost model on this dataset, leveraging learnings from exploration and tuning in (1) +
(2) (using oldest 80% of data).

4) src/time_series/predict.py
- Make predictions for the most-recent 20% of time-series data.
