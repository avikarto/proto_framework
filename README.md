# proto_framework
Basic ML framework for prototyping, currently configured to run a binary classification pipeline for
the sklearn breast cancer dataset.


1) src/modeling/tune.py
- AUC Hyperparameter optimization for the RFC model to be trained.
- python src/modeling/tune.py --help


2) src/modeling/train.py
- Train a RFC model, with or without undersampling to address class imbalance.
- python src/modeling/train.py --help


3) src/modeling/predict.py
- Make predictions using the model trained in step 2.
- python src/modeling/predict.py --help


To run these scripts, you must create a `.env` file in the project root directory which will define
`PROJECT_ROOT`, namely the full path to the directory containing this `.env` file.
