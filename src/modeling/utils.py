# Various helper functions for modeling activities

import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from dotenv import load_dotenv
load_dotenv()
sys.path.append(os.environ['PROJECT_ROOT'] + '/src')
# from data.utils import query_engine  # noqa: E402


def data_splits(df: pd.DataFrame, validate: bool = False) -> pd.DataFrame:
	"""
	Generate train/test splits in data, and optionally also a validation split for model selection
	or hyperparameter tuning.

	Input: df - complete data with features and targets
	Input: validate - Generate a validation split from data?  If not, return train/test.
	Output: train/test splits of df, with also a validation split if validate=True
	"""

	# Split out test data (15% of inital set)
	x_train, x_test, y_train, y_test = train_test_split(
		df.drop(columns=['target']), df['target'],
		test_size=0.15,
		random_state=271828
	)

	if validate:
		# Create train (85% * 85%) and validate (85% * 15%) data from initial train (85%) split
		x_train, x_val, y_train, y_val = train_test_split(
			x_train, y_train,
			test_size=0.15,
			random_state=271828
		)
		return (x_train, x_test, x_val, y_train, y_test, y_val)
	else:
		return (x_train, x_test, y_train, y_test)
# def data_splits


def load_data():
	''' Load raw data into a usable DF '''
	full_data = load_breast_cancer()
	df = pd.DataFrame(
		{
			f: d
			for (f, d) in zip(
				[str(name) for name in full_data['feature_names']],
				full_data['data'].T,
				strict=False
			)
		}
	)
	df['target'] = full_data['target']  # 0=malignant, 1=benign]

	return df
# def load_data
