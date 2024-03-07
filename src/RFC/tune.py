# Execute a random grid search over the RFC model parameter space

import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from dotenv import load_dotenv
load_dotenv()
sys.path.append(os.environ['PROJECT_ROOT'] + '/src')
from RFC.utils import load_data, data_splits  # noqa: E402

parser = argparse.ArgumentParser(
	description="Execute hyperparameter tuning."
)
parser.add_argument(
	'--balanced', default=False, action='store_true',
	help='Tune a model which undersampled from the dominant class.'
)
args = parser.parse_args()


def make_parameter_grid():
	''' Define grid of parameters, to be randomly selected from in hyperparameter search. '''

	# General tree-based parameters
	random_grid = {
		'n_estimators': [int(x) for x in np.linspace(start=100, stop=1000, num=10)],
		'max_depth': [None] + [int(x) for x in np.linspace(5, 100, num=20)]
	}
	# RF-specific parameters
	random_grid['min_samples_split'] = \
		[2, 5, 10]  # Minimum number of samples required to split a node
	random_grid['min_samples_leaf'] = \
		[1, 2, 4]  # Minimum number of samples required at each leaf node
	random_grid['bootstrap'] = [True, False]

	return random_grid
# def make_parameter_grid


if __name__ == "__main__":
	# Load and split data
	df = load_data()

	# Undersample from dominant class if requested, to train on balanced data
	if args.balanced:
		# class 0 is the smaller class
		class0_df = df[df['target'] == 0]
		class1_df = df[df['target'] == 1].sample(
			n=len(df[df['target'] == 0]),
			replace=False,
			random_state=271828
		)
		df = pd.concat([class0_df, class1_df])

	(x_train, x_test, y_train, y_test) = data_splits(df, validate=False)

	# Define parameter grid
	param_grid = make_parameter_grid()

	print('Executing training search for RFC...')
	random_model = RandomizedSearchCV(
		estimator=RandomForestClassifier(),
		param_distributions=param_grid,
		n_iter=100,  # How many random parameter sets are tested?
		cv=3,  # How many cross-validation folds are used in each test?
		scoring='roc_auc',  # Which performance metric to use
		error_score='raise',  # Raise error if fitting/scoring fails
		verbose=3,  # Display scores, compute time, and parameters for each fold
		random_state=271828,
		n_jobs=-1  # Use all available processing
	)

	random_model.fit(x_train, y_train)

	print('Best parameters: ', random_model.best_params_)
# end main
