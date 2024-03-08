import os
import sys
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from dotenv import load_dotenv
load_dotenv()
sys.path.append(os.environ['PROJECT_ROOT'] + '/src')
from time_series.utils import make_ts_data, engineer_features  # noqa: E402


def make_parameter_grid():
	''' Define grid of parameters, to be randomly selected from in hyperparameter search. '''

	# General tree-based parameters
	grid = {
		'max_depth': [None] + [int(x) for x in np.linspace(5, 10, num=6)],
		'n_estimators': [int(x) for x in np.linspace(10, 100, num=10)]
	}
	# XGB-specific parameters
	grid['learning_rate'] = [round(x, 2) for x in np.linspace(.01, .03, num=3)]
	grid['gamma'] = [0, 0.1, 0.5, 1]
	grid['min_child_weight'] = \
		[x for x in range(1, 6)]  # Minimum sum of instance weight(hessian) needed in a child.
	grid['subsample'] = \
		[0.8, 0.9, 1.0]  # Subsample ratio; 0.5 means sample half of data prior to growing trees.

	return grid
# def make_parameter_grid


if __name__ == '__main__':
	# Create dataset
	df = make_ts_data(t_step=0.01)

	# Add features which integrate learnings from explore_ts.ipynb (namely periodic behavior)
	df = engineer_features(df)

	# Only tune on the first 50% of the historic data
	df = df[:len(df)//2]

	# Define parameter grid
	param_grid = make_parameter_grid()

	print('Executing training search for XGB...')
	grid_model = GridSearchCV(
		estimator=XGBRegressor(),
		param_grid=param_grid,
		cv=3,  # How many cross-validation folds are used in each test?  (sequential; KFold)
		error_score='raise',  # Raise error if fitting/scoring fails
		verbose=3,  # Display scores, compute time, and parameters for each fold
		n_jobs=-1  # Use all available processing
	)

	grid_model.fit(df.drop(columns=['value']), df['value'])
	print('Best parameters: ', grid_model.best_params_)
# end main
