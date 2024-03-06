# Train a RFC model, with or without undersampling for imbalanced classes

import os
import sys
import argparse
import subprocess
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from dotenv import load_dotenv
load_dotenv()
sys.path.append(os.environ['PROJECT_ROOT'] + '/src')
from modeling.utils import load_data, data_splits  # noqa: E402

parser = argparse.ArgumentParser(
	description="Train model, with or without undersampling (support for imbalanced data)."
)
parser.add_argument(
	'--balanced', default=False, action='store_true',
	help='Create a model which undersamples from the dominant class.'
)
args = parser.parse_args()


if __name__ == "__main__":
	# Load raw data
	df = load_data()

	saved_model_dir = f"{os.environ['PROJECT_ROOT']}/models"
	subprocess.run(f'mkdir {saved_model_dir}'.split())

	# Train and save models
	print('Starting training loop...')

	# Undersample from dominant class if requested, to train on balanced data
	if args.balanced:
		print('Creating understampled dataset...')
		# class 0 is the smaller class
		class0_df = df[df['target'] == 0]
		class1_df = df[df['target'] == 1].sample(
			n=len(df[df['target'] == 0]),
			replace=False,
			random_state=271828
		)
		df = pd.concat([class0_df, class1_df])
		filter = 'balanced'
	else:
		filter = 'all'
	# end if balanced

	# create training subset
	x_train, _, y_train, _ = data_splits(df)

	print('Training RFC...')
	# These parameters have been optimized through `tune.py`
	rfc = RandomForestClassifier(
		n_estimators=400,
		max_depth=35,
		bootstrap=False,
		min_samples_leaf=1,
		min_samples_split=2
	)
	rfc.fit(x_train, y_train)

	print('Saving model to disk...')
	pickle.dump(rfc, open(f'{saved_model_dir}/rfc_{filter}.sav', 'wb'))
# main
