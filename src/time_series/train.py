import os
import sys
import subprocess
import pickle
from xgboost import XGBRegressor
from dotenv import load_dotenv
load_dotenv()
sys.path.append(os.environ['PROJECT_ROOT'] + '/src')
from time_series.utils import make_ts_data, engineer_features  # noqa: E402


if __name__ == "__main__":
	# Create dataset
	df = make_ts_data(t_step=0.01)

	# Add features which integrate learnings from explore_ts.ipynb (namely periodic behavior)
	df = engineer_features(df)

	# Only train on the first 80% of the historic data
	df = df[:int(len(df)*0.8)]

	print('Training XGB...')
	# These parameters have been optimized through `tune.py`
	xgb = XGBRegressor(
		n_estimators=100,
		learning_rate=0.03,
		max_depth=5,
		subsample=0.8,
		gamma=0.5,
		min_child_weight=5
	)
	xgb.fit(df.drop(columns=['value']), df['value'])

	print('Saving model to disk...')
	saved_model_dir = f"{os.environ['PROJECT_ROOT']}/models"
	subprocess.run(f'mkdir {saved_model_dir}'.split())
	pickle.dump(xgb, open(f'{saved_model_dir}/xgb.sav', 'wb'))
	print('Model has been trained and saved successfully!')
# end main
