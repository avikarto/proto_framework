# Make predictions using a saved trained model

import os
import sys
import subprocess
import pickle
from dotenv import load_dotenv
load_dotenv()
sys.path.append(os.environ['PROJECT_ROOT'] + '/src')
from time_series.utils import make_ts_data, engineer_features  # noqa: E402


if __name__ == "__main__":
	# Create dataset
	df = make_ts_data(t_step=0.01)

	# Add features which integrate learnings from explore_ts.ipynb (namely periodic behavior)
	df = engineer_features(df)

	# Predict on most-recent 20% of data
	df = df[int(len(df)*0.8):]

	xgb = pickle.load(open(f"{os.environ['PROJECT_ROOT']}/models/xgb.sav", 'rb'))

	predictions = xgb.predict(df.drop(columns=['value']))
	df['predictions'] = predictions

	subprocess.run(f"mkdir {os.environ['PROJECT_ROOT']}/predictions".split())
	outfile = f"{os.environ['PROJECT_ROOT']}/predictions/xgb_predict.csv"
	df.to_csv(outfile, index=False)
	print(f'Prediction output successfully written to {outfile}.')
# end main
