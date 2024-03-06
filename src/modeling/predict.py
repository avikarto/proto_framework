# Make predictions using a saved trained model

import os
import sys
import subprocess
import argparse
import pickle
from sklearn.metrics import confusion_matrix
from dotenv import load_dotenv
load_dotenv()
sys.path.append(os.environ['PROJECT_ROOT'] + '/src')
from modeling.utils import load_data, data_splits  # noqa: E402

parser = argparse.ArgumentParser(
	description="Make predictions from a saved model."
)
parser.add_argument(
	'--balanced', default=False, action='store_true',
	help='Predict from a model which undersampled from the dominant class.'
)
args = parser.parse_args()


if __name__ == "__main__":
	# Load data to predict on
	raw_data = load_data()

	filter = 'balanced' if args.balanced else 'all'

	rfc = pickle.load(open(f"{os.environ['PROJECT_ROOT']}/models/rfc_{filter}.sav", 'rb'))

	_, x_test, _, y_test = data_splits(raw_data)
	out_df = x_test.copy()

	preds = rfc.predict(x_test)
	out_df['pred'] = preds

	# probabilities for predictions
	probs = rfc.predict_proba(x_test)
	out_df['prob'] = [round(probs[i][preds[i]], 3) for i in range(len(probs))]

	# Show some model performance metrics
	importances = rfc.feature_importances_
	model_features = list(raw_data.columns)
	print(
		'Feature importances:\n',
		{model_features[i]: round(importances[i], 3) for i in range(len(importances))},
		'\n'
	)
	tn, fp, fn, tp = confusion_matrix(y_test, out_df['pred']).ravel()
	print(f'TP={tp}, FP={fp}, TN={tn}, FN={fn}')
	print(f'Accuracy: {(tp + tn) / len(out_df)}')
	print(f'Sensitivity (TP; true churn): {tp / (tp + fn)}')
	print(f'Specificity (TN; true retention): {tn / (tn + fp)}')

	subprocess.run(f"mkdir {os.environ['PROJECT_ROOT']}/predictions".split())
	outfile = f"{os.environ['PROJECT_ROOT']}/predictions/predict_{filter}.csv"
	out_df.to_csv(outfile, index=False)
	print(f'Prediction output successfully written to {outfile}.')
