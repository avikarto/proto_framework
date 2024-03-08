# Various helper functions for modeling activities

import os
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from random import random, seed
from scipy.fft import fft, fftfreq
from dotenv import load_dotenv
load_dotenv()
sys.path.append(os.environ['PROJECT_ROOT'] + '/src')


def make_ts_data(t_step: float, plot: bool = False) -> pd.DataFrame:
	''' Create raw data with perodicity and random noise
	-----
	Accepts: t_step - time step for the temporal grid
	Returns: df - dataframe with time series data (x=time, y=value)
	'''

	t_step = 0.01
	domain = np.arange(0, 80, t_step)
	omega1 = 0.4
	omega2 = 0.1
	seed(271828)  # set random seed to always use the same dataset
	df = pd.DataFrame({
		'time': [t for t in domain],
		'value': [
			np.sin(2*np.pi*omega1*t)+(random()-0.5) + np.sin(2*np.pi*omega2*t)+(random()-0.5)
			for t in domain
		]
	})

	if plot:
		plt.scatter(df['time'], df['value'], s=3)
		plt.show()

	return df
# def load_data


def add_lag_feature(df, n):
	''' Create a data-lag feature, shifting the raw data by `n` steps '''

	df[f'lag_{n}'] = df['value'].shift(n)
	return df
# def add_lag_feature


def add_rolling_feature(df, n):
	''' Create a rolling-mean feature with moving window of size `n` points '''

	df[f'rolling_{n}'] = df['value'].rolling(window=n).mean()
	return df
# def add_rolling_feature


def get_frequencies(df: pd.DataFrame, target: str, n: int, t_step: float) -> None:
	''' Extract signal frequencies from data, plot the frequency spectrum, and show n frequencies
	https://docs.scipy.org/doc/scipy/tutorial/fft.html
	-----
	Accepts: df - the real data
	Accepts: target - the target column in the data
	Accepts: n - the number of signals to attempt to extract
	'''

	assert n >= 1 and type(n) is int, "Invalid selection for n"

	# discard negative frequency components for this analysis (the //2 does this)
	y_freq = fft(df[target].values)[0:len(df)//2]  # get frequency-space signal from data
	y_freq = np.abs(y_freq)
	x_freq = fftfreq(n=len(df), d=t_step)[:len(df)//2]  # define frequency spectrum. units: 1/t_step

	# Prepare a plot of the frequency-space
	plt.plot(x_freq, 2.0/(2*len(x_freq)) * y_freq)

	# Find strongest signal
	max_inds = [list(y_freq).index(max(y_freq))]
	max_freqs = [x_freq[max_inds[0]]]

	# If n>1, find more signals
	for ns in range(2, n+1):
		for ind in max_inds:  # remove previous maxes from consideration of a new max
			y_freq[ind] = 0
		max_inds.append(list(y_freq).index(max(y_freq)))
		max_freqs.append(x_freq[max_inds[ns-1]])
	# for i

	print(f'The {n} strongest signals had frequencies {max_freqs}')
	plt.show()
# def get_frequencies


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
	''' Add features to raw data '''

	# Add periodic features to data, as identified from fourier analysis
	omega_1, omega_2 = 0.4, 0.1
	df['periodic_1'] = df['time'].apply(lambda x: 2*np.pi*omega_1*np.sin(x))
	df['periodic_2'] = df['time'].apply(lambda x: 2*np.pi*omega_2*np.sin(x))

	# Add a rolling mean feature across each 3 adjacent data points
	df = add_rolling_feature(df, 3)

	# Add a temporal lag feature for 1 time step
	df = add_lag_feature(df, 1)

	return df
# def engineer_features
