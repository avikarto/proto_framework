# This utility handles the creation of a BQ query engine and the execution of SQL therein.

from google.cloud import bigquery
import pandas as pd

bq_client = bigquery.Client()


def run_query(q: str):
	""" Execute a query against BQ, wait for it to complete, and return the result.  NOT async.

	Accepts: q (str) - the query to execute
	"""

	return bq_client.query(q).result().to_dataframe()
# def run_query


def insert_df(df: pd.DataFrame, target: str, append: bool = False):
	""" Insert a DF to a GDO table.  NOT async.
	This creates the table if it does not already exist, for either mode of `append`.

	Accepts: df (DataFrame) - the data to upload
	Accepts: target (str) - the target project, schema, and table
	Accepts: append (bool) - By default, truncate and insert.  Can override to append instead.
	"""

	# Ensure project, schema, and table are defined in the target
	assert len(target.split('.')) == 3

	job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE") if not append \
		else bigquery.LoadJobConfig()

	job = bq_client.load_table_from_dataframe(df, target, job_config=job_config)
	job.result()  # Wait for the job to complete.
# def insert_df


def df_from_bucket_csv(bucket: str, blob: str) -> pd.DataFrame:
	""" Extract CSV blob from a GCS bucket and return contents as a dataframe

	Accepts: bucket (str) - the bucket name in GCS
	Accepts: blob (str) - the name of the CSV file (including .csv)
	"""

	return pd.read_csv(f'gs://{bucket}/{blob}')
# def df_from_bucket_csv
