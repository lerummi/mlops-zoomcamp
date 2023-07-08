#!/usr/bin/env python
# coding: utf-8

import os
import sys
import pickle
import pandas as pd
from typing import List

os.environ["AWS_ACCESS_KEY_ID"] = "admin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "admin"

CATEGORICAL = ["PULocationID", "DOLocationID"]

# INPUT_FILE_PATTERN = "s3://nyc-duration/in/{year:04d}-{month:02d}.parquet"
# os.environ["INPUT_FILE_PATTERN"] = INPUT_FILE_PATTERN
OUTPUT_FILE_PATTERN = "s3://nyc-duration/out/{year:04d}-{month:02d}.parquet"
os.environ["OUTPUT_FILE_PATTERN"] = OUTPUT_FILE_PATTERN
S3_ENDPOINT_URL = "http://localhost:4566"


LOCAL_OPTIONS = {"client_kwargs": {"endpoint_url": S3_ENDPOINT_URL}}


def get_input_path(year, month):
    default_input_pattern = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet"
    input_pattern = os.getenv("INPUT_FILE_PATTERN", default_input_pattern)
    print(input_pattern.format(year=year, month=month))
    return input_pattern.format(year=year, month=month)


def get_output_path(year, month):
    default_output_pattern = "s3://nyc-duration/taxi_type=fhv/year={year:04d}/month={month:02d}/predictions.parquet"
    output_pattern = os.getenv("OUTPUT_FILE_PATTERN", default_output_pattern)
    return output_pattern.format(year=year, month=month)


def read_data(filename: str, options=None) -> pd.DataFrame:
    return pd.read_parquet(filename, storage_options=options)


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df["duration"] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df["duration"] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[CATEGORICAL] = df[CATEGORICAL].fillna(-1).astype("int").astype("str")

    return df


def main(year: int, month: int):
    with open("model.bin", "rb") as f_in:
        dv, lr = pickle.load(f_in)

    input_file = get_input_path(year, month)
    output_file = get_output_path(year, month)

    df = read_data(input_file)
    df.to_parquet(
        "s3://nyc-duration/in/here.parquet",
        engine="pyarrow",
        compression=None,
        index=False,
        storage_options=LOCAL_OPTIONS,
    )

    df = prepare_data(df)
    df["ride_id"] = f"{year:04d}/{month:02d}_" + df.index.astype("str")

    dicts = df[CATEGORICAL].to_dict(orient="records")
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    print("predicted mean duration:", y_pred.mean())

    df_result = pd.DataFrame()
    df_result["ride_id"] = df["ride_id"]
    df_result["predicted_duration"] = y_pred

    df_result.to_parquet(
        output_file, engine="pyarrow", index=False, storage_options=LOCAL_OPTIONS
    )


if __name__ == "__main__":
    year = int(sys.argv[1])
    month = int(sys.argv[2])

    main(year, month)
