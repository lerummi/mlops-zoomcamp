#!/usr/bin/env python
# coding: utf-8

import sys
import pickle
import pandas as pd

from datetime import datetime


with open("model.bin", "rb") as f_in:
    dv, model = pickle.load(f_in)

categorical = ["PULocationID", "DOLocationID"]


def read_data(filename):
    df = pd.read_parquet(filename)

    df["duration"] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df["duration"] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype("int").astype("str")

    return df


def ride_duration_prediction(taxi_type: str, run_id: str, run_date: datetime = None):
    year = run_date.year
    month = run_date.month

    df = read_data(
        f"https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet"
    )
    df["ride_id"] = f"{year:04d}/{month:02d}_" + df.index.astype("str")

    dicts = df[categorical].to_dict(orient="records")
    X_val = dv.transform(dicts)
    df["pred"] = model.predict(X_val)

    print(f"Mean predicted duration: {df['pred'].mean()}")

    ride_id = f"{year:04d}/{month:02d}_" + df.index.astype("str")
    df_result = df[["ride_id", "pred"]]

    df_result.to_parquet(
        "data.parquet", engine="pyarrow", compression=None, index=False
    )


def run():
    taxi_type = sys.argv[1]
    year = int(sys.argv[2])
    month = int(sys.argv[3])

    ride_duration_prediction(
        taxi_type, run_id=None, run_date=datetime(year=year, month=month, day=1)
    )


if __name__ == "__main__":
    run()
