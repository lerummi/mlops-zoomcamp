#!/usr/bin/env python
# coding: utf-8

import sys
import pickle
import pandas as pd


def read_data(filename):
    categorical = ["PULocationID", "DOLocationID"]

    df = pd.read_parquet(filename)

    df["duration"] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df["duration"] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype("int").astype("str")

    return df


def run():
    taxi_type = sys.argv[1]
    year = int(sys.argv[2])
    month = int(sys.argv[3])

    with open("model.bin", "rb") as f_in:
        dv, model = pickle.load(f_in)

    df = read_data(
        "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2022-02.parquet"
    )
    df["ride_id"] = f"{year:04d}/{month:02d}_" + df.index.astype("str")

    dicts = df[categorical].to_dict(orient="records")
    X_val = dv.transform(dicts)
    df["pred"] = model.predict(X_val)

    ride_id = f"{year:04d}/{month:02d}_" + df.index.astype("str")
    df_result = df[["ride_id", "pred"]]

    df_result.to_parquet(
        "data.parquet", engine="pyarrow", compression=None, index=False
    )


if __name__ == "__main__":
    run()
