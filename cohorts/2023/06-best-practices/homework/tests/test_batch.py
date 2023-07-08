import pytest
import pandas as pd
from datetime import datetime

from batch import prepare_data, CATEGORICAL


def dt(hour, minute, second=0):
    return datetime(2022, 1, 1, hour, minute, second)


@pytest.fixture
def input_dataframe():
    data = [
        (None, None, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2), dt(1, 10)),
        (1, 2, dt(2, 2), dt(2, 3)),
        (None, 1, dt(1, 2, 0), dt(1, 2, 50)),
        (2, 3, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),
    ]

    columns = [
        "PULocationID",
        "DOLocationID",
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
    ]
    return pd.DataFrame(data, columns=columns)


@pytest.fixture
def prepared_dataframe():
    data = [
        ("-1", "-1", dt(1, 2), dt(1, 10), 8.0),
        ("1", "-1", dt(1, 2), dt(1, 10), 8.0),
        ("1", "2", dt(2, 2), dt(2, 3), 1.0),
    ]

    columns = [
        "PULocationID",
        "DOLocationID",
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
        "duration",
    ]
    return pd.DataFrame(data, columns=columns)


def test_prepare_data(input_dataframe, prepared_dataframe):
    actual = prepare_data(input_dataframe)
    assert actual.equals(prepared_dataframe)
