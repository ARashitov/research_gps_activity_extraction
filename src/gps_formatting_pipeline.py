import logging
from typing import Dict, Optional

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


class GPSDateTimeFormatter(BaseEstimator, TransformerMixin):
    """
    Drops junk records which exceeding # max_missing_tolerance
    """

    def __init__(
        self,
        colname: str = "datetime",
        input_datetime_format: str = "%Y/%m/%d (%H:%M:%S)",
    ):
        self.input_datetime_format = input_datetime_format
        self.colname = colname

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        try:
            X[self.colname] = pd.to_datetime(X[self.colname], format=self.input_datetime_format)
        except KeyError:
            logging.error(f"{self.colname} is missing")
        return X


class RoutePrimaryKeyGenerator(BaseEstimator, TransformerMixin):
    """
    Drops junk records which exceeding # max_missing_tolerance
    """

    def __init__(
        self,
        datetime_col: str = "datetime",
        plate_no_col: str = "plate_no",
        route_id_col: str = "route_id",
    ):
        self.datetime_col = datetime_col
        self.plate_no_col = plate_no_col
        self.route_id_col = route_id_col

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        date = X[self.datetime_col].dt.date.astype("str")
        vehicle_id = X[self.plate_no_col].astype(str)
        X[self.route_id_col] = date + " :: " + vehicle_id
        return X


class ColumnRenamer(BaseEstimator, TransformerMixin):
    """
    Drops junk records which exceeding # max_missing_tolerance
    """
    # flake8: noqa: B006

    def __init__(
        self,
        rename_params: Dict[str, str] = {
            "日期(時間)": "datetime",
            "Date(time)": "datetime",
            "車牌號碼\n(香港/國內) ": "plate_no",
            "Plate no.\n(HK/PRC)": "plate_no",
            "緯度": "lat",
            "經度": "lon",
            "Latitude": "lat",
            "Longitude": "lon",
        },
    ):
        self.rename_params = rename_params

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.rename(columns=self.rename_params)
        return X


def factory_raw_gps_formatter_pipeline() -> Pipeline:
    return Pipeline([
        ('column_renamer', ColumnRenamer()),
        ('datetime_formatter', GPSDateTimeFormatter(colname="datetime")),
        ('route_p_key_generator', RoutePrimaryKeyGenerator()),
    ])
