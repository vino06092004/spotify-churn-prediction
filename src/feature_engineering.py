import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Numerical features (original + engineered)
NUM_COLS = [
    'age',
    'listening_time',
    'songs_played_per_day',
    'skip_rate',
    'ads_listened_per_week',
    'skip_to_listening_ratio',
    'songs_per_hour'
]

# Categorical features
CAT_COLS = [
    'gender',
    'country',
    'subscription_type',
    'device_type',
    'offline_listening'
]

class SpotifyFeatureEngineer(BaseEstimator, TransformerMixin):
    """Engineers domain-specific engagement ratios for streaming data."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        
        # 1. Skip rate intensity relative to daily listening time
        df['skip_to_listening_ratio'] = df['skip_rate'] / (df['listening_time'] + 0.1)

        # 2. Listening density: songs played per hour of listening time
        df['songs_per_hour'] = df['songs_played_per_day'] / (df['listening_time'] + 0.1)

        return df

def build_spotify_preprocessor():
    """Builds the preprocessing ColumnTransformer."""
    return ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUM_COLS),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), CAT_COLS)
        ]
    )