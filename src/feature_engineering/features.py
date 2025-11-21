"""Feature engineering for fraud detection"""
import pandas as pd
from datetime import datetime

def create_features(df):
    """Generate fraud detection features"""
    # Temporal features
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
    
    # Amount-based features
    df['amount_log'] = df['amount'].apply(lambda x: pd.np.log(x + 1))
    df['amount_z_score'] = (df['amount'] - df['amount'].mean()) / df['amount'].std()
    
    # Velocity features (transactions per hour)
    df['txn_count_1h'] = df.groupby('card_id')['timestamp'].transform(
        lambda x: x.rolling('1H').count()
    )
    
    # Location features
    df['distance_from_last'] = calculate_distance(df)
    
    return df

def calculate_distance(df):
    """Calculate distance from previous transaction"""
    # Haversine formula implementation
    return 0  # Placeholder
