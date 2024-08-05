import pandas as pd
import logging

def load_data(filepath):
    try:
        data = pd.read_csv(filepath)
        return data
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def preprocess_data(data):
    try:
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.set_index('Date')
        return data
    except Exception as e:
        logging.error(f"Error preprocessing data: {e}")
        raise
