import pandas as pd
import numpy as np
from models.classification.preprocessor import MaintenancePreprocessor

def test_preprocessing_logic():
    preprocessor = MaintenancePreprocessor()
    
    # Mock input data
    data = {
        'UDI': [1],
        'Product ID': ['L12345'],
        'Type': ['L'],
        'Air temperature [K]': [300.0],
        'Process temperature [K]': [310.0],
        'Rotational speed [rpm]': [1500.0],
        'Torque [Nm]': [40.0],
        'Tool wear [min]': [50],
        'Failure Type': ['No Failure']
    }
    df = pd.DataFrame(data)
    
    # Process
    processed_df = preprocessor.preprocess(df)
    
    # Assertions
    assert 'UDI' not in processed_df.columns
    assert 'Product ID' not in processed_df.columns
    assert processed_df['Type'].iloc[0] == 0
    assert 'Power' in processed_df.columns
    assert 'Temp_diff' in processed_df.columns
    assert 'Wear_rate' in processed_df.columns
    assert processed_df['Target_Class'].iloc[0] == 'NORMAL_WEAR'
    
    print("Preprocessor tests passed!")

if __name__ == "__main__":
    test_preprocessing_logic()
