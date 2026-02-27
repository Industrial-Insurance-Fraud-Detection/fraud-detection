import pandas as pd
import numpy as np

class MaintenancePreprocessor:
    def __init__(self):
        self.type_mapping = {'L': 0, 'M': 1, 'H': 2}
        
    def preprocess(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        Main preprocessing pipeline.
        """
        df = df.copy()
        
        # 1. Drop UDI and Product ID
        cols_to_drop = ['UDI', 'Product ID']
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
        
        # 2. Encode Type column
        if 'Type' in df.columns:
            df['Type'] = df['Type'].map(self.type_mapping)
            
        # 3. Engineer new features
        # Mapping column names to match standard Kaggle names (handling potential space/K variants)
        df = self._rename_columns(df)
        
        df['Power'] = df['Torque'] * df['Rotational_speed']
        df['Temp_diff'] = df['Process_temperature'] - df['Air_temperature']
        
        # Avoid division by zero for Wear rate
        df['Wear_rate'] = df['Tool_wear'] / (df['Rotational_speed'] + 1e-9)
        
        if is_training and 'Failure_Type' in df.columns:
            df['Target_Class'] = df['Failure_Type'].apply(self._map_failure_types)
            
        return df

    def _rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names for easier manipulation.
        """
        rename_dict = {
            'Air temperature [K]': 'Air_temperature',
            'Process temperature [K]': 'Process_temperature',
            'Rotational speed [rpm]': 'Rotational_speed',
            'Torque [Nm]': 'Torque',
            'Tool wear [min]': 'Tool_wear',
            'Failure Type': 'Failure_Type'
        }
        return df.rename(columns=rename_dict)

    def _map_failure_types(self, failure_type: str) -> str:
        """
        Maps failure types to 4 required classes:
        No Failure / Random Failures (RNF) -> NORMAL_WEAR
        TWF / HDF / OSF / PWF (and full names) -> REAL_FAILURE
        """
        normal_types = ['No Failure', 'RNF', 'Random Failures']
        failure_types = [
            'TWF', 'Tool Wear Failure',
            'HDF', 'Heat Dissipation Failure',
            'OSF', 'Overstrain Failure',
            'PWF', 'Power Failure'
        ]
        
        if failure_type in normal_types:
            return 'NORMAL_WEAR'
        elif failure_type in failure_types:
            return 'REAL_FAILURE'
        return failure_type # Keeps FAKE/SABOTAGE if already present
