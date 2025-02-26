"""
Utility functions for data processing and handling.
"""
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset

class TabularDataProcessor:
    """
    Class to process tabular healthcare data for synthetic generation.
    Handles preprocessing, encoding, and batching of tabular data.
    """
    
    def __init__(self, 
                 categorical_columns: Optional[List[str]] = None,
                 numerical_columns: Optional[List[str]] = None,
                 target_column: Optional[str] = None,
                 normalization: str = 'standard',
                 missing_value_strategy: str = 'mean'):
        """
        Initialize the data processor.
        
        Args:
            categorical_columns: List of categorical column names
            numerical_columns: List of numerical column names
            target_column: Name of the target column (if any)
            normalization: Type of normalization ('standard', 'minmax', or None)
            missing_value_strategy: Strategy for missing values ('mean', 'median', 'mode', or 'drop')
        """
        self.categorical_columns = categorical_columns or []
        self.numerical_columns = numerical_columns or []
        self.target_column = target_column
        self.normalization = normalization
        self.missing_value_strategy = missing_value_strategy
        
        self.categorical_encoders = {}
        self.numerical_scaler = None
        self.column_order = []
        self.feature_dimensions = {}
        self.data_statistics = {}
    
    def fit(self, data: pd.DataFrame) -> 'TabularDataProcessor':
        """
        Fit preprocessing transformations on the data.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Self instance for chaining
        """
        # Auto-detect columns if not specified
        if not self.categorical_columns and not self.numerical_columns:
            self._auto_detect_column_types(data)
        
        self.column_order = self.numerical_columns + self.categorical_columns
        if self.target_column and self.target_column not in self.column_order:
            self.column_order.append(self.target_column)
        
        # Handle missing values
        data = self._handle_missing_values(data)
        
        # Fit encoders for categorical columns
        for col in self.categorical_columns:
            encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
            encoder.fit(data[[col]])
            self.categorical_encoders[col] = encoder
            self.feature_dimensions[col] = len(encoder.categories_[0])
            
            # Store category mapping for later decoding
            self.data_statistics[f"{col}_categories"] = encoder.categories_[0].tolist()
        
        # Fit scaler for numerical columns
        if self.numerical_columns:
            if self.normalization == 'standard':
                self.numerical_scaler = StandardScaler()
            elif self.normalization == 'minmax':
                self.numerical_scaler = MinMaxScaler()
            
            if self.normalization and self.numerical_scaler:
                self.numerical_scaler.fit(data[self.numerical_columns])
            
            # Store statistics for each numerical column
            for col in self.numerical_columns:
                self.feature_dimensions[col] = 1
                self.data_statistics[f"{col}_mean"] = float(data[col].mean())
                self.data_statistics[f"{col}_std"] = float(data[col].std())
                self.data_statistics[f"{col}_min"] = float(data[col].min())
                self.data_statistics[f"{col}_max"] = float(data[col].max())
        
        # Calculate total dimensions
        self.total_dimensions = sum(self.feature_dimensions.values())
        
        return self
    
    def transform(self, data: pd.DataFrame) -> np.ndarray:
        """
        Transform data using fitted preprocessing.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Transformed data as numpy array
        """
        # Handle missing values
        data = self._handle_missing_values(data)
        
        # Process numerical features
        numerical_data = None
        if self.numerical_columns:
            if self.normalization and self.numerical_scaler:
                numerical_data = self.numerical_scaler.transform(data[self.numerical_columns])
            else:
                numerical_data = data[self.numerical_columns].values
        
        # Process categorical features
        categorical_data_parts = []
        for col in self.categorical_columns:
            if col in data.columns:
                encoded = self.categorical_encoders[col].transform(data[[col]])
                categorical_data_parts.append(encoded)
        
        # Combine all features
        combined_parts = []
        if numerical_data is not None:
            combined_parts.append(numerical_data)
        if categorical_data_parts:
            combined_parts.extend(categorical_data_parts)
        
        if not combined_parts:
            return np.array([])
        
        return np.hstack(combined_parts)
    
    def inverse_transform(self, data: np.ndarray) -> pd.DataFrame:
        """
        Inverse transform processed data back to original format.
        
        Args:
            data: Transformed data
            
        Returns:
            DataFrame with original format
        """
        result_dict = {}
        current_pos = 0
        
        # Inverse transform numerical features
        if self.numerical_columns:
            num_features = len(self.numerical_columns)
            numerical_data = data[:, current_pos:current_pos + num_features]
            current_pos += num_features
            
            if self.normalization and self.numerical_scaler:
                numerical_data = self.numerical_scaler.inverse_transform(numerical_data)
            
            for i, col in enumerate(self.numerical_columns):
                result_dict[col] = numerical_data[:, i]
        
        # Inverse transform categorical features
        for col in self.categorical_columns:
            dim = self.feature_dimensions[col]
            col_data = data[:, current_pos:current_pos + dim]
            current_pos += dim
            
            # Convert one-hot back to categorical
            indices = np.argmax(col_data, axis=1)
            categories = self.data_statistics[f"{col}_categories"]
            result_dict[col] = [categories[idx] if idx < len(categories) else None for idx in indices]
        
        return pd.DataFrame(result_dict)
    
    def fit_transform(self, data: pd.DataFrame) -> np.ndarray:
        """
        Fit and transform in one step.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Transformed data
        """
        return self.fit(data).transform(data)
    
    def create_dataloader(self, 
                         data: Union[pd.DataFrame, np.ndarray], 
                         batch_size: int = 64, 
                         shuffle: bool = True) -> DataLoader:
        """
        Create a PyTorch DataLoader from data.
        
        Args:
            data: Input data (DataFrame or numpy array)
            batch_size: Batch size for DataLoader
            shuffle: Whether to shuffle the data
            
        Returns:
            PyTorch DataLoader
        """
        if isinstance(data, pd.DataFrame):
            data = self.transform(data)
        
        tensor_data = torch.tensor(data.astype(np.float32))
        dataset = TensorDataset(tensor_data)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def train_test_val_split(self, 
                            data: pd.DataFrame, 
                            train_size: float = 0.7, 
                            test_size: float = 0.15,
                            val_size: float = 0.15,
                            random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train, test, and validation sets.
        
        Args:
            data: Input DataFrame
            train_size: Proportion for training
            test_size: Proportion for testing
            val_size: Proportion for validation
            random_state: Random seed
            
        Returns:
            Tuple of (train_data, test_data, val_data) as numpy arrays
        """
        assert abs(train_size + test_size + val_size - 1.0) < 1e-10, "Split proportions must sum to 1"
        
        # First split off the test set
        relative_test_size = test_size / (test_size + train_size + val_size)
        train_val_data, test_data = train_test_split(
            data, test_size=relative_test_size, random_state=random_state
        )
        
        # Then split the train_val into train and validation
        relative_val_size = val_size / (train_size + val_size)
        train_data, val_data = train_test_split(
            train_val_data, test_size=relative_val_size, random_state=random_state
        )
        
        # Transform all sets
        return (
            self.transform(train_data),
            self.transform(test_data),
            self.transform(val_data)
        )
    
    def _auto_detect_column_types(self, data: pd.DataFrame) -> None:
        """
        Automatically detect numerical and categorical columns.
        
        Args:
            data: Input DataFrame
        """
        self.numerical_columns = []
        self.categorical_columns = []
        
        for col in data.columns:
            if col == self.target_column:
                continue
                
            if pd.api.types.is_numeric_dtype(data[col]):
                if data[col].nunique() < 10 and data[col].nunique() / len(data[col]) < 0.05:
                    self.categorical_columns.append(col)
                else:
                    self.numerical_columns.append(col)
            else:
                self.categorical_columns.append(col)
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values according to specified strategy.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with handled missing values
        """
        if self.missing_value_strategy == 'drop':
            return data.dropna()
        
        data_copy = data.copy()
        
        for col in self.numerical_columns:
            if data_copy[col].isnull().any():
                if self.missing_value_strategy == 'mean':
                    data_copy[col].fillna(data_copy[col].mean(), inplace=True)
                elif self.missing_value_strategy == 'median':
                    data_copy[col].fillna(data_copy[col].median(), inplace=True)
                elif self.missing_value_strategy == 'mode':
                    data_copy[col].fillna(data_copy[col].mode()[0], inplace=True)
        
        for col in self.categorical_columns:
            if data_copy[col].isnull().any():
                data_copy[col].fillna(data_copy[col].mode()[0], inplace=True)
                
        return data_copy
    
    def get_data_statistics(self) -> Dict[str, Any]:
        """
        Get statistics of the data.
        
        Returns:
            Dictionary of data statistics
        """
        return self.data_statistics
    
    def get_feature_dimensions(self) -> Dict[str, int]:
        """
        Get dimensions of each feature after transformation.
        
        Returns:
            Dictionary of feature dimensions
        """
        return self.feature_dimensions
    
    def get_total_dimensions(self) -> int:
        """
        Get total dimensions of transformed data.
        
        Returns:
            Total dimensions
        """
        return self.total_dimensions


def load_dataset(filepath: str, **kwargs) -> pd.DataFrame:
    """
    Load dataset from file.
    
    Args:
        filepath: Path to the dataset file
        **kwargs: Additional arguments to pass to pd.read_csv or pd.read_excel
        
    Returns:
        Loaded DataFrame
    """
    if filepath.endswith('.csv'):
        return pd.read_csv(filepath, **kwargs)
    elif filepath.endswith(('.xls', '.xlsx')):
        return pd.read_excel(filepath, **kwargs)
    elif filepath.endswith('.parquet'):
        return pd.read_parquet(filepath, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {filepath}")


def save_dataset(data: pd.DataFrame, filepath: str, **kwargs) -> None:
    """
    Save dataset to file.
    
    Args:
        data: DataFrame to save
        filepath: Path to save to
        **kwargs: Additional arguments to pass to DataFrame.to_csv or DataFrame.to_excel
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    if filepath.endswith('.csv'):
        data.to_csv(filepath, index=False, **kwargs)
    elif filepath.endswith(('.xls', '.xlsx')):
        data.to_excel(filepath, index=False, **kwargs)
    elif filepath.endswith('.parquet'):
        data.to_parquet(filepath, index=False, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {filepath}") 