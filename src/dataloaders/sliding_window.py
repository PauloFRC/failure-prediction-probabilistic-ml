import torch
from torch.utils.data import Dataset, DataLoader
import polars as pl
import numpy as np
from datetime import timedelta
import re
from sklearn.ensemble import IsolationForest
from enum import Enum

class PredictionType(Enum):
    NORMAL = 0
    CLEAR_FAILURE = 1
    PRE_FAILURE = 2

# TODO: adicionar filtro temporal (x horas pré-falha)
class LogsSlidingWindow(Dataset):
    def __init__(
            self, 
            df: pl.DataFrame, 
            window_size='5m', 
            step_size='1m', 
            n_prediction_window=1,
            event_ids=None, 
            filter_strategy='none', 
            filter_params=None
            ):  
        self.filter_strategy = filter_strategy
        self.filter_params = filter_params or {}
        self.n_prediction_window = n_prediction_window

        self.df = df.sort('Timestamp')
        print(f"Sorted {len(self.df)} rows by timestamp")

        self.mask_input_failures = True if filter_strategy in ['label', 'combined'] else False
        
        # create mapping of EventId to index for count vector
        if event_ids is None:
            self.event_ids = sorted(self.df['EventId'].unique().to_list())
        else:
            self.event_ids = event_ids
        
        # create a dictionary: EventId -> position in count vector
        self.event_to_idx = {e: i for i, e in enumerate(self.event_ids)}
        self.n_events = len(self.event_ids)
        print(f"Found {self.n_events} unique event types")

        # create mapping column in DataFrame
        mapping_df = pl.DataFrame({
            "EventId": self.event_ids, 
            "EventIndex": np.arange(self.n_events, dtype=np.int32)
        })

        self.df = self.df.join(mapping_df, on="EventId", how="left")
        
        self.window_size = self._parse_duration(window_size)
        self.step_size = self._parse_duration(step_size)

        # calculate prediction horizon
        self.prediction_horizon = self.step_size * self.n_prediction_window
        print(f"Prediction Horizon: +{self.prediction_horizon} beyond current window")
        print(f"Checking for failures in the next {n_prediction_window} sliding windows.")
        
        start_time = self.df['Timestamp'].min()
        end_time = self.df['Timestamp'].max()

        print(f'Period -> Start time: {start_time}, End time: {end_time}')
        
        self.window_starts = []
        current = start_time
        
        # Create windows until we can't fit another full window
        while current + self.window_size <= end_time:
            self.window_starts.append(current)
            current += self.step_size
        
        print(f"Generated {len(self.window_starts)} sliding windows")
        
        # build indices
        self._build_index()

        if filter_strategy in ['isolation_forest', 'combined']:
            contamination = self.filter_params.get('contamination', 0.05)
            self._apply_isolation_forest_filtering(contamination)
    
    def _parse_duration(self, duration_str):
        match = re.match(r'(\d+)([smhd])', duration_str)
        if not match:
            raise ValueError(f"Invalid duration format: {duration_str}. Use format like '5m', '30s', '2h'")
        
        value = int(match.group(1))
        unit = match.group(2)
        
        units = {
            's': 'seconds',
            'm': 'minutes', 
            'h': 'hours',
            'd': 'days'
        }
        
        return timedelta(**{units[unit]: value})
    
    def _build_index(self):
        print("Building index...")        

        timestamps = self.df['Timestamp'].to_numpy()
        
        starts = np.array(self.window_starts)
        
        ends = starts + self.window_size
        
        start_idxs = np.searchsorted(timestamps, starts, side='left')
        end_idxs = np.searchsorted(timestamps, ends, side='left')

        pred_ends = ends + self.prediction_horizon
        pred_end_idxs = np.searchsorted(timestamps, pred_ends, side='left')

        self.window_indices = np.column_stack((start_idxs, end_idxs, pred_end_idxs))
        
        print(f"Index built. Shape: {self.window_indices.shape}")
    
    def __len__(self):
        return len(self.window_starts)
    
    def __getitem__(self, idx):
        start_idx, end_idx, pred_end_idx = self.window_indices[idx]
        
        window_data = self.df[start_idx:end_idx]

        future_data = self.df[end_idx:pred_end_idx]

        count_tensor, label_tensor, prediction_type_tensor = self._count_vectorize(window_data, future_data)
        
        return count_tensor, label_tensor, prediction_type_tensor

    def _count_vectorize(self, window_data, future_data):
        # create count vector
        count_vector = np.zeros(self.n_events, dtype=np.float32)

        if len(window_data) > 0:
            if self.mask_input_failures:
                valid_logs = window_data.filter(pl.col('Anomaly') == False)
            else:
                valid_logs = window_data

            valid_logs = valid_logs.drop_nulls(subset=["EventIndex"])

            if len(valid_logs) > 0:
                counts = valid_logs['EventIndex'].value_counts()
                indices = counts['EventIndex'].to_numpy().astype(np.int64)
                values = counts['count'].to_numpy().astype(np.float32)

                count_vector[indices] = values

            # apply log normalization | TODO: explicar ou validar se é o melhor método
            count_vector = np.log1p(count_vector)
            
            # label is anomaly if any failure event occurred in this window
            anomaly = window_data.select(pl.col("Anomaly").any()).item()
        
        is_current_anomalous = False
        if len(window_data) > 0:
            is_current_anomalous = window_data.select(pl.col("Anomaly").any()).item()

        is_future_anomalous = False
        if len(future_data) > 0:
            is_future_anomalous = future_data.select(pl.col("Anomaly").any()).item()

        if is_current_anomalous:
            pred_type = PredictionType.CLEAR_FAILURE
            anomaly = True
        elif is_future_anomalous:
            pred_type = PredictionType.PRE_FAILURE
            anomaly = True
        else:
            pred_type = PredictionType.NORMAL
            anomaly = False
              
        count_tensor = torch.tensor(count_vector, dtype=torch.float32)
        label_tensor = torch.tensor(1 if anomaly else 0, dtype=torch.long)
        prediction_type = torch.tensor(pred_type.value, dtype=torch.long)
        
        return count_tensor, label_tensor, prediction_type
    
    def _apply_label_filtering(self):
        original_len = len(self.df)
        self.df = self.df.filter(pl.col('Anomaly') == False)
        removed = original_len - len(self.df)
        print(f"Label filtering: Removed {removed} anomalous logs ({removed/original_len*100:.1f}%)")

    def _apply_isolation_forest_filtering(self, contamination=0.1):                
        # generate count vectors for all windows
        count_vectors = []
        for idx in range(len(self)):
            count_tensor, _, _ = self.__getitem__(idx)
            count_vectors.append(count_tensor.numpy())
        
        X = np.array(count_vectors)
        
        iso_forest = IsolationForest(
            contamination=contamination, 
            random_state=42,
            n_jobs=-1
        )
        predictions = iso_forest.fit_predict(X)
        
        # keep only normal windows
        normal_mask = predictions == 1
        self.window_starts = [w for i, w in enumerate(self.window_starts) if normal_mask[i]]
        self.window_indices = self.window_indices[normal_mask]
        
        removed = (~normal_mask).sum()
        print(f"Isolation Forest: Removed {removed} windows ({removed/len(normal_mask)*100:.1f}%)")
        print(f"Remaining windows: {len(self.window_starts)}")
