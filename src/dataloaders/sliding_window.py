import torch
from torch.utils.data import Dataset, DataLoader
import polars as pl
import numpy as np
from datetime import timedelta
import re


class LogsSlidingWindow(Dataset):
    def __init__(self, df: pl.DataFrame, window_size='5m', step_size='1m', event_ids=None):        
        self.df = df.sort('Timestamp')
        print(f"Sorted {len(self.df)} rows by timestamp")
        
        # Create mapping of EventId to index for count vector
        if event_ids is None:
            self.event_ids = sorted(self.df['EventId'].unique().to_list())
        else:
            self.event_ids = event_ids
        
        # Create a dictionary: EventId -> position in count vector
        self.event_to_idx = {e: i for i, e in enumerate(self.event_ids)}
        self.n_events = len(self.event_ids)
        print(f"Found {self.n_events} unique event types")

        # Create mapping column in DataFrame
        mapping_df = pl.DataFrame({
            "EventId": self.event_ids, 
            "EventIndex": range(len(self.event_ids))
        })
        self.df = self.df.join(mapping_df, on="EventId", how="left")
        
        self.window_size = self._parse_duration(window_size)
        self.step_size = self._parse_duration(step_size)
        
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
        print(f"Window size: {window_size}, Step size: {step_size}")
        
        # Instead of filtering the DataFrame each time, store row ranges
        self._build_index()
    
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

        self.window_indices = np.column_stack((start_idxs, end_idxs))
        
        print(f"Index built. Shape: {self.window_indices.shape}")
    
    def __len__(self):
        return len(self.window_starts)
    
    def __getitem__(self, idx):
        start_idx, end_idx = self.window_indices[idx]
        
        window_data = self.df[start_idx:end_idx]
        
        # Create count vector (initially all zeros)
        count_vector = np.zeros(self.n_events, dtype=np.float32)
        anomaly = False

        if len(window_data) > 0:
            counts = window_data['EventIndex'].value_counts()

            indices = counts['EventIndex'].to_numpy()
            values = counts['count'].to_numpy()

            count_vector[indices] = values
            
             # Apply log normalization | TODO: explicar ou validar se é o melhor método
            count_vector = np.log1p(count_vector)
            
            # Label is anomaly if any failure event occurred in this window
            anomaly = window_data['Label'].any()
        
        # Convert to PyTorch tensors
        count_tensor = torch.tensor(count_vector, dtype=torch.float32)
        label_tensor = torch.tensor(anomaly, dtype=torch.long)
        
        return count_tensor, label_tensor
