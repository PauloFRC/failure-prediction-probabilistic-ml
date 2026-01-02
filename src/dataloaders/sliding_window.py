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
        
        self.window_indices = []
        
        for i, start in enumerate(self.window_starts):
            end = start + self.window_size
            
            # Binary search to find first row >= start time
            start_idx = np.searchsorted(timestamps, start, side='left')
            
            # Binary search to find first row >= end time
            end_idx = np.searchsorted(timestamps, end, side='left')
            
            # Store the row range for this window
            self.window_indices.append((start_idx, end_idx))
            
            if (i + 1) % 10000 == 0:
                print(f"Indexed {i + 1}/{len(self.window_starts)} windows...")
        
        print("Index built successfully!")
    
    def __len__(self):
        return len(self.window_starts)
    
    def __getitem__(self, idx):
        start_idx, end_idx = self.window_indices[idx]
        
        window_data = self.df[start_idx:end_idx]
        
        # Create count vector (initially all zeros)
        count_vector = np.zeros(self.n_events, dtype=np.float32)
        if len(window_data) > 0:
            # Count how many times each EventId appears in this window
            event_counts = (
                window_data
                .group_by('EventId')
                .agg(pl.len().alias('count'))
            )
            
            for row in event_counts.iter_rows(named=True):
                event_id = row['EventId']
                if event_id in self.event_to_idx:
                    idx_pos = self.event_to_idx[event_id]
                    count_vector[idx_pos] = row['count']
            
            # Label is anomaly if any failure event occurred in this window
            anomaly = window_data['Label'].any()
        else:
            anomaly = False 
        
        # Convert to PyTorch tensors
        count_tensor = torch.tensor(count_vector, dtype=torch.float32)
        label_tensor = torch.tensor(anomaly, dtype=torch.long)
        
        return count_tensor, label_tensor
