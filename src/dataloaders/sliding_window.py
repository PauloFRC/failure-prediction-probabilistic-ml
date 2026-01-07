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

class LogsSlidingWindow(Dataset):
    def __init__(
            self, 
            df: pl.DataFrame,
            window_size='5m', 
            step_size='1m', 
            n_prediction_window=1,
            event_ids=None, 
            filter_strategy='none', 
            filter_params=None,
            embedder=None,
            mode='count' # pptions: 'count', 'bert', 'hybrid'
        ):  
        self.filter_strategy = filter_strategy
        self.filter_params = filter_params or {}
        self.n_prediction_window = n_prediction_window
        self.embedder = embedder
        self.mode = mode

        if self.mode in ['bert', 'hybrid'] and self.embedder is None:
            raise ValueError(f"Mode '{self.mode}' requires an embedder object.")

        self.df = df.sort('Timestamp')
        print(f"Sorted {len(self.df)} rows by timestamp")

        self.mask_input_failures = True if filter_strategy in ['label', 'combined'] else False
        
        # create mapping of EventId to index for count vector
        if event_ids is None:
            self.event_ids = sorted(self.df['EventId'].unique().to_list())
        else:
            self.event_ids = event_ids

        # create mapping of failure types
        unique_failures = (
            self.df.filter(pl.col('Anomaly') == True)
            ['Label']
            .unique()
            .sort()
            .to_list()
        )        
        self.failure_ids = unique_failures
        self.failure_to_idx = {name: i for i, name in enumerate(self.failure_ids)}
        self.n_failure_types = len(self.failure_ids)
        print(f"Found {self.n_failure_types} failure types: {self.failure_ids}")
        
        # create a dictionary: EventId -> position in count vector
        self.event_to_idx = {e: i for i, e in enumerate(self.event_ids)}
        self.n_events = len(self.event_ids)
        print(f"Found {self.n_events} unique event types")

        if self.mode == 'bert':
            # BERT + intensity (1)
            self.input_dim = self.embedder.embedding_dim + 1
        elif self.mode == 'hybrid':
            # count + BERT
            self.input_dim = self.n_events + self.embedder.embedding_dim
        else:
            # just count
            self.input_dim = self.n_events

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
            contamination = self.filter_params.get('contamination', 0.01)
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

        count_tensor, label_tensor, prediction_type_tensor, failure_types_found = self._count_vectorize(window_data, future_data)
        
        return count_tensor, label_tensor, prediction_type_tensor, failure_types_found

    def _count_vectorize(self, window_data, future_data):
        if self.mask_input_failures and len(window_data) > 0:
            valid_logs = window_data.filter(pl.col('Anomaly') == False)
        else:
            valid_logs = window_data
            
        has_data = len(valid_logs) > 0

        if self.mode in ['count', 'hybrid']:
            count_vec = np.zeros(self.n_events, dtype=np.float32)
            if has_data:
                valid_logs = valid_logs.drop_nulls(subset=["EventIndex"])
                if len(valid_logs) > 0:
                    counts = valid_logs['EventIndex'].value_counts()
                    indices = counts['EventIndex'].to_numpy().astype(np.int64)
                    values = counts['count'].to_numpy().astype(np.float32)
                    count_vec[indices] = values
                count_vec = np.log1p(count_vec)

        if self.mode in ['bert', 'hybrid']:
            if has_data:
                counts_df = valid_logs['EventId'].value_counts()
                eids = counts_df['EventId'].to_list()
                counts = counts_df['count'].to_list()
                
                bert_vec = self.embedder.transform_window(eids, counts)
                
                if self.mode == 'bert':
                    total_logs = np.sum(counts)
                    intensity = np.log1p(total_logs)
                    bert_vec = np.concatenate([bert_vec, [intensity]])
            else:
                dim = self.embedder.embedding_dim + (1 if self.mode == 'bert' else 0)
                bert_vec = np.zeros(dim, dtype=np.float32)

        if self.mode == 'hybrid':
            final_vector = np.concatenate([count_vec, bert_vec])
        elif self.mode == 'bert':
            final_vector = bert_vec
        else:
            final_vector = count_vec

        count_tensor = torch.tensor(final_vector, dtype=torch.float32)
        
        is_current_anomalous = False
        if len(window_data) > 0:
            # label is anomaly if any failure event occurred in this window
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
              
        label_tensor = torch.tensor(1 if anomaly else 0, dtype=torch.long)
        prediction_type = torch.tensor(pred_type.value, dtype=torch.long)

        # get failure types found 
        failure_vec = np.zeros(self.n_failure_types, dtype=np.float32)
        current_failures = []
        if len(window_data) > 0:
            current_failures = (
                window_data.filter(pl.col("Anomaly") == True)
                ['Label']
                .unique()
                .to_list()
            )            
        future_failures = []
        if len(future_data) > 0:
            future_failures = (
                future_data.filter(pl.col("Anomaly") == True)
                ['Label']
                .unique()
                .to_list()
            )
        all_found_failures = set(current_failures + future_failures)
        for f_name in all_found_failures:
            if f_name in self.failure_to_idx:
                idx = self.failure_to_idx[f_name]
                failure_vec[idx] = 1.0
        failure_type_tensor = torch.tensor(failure_vec, dtype=torch.float32)
        
        return count_tensor, label_tensor, prediction_type, failure_type_tensor
    
    def _apply_label_filtering(self):
        original_len = len(self.df)
        self.df = self.df.filter(pl.col('Anomaly') == False)
        removed = original_len - len(self.df)
        print(f"Label filtering: Removed {removed} anomalous logs ({removed/original_len*100:.1f}%)")

    def _apply_isolation_forest_filtering(self, contamination=0.1):                
        # generate count vectors for all windows
        count_vectors = []
        for idx in range(len(self)):
            count_tensor, _, _, _ = self.__getitem__(idx)
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
    
    def get_failure_map(self):
        return {v: k for k, v in self.failure_to_idx.items()}
