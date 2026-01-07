from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import torch

class BertLogEmbedder:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.embedding_map = {}
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def fit(self, df):
        unique_logs = df.select(['EventId', 'Content']).unique()
        
        ids = unique_logs['EventId'].to_list()
        templates = unique_logs['Content'].to_list()
        
        print(f"Encoding {len(ids)} unique Drain templates...")
        
        embeddings = self.model.encode(templates, batch_size=64, show_progress_bar=True)
        
        self.embedding_map = {
            eid: emb for eid, emb in zip(ids, embeddings)
        }
        
        print(f"Embeddings ready. Dimension: {self.embedding_dim}")
        return self

    def transform_window(self, event_ids, counts):
        if len(event_ids) == 0:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        vectors = []
        valid_counts = []
        
        for eid, count in zip(event_ids, counts):
            if eid in self.embedding_map:
                vectors.append(self.embedding_map[eid])
                valid_counts.append(count)
            else:
                pass
        
        if len(vectors) == 0:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        vectors = np.array(vectors)
        weights = np.array(valid_counts).reshape(-1, 1)
        
        weighted_sum = np.sum(vectors * weights, axis=0)
        total_count = np.sum(weights)
        
        avg_vector = weighted_sum / total_count if total_count > 0 else weighted_sum
        
        return avg_vector