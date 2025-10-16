# =========================================================================
# Copyright (C) 2024. The FuxiCTR Library. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================


import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
import pandas as pd
import polars as pl


class ParquetDataset(Dataset):
    def __init__(self, feature_map, data_path):
        self.feature_map = feature_map
        self.darray = self.load_data(data_path)
        
    def __getitem__(self, index):
        return self.darray[index, :]
    
    def __len__(self):
        return self.darray.shape[0]

    def load_data(self, data_path):
        df = pd.read_parquet(data_path)
        all_cols = list(self.feature_map.features.keys()) + self.feature_map.labels
        self.all_cols = all_cols
        self.df = df  # Save original DataFrame for cat_sample_select
        data_arrays = []
        for col in all_cols:
            if df[col].dtype == "object":
                array = np.array(df[col].to_list())
            else:
                array = df[col].to_numpy()
            data_arrays.append(array)
        return np.column_stack(data_arrays)

    def cat_sample_select(self, sample_ratio=1.0, user_col=None):
        """
        Aggregate data by user, select sample subset for each user, then merge to form new training set
        
        Args:
            sample_ratio: Sample ratio for each user (0-1), e.g. 0.7 means select 70% samples per user
            user_col: User column name, None for auto detection
            
        Returns:
            Resampled data array as np.column_stack(data_arrays)
        """
        # 1. Find user column index
        if user_col is None:
            # Auto detect user column
            possible_names = ['user', 'user_id', 'userid', 'uid']
            user_col = None
            for name in possible_names:
                if name in self.all_cols:
                    user_col = name
                    break
            
            if user_col is None:
                # Find column containing 'user'
                for col in self.all_cols:
                    if 'user' in col.lower():
                        user_col = col
                        break
            
            if user_col is None:
                raise ValueError("Cannot find user column. Please specify user_col parameter.")
        
        user_col_idx = self.all_cols.index(user_col)
        
        # 2. Aggregate by user
        user_ids = self.darray[:, user_col_idx]
        unique_users = np.unique(user_ids)
        
        # 3. Select sample subset for each user
        selected_parts = {i: [] for i in range(len(self.all_cols))}
        
        for user_id in unique_users:
            # Get all record indices for this user
            user_mask = user_ids == user_id
            user_indices = np.where(user_mask)[0]
            n = len(user_indices)
            
            # Select sample subset for this user
            select_count = int(n * sample_ratio)
            if select_count > 0:
                selected_indices = user_indices[:select_count]
                
                # Add selected data for this user
                for col_idx in range(len(self.all_cols)):
                    selected_parts[col_idx].append(self.darray[selected_indices, col_idx])
        
        # 4. Merge all users' parts and reorganize as np.column_stack(data_arrays)
        data_arrays = []
        for col_idx in range(len(self.all_cols)):
            if selected_parts[col_idx]:
                col_data = np.concatenate(selected_parts[col_idx], axis=0)
            else:
                col_data = np.array([])
            data_arrays.append(col_data)
        
        total_selected = len(data_arrays[0]) if len(data_arrays) > 0 and len(data_arrays[0]) > 0 else 0
        print(f"Selected {total_selected} samples from {len(unique_users)} users (ratio: {sample_ratio})")
        
        if total_selected == 0:
            return np.array([]).reshape(0, len(self.all_cols))
        
        return np.column_stack(data_arrays)



class ParquetDataLoader(DataLoader):
    def __init__(self, feature_map, data_path, batch_size=32, shuffle=False,
                 num_workers=1, **kwargs):
        if not data_path.endswith(".parquet"):
            data_path += ".parquet"
        self.dataset = ParquetDataset(feature_map, data_path)
        super().__init__(dataset=self.dataset, batch_size=batch_size,
                         shuffle=shuffle, num_workers=num_workers,
                         collate_fn=BatchCollator(feature_map))
        self.num_samples = len(self.dataset)
        self.num_blocks = 1
        self.num_batches = int(np.ceil(self.num_samples / self.batch_size))

    def __len__(self):
        return self.num_batches


class BatchCollator(object):
    def __init__(self, feature_map):
        self.feature_map = feature_map

    def __call__(self, batch):
        batch_tensor = default_collate(batch)
        all_cols = list(self.feature_map.features.keys()) + self.feature_map.labels
        batch_dict = dict()
        for col in all_cols:
            batch_dict[col] = batch_tensor[:, self.feature_map.get_column_index(col)]
        return batch_dict
