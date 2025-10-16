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
from itertools import chain
from torch.utils.data.dataloader import default_collate
from torch.utils.data import IterDataPipe, DataLoader, get_worker_info
import glob
import polars as pl
import pandas as pd
import os


class ParquetIterDataPipe(IterDataPipe):
    def __init__(self, data_blocks, feature_map, split="train", user_aggregate=False, 
                 min_user_records=100, train_ratio=0.8, valid_ratio=0.1):
        self.feature_map = feature_map
        self.data_blocks = data_blocks
        self.split = split
        self.user_aggregate = user_aggregate
        self.min_user_records = min_user_records
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio

    def load_data(self, data_path):
        df = pd.read_parquet(data_path)
        all_cols = list(self.feature_map.features.keys()) + self.feature_map.labels
        
        if self.user_aggregate and "user" in df.columns:
            # Filter users with records < min_user_records
            user_counts = df.groupby("user").size()
            valid_users = user_counts[user_counts >= self.min_user_records].index
            df = df[df["user"].isin(valid_users)]
            
            # Split data by user
            split_data = []
            for user, user_df in df.groupby("user"):
                n = len(user_df)
                train_end = int(n * self.train_ratio)
                valid_end = int(n * (self.train_ratio + self.valid_ratio))
                
                if self.split == "train":
                    user_split = user_df.iloc[:train_end]
                elif self.split == "valid":
                    user_split = user_df.iloc[train_end:valid_end]
                elif self.split == "test":
                    user_split = user_df.iloc[valid_end:]
                else:
                    user_split = user_df  # all data
                
                if len(user_split) > 0:
                    split_data.append(user_split)
            
            if len(split_data) == 0:
                return np.array([]).reshape(0, len(all_cols))
            
            df = pd.concat(split_data, ignore_index=True)
        
        data_arrays = []
        for col in all_cols:
            if df[col].dtype == "object":
                array = np.array(df[col].to_list())
            else:
                array = df[col].to_numpy()
            data_arrays.append(array)
        
        if len(data_arrays) == 0 or len(data_arrays[0]) == 0:
            return np.array([]).reshape(0, len(all_cols))
        
        return np.column_stack(data_arrays)

    def read_block(self, data_block):
        darray = self.load_data(data_block)
        for idx in range(darray.shape[0]):
            yield darray[idx, :]

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None: # single-process data loading
            block_list = self.data_blocks
        else: # in a worker process
            block_list = [
                block
                for idx, block in enumerate(self.data_blocks)
                if idx % worker_info.num_workers == worker_info.id
            ]
        return chain.from_iterable(map(self.read_block, block_list))


class ParquetBlockDataLoader(DataLoader):
    def __init__(self, feature_map, data_path, split="train", batch_size=32, shuffle=False,
                 num_workers=1, buffer_size=100000, user_aggregate=False, 
                 min_user_records=100, train_ratio=0.8, valid_ratio=0.1, **kwargs):
        """
        Args:
            feature_map: Feature map object
            data_path: Path to parquet files
            split: Data split ('train', 'valid', 'test')
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of data loading workers
            buffer_size: Shuffle buffer size
            user_aggregate: Whether to aggregate data by user and split per user
            min_user_records: Minimum number of records per user (default: 100)
            train_ratio: Training data ratio per user (default: 0.8)
            valid_ratio: Validation data ratio per user (default: 0.1)
        """
        if not data_path.endswith("parquet"):
            data_path += ".parquet"
        print(f"Absolute data_path: {os.path.abspath(data_path)}")
        data_blocks = sorted(glob.glob(data_path)) # sort by part name
        assert len(data_blocks) > 0, f"invalid data_path: {data_path}"
        self.data_blocks = data_blocks
        self.num_blocks = len(self.data_blocks)
        self.feature_map = feature_map
        self.batch_size = batch_size
        self.split = split
        self.user_aggregate = user_aggregate
        self.min_user_records = min_user_records
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.num_batches, self.num_samples = self.count_batches_and_samples()
        datapipe = ParquetIterDataPipe(self.data_blocks, feature_map, split=split,
                                       user_aggregate=user_aggregate,
                                       min_user_records=min_user_records,
                                       train_ratio=train_ratio,
                                       valid_ratio=valid_ratio)
        if shuffle:
            datapipe = datapipe.shuffle(buffer_size=buffer_size)
        elif split == "test":
            num_workers = 1 # multiple workers cannot keep the order of data reading
        super().__init__(dataset=datapipe,
                         batch_size=batch_size,
                         num_workers=num_workers,
                         collate_fn=BatchCollator(feature_map))

    def __len__(self):
        return self.num_batches

    def count_batches_and_samples(self):
        num_samples = 0
        for data_block in self.data_blocks:
            df_scan = pl.scan_parquet(data_block)
            
            if self.user_aggregate:
                # Load data to apply user filtering and splitting
                df = df_scan.collect()
                
                if "user" in df.columns:
                    # Filter users with records < min_user_records
                    user_counts = df.group_by("user").agg(pl.count().alias("count"))
                    valid_users = user_counts.filter(pl.col("count") >= self.min_user_records)["user"]
                    df = df.filter(pl.col("user").is_in(valid_users))
                    
                    # Count samples after split
                    for user in df["user"].unique():
                        user_df = df.filter(pl.col("user") == user)
                        n = len(user_df)
                        train_end = int(n * self.train_ratio)
                        valid_end = int(n * (self.train_ratio + self.valid_ratio))
                        
                        if self.split == "train":
                            num_samples += train_end
                        elif self.split == "valid":
                            num_samples += (valid_end - train_end)
                        elif self.split == "test":
                            num_samples += (n - valid_end)
                        else:
                            num_samples += n
                else:
                    num_samples += len(df)
            else:
                num_samples += df_scan.select(pl.count()).collect().item()
        
        num_batches = int(np.ceil(num_samples / self.batch_size)) if num_samples > 0 else 0
        return num_batches, num_samples


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
