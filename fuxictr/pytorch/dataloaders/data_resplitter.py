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
import pandas as pd
import os
import logging


class DatasetReSplitter(object):
    """
    Dataset merging and re-splitting utility class
    
    Features:
    1. Merge train, validation and test sets
    2. Re-split data by specified ratios
    3. Return re-split data
    
    Note: user_split mode implementation references the user_aggregate logic in parquet_block_dataloader.py
    
    Example:
        # Mode 1: Split within each user, then merge (suitable for sequential data)
        splitter = DatasetReSplitter(
            feature_map=feature_map,
            train_path='./data/train.npz',
            valid_path='./data/valid.npz', 
            test_path='./data/test.npz',
            data_format='npz',
            split_mode='user_split',  # Split within each user
            user_col='user_id',       # User column name (optional, auto-detected)
            min_records=5             # Filter users with less than 5 records
        )
        
        train_data, valid_data, test_data = splitter.split(
            split_ratios=[0.7, 0.15, 0.15],
            random_seed=2024
        )
        
        # Mode 2: Group by user to avoid user-level data leakage
        splitter = DatasetReSplitter(
            feature_map=feature_map,
            train_path='./data/train.npz',
            valid_path='./data/valid.npz', 
            test_path='./data/test.npz',
            data_format='npz',
            split_mode='user_group'  # Group by user
        )
    """
    
    def __init__(self, feature_map, train_path=None, valid_path=None, test_path=None, 
                 data_format='npz', split_mode='user_group', user_col=None, min_records=5):
        """
        Initialize dataset re-splitter
        
        Args:
            feature_map: Feature map object
            train_path: Training set path
            valid_path: Validation set path
            test_path: Test set path
            data_format: Data format, 'npz' or 'parquet'
            split_mode: Split mode
                - 'user_group': Group by user, all data of each user in same set (avoid user-level leakage)
                - 'user_split': Split within each user, then merge (suitable for sequential data)
                - 'random': Random sampling
            user_col: User column name, None for auto detection
            min_records: Minimum records per user, users with fewer records will be filtered
        """
        self.feature_map = feature_map
        self.train_path = train_path
        self.valid_path = valid_path
        self.test_path = test_path
        self.data_format = data_format
        self.split_mode = split_mode
        self.user_col = user_col
        self.min_records = min_records
        self.merged_data = None
        
        # Auto identify user column
        if self.split_mode in ['user_group', 'user_split'] and self.user_col is None:
            self.user_col = self._identify_user_column()
    
    def _identify_user_column(self):
        """Auto identify user ID column"""
        # Common user ID column names
        possible_names = ['user_id', 'userid', 'user', 'uid', 'UserID', 'UserId']
        
        all_cols = list(self.feature_map.features.keys())
        
        # Priority: exact match
        for name in possible_names:
            if name in all_cols:
                logging.info(f"Auto-identified user column: {name}")
                return name
        
        # Find column containing 'user'
        for col in all_cols:
            if 'user' in col.lower():
                logging.info(f"Auto-identified user column: {col}")
                return col
        
        # If not found, use first column (usually user ID)
        if len(all_cols) > 0:
            logging.warning(f"User column not found, using first column: {all_cols[0]}")
            return all_cols[0]
        
        raise ValueError("Cannot identify user column. Please specify user_col parameter.")
        
    def merge(self):
        """
        Merge all datasets
        
        Returns:
            merged_data: Merged data (dict or DataFrame)
        """
        logging.info("Merging datasets...")
        
        if self.data_format == 'npz':
            self.merged_data = self._merge_npz()
        elif self.data_format in ['parquet', 'csv']:
            self.merged_data = self._merge_parquet()
        else:
            raise ValueError(f"Unsupported data format: {self.data_format}")
        
        if self.data_format == 'npz':
            total_samples = len(self.merged_data[list(self.merged_data.keys())[0]])
        else:
            total_samples = len(self.merged_data)
            
        logging.info(f"Total merged samples: {total_samples}")
        return self.merged_data
    
    def split(self, split_ratios=[0.7, 0.15, 0.15], random_seed=2024, shuffle=True):
        """
        Re-split datasets
        
        Args:
            split_ratios: Split ratios [train_ratio, valid_ratio, test_ratio]
            random_seed: Random seed
            shuffle: Whether to shuffle data
            
        Returns:
            train_data, valid_data, test_data: Re-split data
        """
        # Check parameters
        assert len(split_ratios) == 3, "split_ratios must be a list of 3 values [train, valid, test]"
        assert abs(sum(split_ratios) - 1.0) < 1e-6, "split_ratios must sum to 1.0"
        
        # Merge data if not already merged
        if self.merged_data is None:
            self.merge()
        
        logging.info(f"Re-splitting datasets with ratios: {split_ratios}")
        
        if self.data_format == 'npz':
            return self._split_npz(split_ratios, random_seed, shuffle)
        else:
            return self._split_parquet(split_ratios, random_seed, shuffle)
    
    def _merge_npz(self):
        """Merge NPZ format data"""
        data_files = []
        
        # Collect all data files
        if self.train_path:
            train_file = self.train_path if self.train_path.endswith('.npz') else self.train_path + '.npz'
            if os.path.exists(train_file):
                data_files.append(train_file)
                
        if self.valid_path:
            valid_file = self.valid_path if self.valid_path.endswith('.npz') else self.valid_path + '.npz'
            if os.path.exists(valid_file):
                data_files.append(valid_file)
                
        if self.test_path:
            test_file = self.test_path if self.test_path.endswith('.npz') else self.test_path + '.npz'
            if os.path.exists(test_file):
                data_files.append(test_file)
        
        if not data_files:
            raise ValueError("No valid data files found to merge")
        
        # Merge all data
        all_cols = list(self.feature_map.features.keys()) + self.feature_map.labels
        merged_data = {col: [] for col in all_cols}
        
        for data_file in data_files:
            logging.info(f"Loading {data_file}...")
            data_dict = np.load(data_file)
            for col in all_cols:
                merged_data[col].append(data_dict[col])
        
        # Concatenate data
        for col in all_cols:
            merged_data[col] = np.concatenate(merged_data[col], axis=0)
        
        return merged_data
    
    def _merge_parquet(self):
        """Merge Parquet format data"""
        data_files = []
        
        # Collect all data files
        if self.train_path:
            train_file = self.train_path if self.train_path.endswith('.parquet') else self.train_path + '.parquet'
            if os.path.exists(train_file):
                data_files.append(train_file)
                
        if self.valid_path:
            valid_file = self.valid_path if self.valid_path.endswith('.parquet') else self.valid_path + '.parquet'
            if os.path.exists(valid_file):
                data_files.append(valid_file)
                
        if self.test_path:
            test_file = self.test_path if self.test_path.endswith('.parquet') else self.test_path + '.parquet'
            if os.path.exists(test_file):
                data_files.append(test_file)
        
        if not data_files:
            raise ValueError("No valid data files found to merge")
        
        # Merge all data
        dfs = []
        for data_file in data_files:
            logging.info(f"Loading {data_file}...")
            dfs.append(pd.read_parquet(data_file))
        
        merged_df = pd.concat(dfs, ignore_index=True)
        return merged_df
    
    def _split_npz_by_user_records(self, split_ratios, random_seed, shuffle):
        """
        Split within each user, then merge all users' corresponding parts
        Reference: user_aggregate logic in parquet_block_dataloader.py
        """
        logging.info(f"Splitting by user records (user_col: {self.user_col})")
        
        all_cols = list(self.feature_map.features.keys()) + self.feature_map.labels
        user_ids = self.merged_data[self.user_col]
        unique_users = np.unique(user_ids)
        
        # Store split results for each user
        train_parts = {col: [] for col in all_cols}
        valid_parts = {col: [] for col in all_cols}
        test_parts = {col: [] for col in all_cols}
        
        filtered_users = 0
        total_users = len(unique_users)
        
        # Calculate ratios
        train_ratio = split_ratios[0]
        valid_ratio = split_ratios[1]
        
        for user_id in unique_users:
            # Get all records for this user
            user_mask = user_ids == user_id
            user_indices = np.where(user_mask)[0]
            n = len(user_indices)
            
            # Filter users with insufficient records
            if n < self.min_records:
                filtered_users += 1
                continue
            
            # Split by ratio (consistent with parquet_block_dataloader.py logic)
            train_end = int(n * train_ratio)
            valid_end = int(n * (train_ratio + valid_ratio))
            
            # Split this user's data
            user_train_indices = user_indices[:train_end]
            user_valid_indices = user_indices[train_end:valid_end]
            user_test_indices = user_indices[valid_end:]
            
            # Add to corresponding parts
            for col in all_cols:
                if len(user_train_indices) > 0:
                    train_parts[col].append(self.merged_data[col][user_train_indices])
                if len(user_valid_indices) > 0:
                    valid_parts[col].append(self.merged_data[col][user_valid_indices])
                if len(user_test_indices) > 0:
                    test_parts[col].append(self.merged_data[col][user_test_indices])
        
        # Merge all users' corresponding parts
        train_data = {}
        valid_data = {}
        test_data = {}
        
        for col in all_cols:
            if train_parts[col]:
                train_data[col] = np.concatenate(train_parts[col], axis=0)
            else:
                train_data[col] = np.array([])
            
            if valid_parts[col]:
                valid_data[col] = np.concatenate(valid_parts[col], axis=0)
            else:
                valid_data[col] = np.array([])
            
            if test_parts[col]:
                test_data[col] = np.concatenate(test_parts[col], axis=0)
            else:
                test_data[col] = np.array([])
        
        valid_users = total_users - filtered_users
        
        logging.info(f"Filtered {filtered_users} users with less than {self.min_records} records")
        logging.info(f"Valid users: {valid_users}")
        logging.info(f"Re-split train samples: {len(train_data[all_cols[0]])}")
        logging.info(f"Re-split valid samples: {len(valid_data[all_cols[0]])}")
        logging.info(f"Re-split test samples: {len(test_data[all_cols[0]])}")
        
        return train_data, valid_data, test_data
    
    def _split_npz(self, split_ratios, random_seed, shuffle):
        """Re-split NPZ format data"""
        all_cols = list(self.feature_map.features.keys()) + self.feature_map.labels
        total_samples = len(self.merged_data[all_cols[0]])
        
        if self.split_mode == 'user_split':
            # Split within each user, then merge
            return self._split_npz_by_user_records(split_ratios, random_seed, shuffle)
        elif self.split_mode == 'user_group':
            # Split by user groups
            logging.info(f"Splitting by user groups (user_col: {self.user_col})")
            
            # Get all unique users
            user_ids = self.merged_data[self.user_col]
            unique_users = np.unique(user_ids)
            total_users = len(unique_users)
            logging.info(f"Total unique users: {total_users}")
            
            # Shuffle user order
            if shuffle:
                np.random.seed(random_seed)
                unique_users = np.random.permutation(unique_users)
            
            # Calculate number of users per set
            train_user_size = int(total_users * split_ratios[0])
            valid_user_size = int(total_users * split_ratios[1])
            
            # Split users
            train_users = unique_users[:train_user_size]
            valid_users = unique_users[train_user_size:train_user_size+valid_user_size]
            test_users = unique_users[train_user_size+valid_user_size:]
            
            # Get sample indices by user ID
            train_mask = np.isin(user_ids, train_users)
            valid_mask = np.isin(user_ids, valid_users)
            test_mask = np.isin(user_ids, test_users)
            
            train_indices = np.where(train_mask)[0]
            valid_indices = np.where(valid_mask)[0]
            test_indices = np.where(test_mask)[0]
            
            # Split data
            train_data = {col: self.merged_data[col][train_indices] for col in all_cols}
            valid_data = {col: self.merged_data[col][valid_indices] for col in all_cols}
            test_data = {col: self.merged_data[col][test_indices] for col in all_cols}
            
            logging.info(f"Re-split train: {len(train_users)} users, {len(train_indices)} samples")
            logging.info(f"Re-split valid: {len(valid_users)} users, {len(valid_indices)} samples")
            logging.info(f"Re-split test: {len(test_users)} users, {len(test_indices)} samples")
            
        else:
            # Random sampling split (original logic)
            logging.info("Splitting by random sampling")
            
            # Create indices
            indices = np.arange(total_samples)
            
            # Shuffle data
            if shuffle:
                np.random.seed(random_seed)
                indices = np.random.permutation(total_samples)
            
            # Calculate split positions
            train_size = int(total_samples * split_ratios[0])
            valid_size = int(total_samples * split_ratios[1])
            
            train_indices = indices[:train_size]
            valid_indices = indices[train_size:train_size+valid_size]
            test_indices = indices[train_size+valid_size:]
            
            # Split data
            train_data = {col: self.merged_data[col][train_indices] for col in all_cols}
            valid_data = {col: self.merged_data[col][valid_indices] for col in all_cols}
            test_data = {col: self.merged_data[col][test_indices] for col in all_cols}
            
            logging.info(f"Re-split train samples: {len(train_indices)}")
            logging.info(f"Re-split valid samples: {len(valid_indices)}")
            logging.info(f"Re-split test samples: {len(test_indices)}")
        
        return train_data, valid_data, test_data
    
    def _split_parquet_by_user_records(self, split_ratios, random_seed, shuffle):
        """
        Split within each user, then merge all users' corresponding parts (Parquet format)
        Reference: user_aggregate logic in parquet_block_dataloader.py
        """
        logging.info(f"Splitting by user records (user_col: {self.user_col})")
        
        # Calculate ratios
        train_ratio = split_ratios[0]
        valid_ratio = split_ratios[1]
        
        # Filter users with insufficient records (consistent with parquet_block_dataloader.py)
        user_counts = self.merged_data.groupby(self.user_col).size()
        valid_users = user_counts[user_counts >= self.min_records].index
        df_filtered = self.merged_data[self.merged_data[self.user_col].isin(valid_users)]
        
        filtered_users = len(user_counts) - len(valid_users)
        total_users = len(user_counts)
        
        # Store split results for each user
        train_parts = []
        valid_parts = []
        test_parts = []
        
        # Split for each user (consistent with parquet_block_dataloader.py)
        for user, user_df in df_filtered.groupby(self.user_col):
            n = len(user_df)
            train_end = int(n * train_ratio)
            valid_end = int(n * (train_ratio + valid_ratio))
            
            # Split this user's data
            user_train = user_df.iloc[:train_end]
            user_valid = user_df.iloc[train_end:valid_end]
            user_test = user_df.iloc[valid_end:]
            
            # Add to corresponding parts
            if len(user_train) > 0:
                train_parts.append(user_train)
            if len(user_valid) > 0:
                valid_parts.append(user_valid)
            if len(user_test) > 0:
                test_parts.append(user_test)
        
        # Merge all users' corresponding parts
        train_data = pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame()
        valid_data = pd.concat(valid_parts, ignore_index=True) if valid_parts else pd.DataFrame()
        test_data = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame()
        
        valid_users_count = len(valid_users)
        
        logging.info(f"Filtered {filtered_users} users with less than {self.min_records} records")
        logging.info(f"Valid users: {valid_users_count}")
        logging.info(f"Re-split train samples: {len(train_data)}")
        logging.info(f"Re-split valid samples: {len(valid_data)}")
        logging.info(f"Re-split test samples: {len(test_data)}")
        
        return train_data, valid_data, test_data
    
    def _split_parquet(self, split_ratios, random_seed, shuffle):
        """Re-split Parquet format data"""
        total_samples = len(self.merged_data)
        
        if self.split_mode == 'user_split':
            # Split within each user, then merge
            return self._split_parquet_by_user_records(split_ratios, random_seed, shuffle)
        elif self.split_mode == 'user_group':
            # Split by user groups
            logging.info(f"Splitting by user groups (user_col: {self.user_col})")
            
            # Get all unique users
            unique_users = self.merged_data[self.user_col].unique()
            total_users = len(unique_users)
            logging.info(f"Total unique users: {total_users}")
            
            # Shuffle user order
            if shuffle:
                np.random.seed(random_seed)
                unique_users = np.random.permutation(unique_users)
            
            # Calculate number of users per set
            train_user_size = int(total_users * split_ratios[0])
            valid_user_size = int(total_users * split_ratios[1])
            
            # Split users
            train_users = unique_users[:train_user_size]
            valid_users = unique_users[train_user_size:train_user_size+valid_user_size]
            test_users = unique_users[train_user_size+valid_user_size:]
            
            # Filter data by user ID
            train_data = self.merged_data[self.merged_data[self.user_col].isin(train_users)].reset_index(drop=True)
            valid_data = self.merged_data[self.merged_data[self.user_col].isin(valid_users)].reset_index(drop=True)
            test_data = self.merged_data[self.merged_data[self.user_col].isin(test_users)].reset_index(drop=True)
            
            logging.info(f"Re-split train: {len(train_users)} users, {len(train_data)} samples")
            logging.info(f"Re-split valid: {len(valid_users)} users, {len(valid_data)} samples")
            logging.info(f"Re-split test: {len(test_users)} users, {len(test_data)} samples")
            
        else:
            # Random sampling split (original logic)
            logging.info("Splitting by random sampling")
            
            # Shuffle data
            if shuffle:
                merged_df = self.merged_data.sample(frac=1, random_state=random_seed).reset_index(drop=True)
            else:
                merged_df = self.merged_data.copy()
            
            # Calculate split positions
            train_size = int(total_samples * split_ratios[0])
            valid_size = int(total_samples * split_ratios[1])
            
            # Split data
            train_data = merged_df[:train_size]
            valid_data = merged_df[train_size:train_size+valid_size]
            test_data = merged_df[train_size+valid_size:]
            
            logging.info(f"Re-split train samples: {len(train_data)}")
            logging.info(f"Re-split valid samples: {len(valid_data)}")
            logging.info(f"Re-split test samples: {len(test_data)}")
        
        return train_data, valid_data, test_data
    
    def save(self, train_data, valid_data, test_data, output_dir, prefix='resplit'):
        """
        Save re-split data to files
        
        Args:
            train_data: Training set data (can be None)
            valid_data: Validation set data (can be None)
            test_data: Test set data (can be None)
            output_dir: Output directory
            prefix: File name prefix
            
        Returns:
            train_path, valid_path, test_path: Saved file paths (None if data is None)
        """
        os.makedirs(output_dir, exist_ok=True)
        
        train_path = None
        valid_path = None
        test_path = None
        
        if self.data_format == 'npz':
            # Save train data
            if train_data is not None:
                train_path = os.path.join(output_dir, f'{prefix}_train.npz')
                np.savez_compressed(train_path, **train_data)
            else:
                logging.warning(f"Train data is None, skipping save for {prefix}_train.npz")
            
            # Save valid data
            if valid_data is not None:
                valid_path = os.path.join(output_dir, f'{prefix}_valid.npz')
                np.savez_compressed(valid_path, **valid_data)
            else:
                logging.warning(f"Valid data is None, skipping save for {prefix}_valid.npz")
            
            # Save test data
            if test_data is not None:
                test_path = os.path.join(output_dir, f'{prefix}_test.npz')
                np.savez_compressed(test_path, **test_data)
            else:
                logging.warning(f"Test data is None, skipping save for {prefix}_test.npz")
            
        else:  # parquet
            # Save train data
            if train_data is not None:
                train_path = os.path.join(output_dir, f'{prefix}_train.parquet')
                train_data.to_parquet(train_path, index=False)
            else:
                logging.warning(f"Train data is None, skipping save for {prefix}_train.parquet")
            
            # Save valid data
            if valid_data is not None:
                valid_path = os.path.join(output_dir, f'{prefix}_valid.parquet')
                valid_data.to_parquet(valid_path, index=False)
            else:
                logging.warning(f"Valid data is None, skipping save for {prefix}_valid.parquet")
            
            # Save test data
            if test_data is not None:
                test_path = os.path.join(output_dir, f'{prefix}_test.parquet')
                test_data.to_parquet(test_path, index=False)
            else:
                logging.warning(f"Test data is None, skipping save for {prefix}_test.parquet")
        
        saved_files = [f for f in [train_path, valid_path, test_path] if f is not None]
        if saved_files:
            logging.info(f"Saved {len(saved_files)} dataset(s) to {output_dir}")
        
        return train_path, valid_path, test_path

