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

import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))

import sys
sys.path.append('../../..')
sys.path.append('../../../../')

import logging
import fuxictr_version
from fuxictr import datasets
from datetime import datetime
from fuxictr.utils import load_config, set_logger, print_to_json, print_to_list
from fuxictr.features import FeatureMap
from fuxictr.pytorch.dataloaders import RankDataLoader, DatasetReSplitter
from fuxictr.pytorch.dataloaders.parquet_dataloader import ParquetDataLoader as ParquetLoader
from fuxictr.pytorch.torch_utils import seed_everything
import pandas as pd
import numpy as np
from fuxictr.preprocess import FeatureProcessor, build_dataset
import src
import gc
import argparse
import torch
from pathlib import Path


def identify_user_features(feature_map, user_col=None):
    """
    Identify user-related feature fields
    """
    user_features = []
    
    for feature_name, feature_spec in feature_map.features.items():
        # Identify through source field
        if feature_spec.get('source', '').lower() in ['user', 'users']:
            user_features.append(feature_name)
            continue
        
        # Identify through feature name
        if 'user' in feature_name.lower():
            user_features.append(feature_name)
            continue
        
        # If user_col is specified
        if user_col and feature_name == user_col:
            user_features.append(feature_name)
    
    logging.info(f"Identified user feature fields: {user_features}")
    return user_features


def transfer_weights_except_user_embeddings(pretrained_checkpoint, new_model, user_features):
    """
    Load parameters from pretrained model, but skip user embedding related parameters
    """
    logging.info("="*80)
    logging.info("Loading pretrained parameters (skipping user embeddings)...")
    logging.info("="*80)
    
    # Load pretrained parameters
    pretrained_state = torch.load(pretrained_checkpoint, map_location="cpu")
    new_state = new_model.state_dict()
    
    transferred_params = []
    skipped_params = []
    
    for name, param in pretrained_state.items():
        # Check if it's a user-related parameter
        is_user_param = False
        for user_feat in user_features:
            # Check if parameter name contains user feature name
            if user_feat in name and 'embedding' in name.lower():
                is_user_param = True
                break
        
        if is_user_param:
            skipped_params.append(name)
            logging.info(f"  Skipping user parameter: {name} (shape: {param.shape})")
        elif name in new_state:
            if param.shape == new_state[name].shape:
                new_state[name].copy_(param)
                transferred_params.append(name)
            else:
                skipped_params.append(name)
                logging.info(f"  Shape mismatch, skipping: {name} (pretrain: {param.shape}, new: {new_state[name].shape})")
        else:
            skipped_params.append(name)
    
    new_model.load_state_dict(new_state)
    
    logging.info(f"Parameter transfer completed: {len(transferred_params)} parameters transferred, {len(skipped_params)} parameters skipped")
    for name, param in new_model.named_parameters():
        if name in transferred_params:
            param.requires_grad = False
    return new_model


if __name__ == '__main__':
    ''' 
    Usage: python run_expid.py --config {config_dir} --expid {experiment_id} --gpu {gpu_device_id}
    
    Support transfer learning mode (enabled through config file):
    1. Split data by user groups with 7:3 ratio (pretrain_data vs new_user_data)
    2. Train model using pretrain_data and save
    3. Load pretrained parameters (except user embeddings) to new model
    4. Split new_user_data by 8:2 ratio per user into query and test
    5. Fine-tune using query data, test on test data
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/', help='The config directory.')
    parser.add_argument('--expid', type=str, default='DNN_cat_frappe', help='The experiment id to run.')
    parser.add_argument('--gpu', type=int, default=0, help='The gpu index, -1 for cpu')
    args = vars(parser.parse_args())
    
    experiment_id = args['expid']
    params = load_config(args['config'], experiment_id)
    params['gpu'] = args['gpu']
    set_logger(params)
    logging.info("Params: " + print_to_json(params))
    seed_everything(seed=params['seed'])

    data_dir = os.path.join(params['data_root'], params['dataset_id'])
    feature_map_json = os.path.join(data_dir, "feature_map.json")
    if params["data_format"] == "csv":
        # Build feature_map and transform data
        feature_encoder = FeatureProcessor(**params)
        params["train_data"], params["valid_data"], params["test_data"] = \
            build_dataset(feature_encoder, **params)
    feature_map = FeatureMap(params['dataset_id'], data_dir)
    feature_map.load(feature_map_json, params)
    logging.info("Feature specs: " + print_to_json(feature_map.features))
    
    # Check if transfer learning mode is enabled
    enable_transfer_learning = params.get('enable_transfer_learning', False)
    
    if enable_transfer_learning:
        logging.info("="*80)
        logging.info("Starting transfer learning mode (user cold-start scenario)")
        logging.info("="*80)
        
        # ========== Step 1: Split data by user groups with 7:3 ratio ==========
        logging.info("\n[Step 1] Splitting data by user groups with 7:3 ratio...")
        user_split_dir = os.path.join(data_dir, 'transfer_learning_user_split')
        os.makedirs(user_split_dir, exist_ok=True)
        
        splitter_user_group = DatasetReSplitter(
            feature_map=feature_map,
            train_path=params.get('train_data'),
            valid_path=params.get('valid_data'),
            test_path=params.get('test_data'),
            data_format=params.get('data_format', 'npz'),
            split_mode='user_group',  # Split by user groups, ensuring no user overlap
            user_col=params.get('user_col', None),
            min_records=params.get('min_records', 5)
        )
        
        pretrain_data, new_user_data, _ = splitter_user_group.split(
            split_ratios=[0.7, 0.3, 0.0],
            random_seed=params.get('seed', 2024),
            shuffle=True
        )
        
        pretrain_path, new_user_path, _ = splitter_user_group.save(
            pretrain_data, new_user_data, None,
            output_dir=user_split_dir,
            prefix='user_group'
        )
        
        logging.info(f"  ✓ Pretrain data: {pretrain_path} ({len(pretrain_data)} samples)")
        logging.info(f"  ✓ New user data: {new_user_path} ({len(new_user_data)} samples)")
        
        # ========== Step 2: Train model using pretrain_data ==========
        logging.info("\n[Step 2] Pretraining with 70% user data...")
        
        # Split pretrain_data into train and valid (8:1 by sample count, not by user)
        logging.info("  Splitting pretrain data into train:valid = 8:1 (by sample count)...")
        splitter_pretrain = DatasetReSplitter(
            feature_map=feature_map,
            train_path=pretrain_path,
            valid_path=None,
            test_path=None,
            data_format=params.get('data_format', 'npz'),
            split_mode='random',  # Random split by samples, not by user
            user_col=params.get('user_col', None),
            min_records=0  # No filtering here
        )
        
        pretrain_train_data, pretrain_valid_data, _ = splitter_pretrain.split(
            split_ratios=[8.0/9.0, 1.0/9.0, 0.0],  # 8:1 ratio
            random_seed=params.get('seed', 2024),
            shuffle=True
        )
        
        pretrain_train_path, pretrain_valid_path, _ = splitter_pretrain.save(
            pretrain_train_data, pretrain_valid_data, None,
            output_dir=user_split_dir,
            prefix='pretrain'
        )
        
        logging.info(f"    Train samples: {len(pretrain_train_data)}")
        logging.info(f"    Valid samples: {len(pretrain_valid_data)}")
        
        pretrain_params = params.copy()
        pretrain_params['train_data'] = pretrain_train_path
        pretrain_params['valid_data'] = pretrain_valid_path
        pretrain_params['test_data'] = None
        
        model_class = getattr(src, params['model'])
        pretrain_model = model_class(feature_map, **pretrain_params)
        pretrain_model.count_parameters()
        
        train_gen, valid_gen = RankDataLoader(feature_map, stage='train', **pretrain_params).make_iterator()
        pretrain_model.fit(train_gen, validation_data=valid_gen, **pretrain_params)
        
        # Save pretrained model
        pretrain_checkpoint = os.path.join(pretrain_params.get('model_root', './checkpoints/'),
                                          pretrain_params['dataset_id'],
                                          'pretrained_model.pth')
        os.makedirs(os.path.dirname(pretrain_checkpoint), exist_ok=True)
        pretrain_model.save_weights(pretrain_checkpoint)
        logging.info(f"  ✓ Pretrained model saved: {pretrain_checkpoint}")
        
        del train_gen, valid_gen, pretrain_model
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # ========== Step 3: Prepare new user model and transfer parameters ==========
        logging.info("\n[Step 3] Creating new user model and transferring parameters (except user embeddings)...")
        
        # Identify user features
        user_features = identify_user_features(feature_map, params.get('user_col', None))
        
        # ========== Step 4: Split new_user_data by 8:2 ratio per user ==========
        logging.info("\n[Step 4] Splitting 30% user data by 8:2 ratio per user into query and test...")
        
        per_user_split_dir = os.path.join(data_dir, 'transfer_learning_per_user_split')
        os.makedirs(per_user_split_dir, exist_ok=True)
        
        splitter_per_user = DatasetReSplitter(
            feature_map=feature_map,
            train_path=new_user_path,
            valid_path=None,
            test_path=None,
            data_format=params.get('data_format', 'npz'),
            split_mode='user_split',  # Split within each user
            user_col=params.get('user_col', None),
            min_records=params.get('min_records', 5)
        )
        
        query_data, test_data, _ = splitter_per_user.split(
            split_ratios=[0.8, 0.2, 0.0],
            random_seed=params.get('seed', 2024),
            shuffle=params.get('resplit_shuffle', True)
        )
        
        query_path, test_path, _ = splitter_per_user.save(
            query_data, test_data, None,
            output_dir=per_user_split_dir,
            prefix='per_user'
        )
        
        logging.info(f"  ✓ Query data: {query_path} ({len(query_data)} samples)")
        logging.info(f"  ✓ Test data: {test_path} ({len(test_data)} samples)")
        
        # ========== Fine-tune using query data, test on test data ==========
        logging.info("\n[Step 5] Fine-tuning model on new user data...")
        
        # ========== Apply cat_sample_select to finetune training data ==========
        logging.info("  Applying cat_sample_select to select sample subset per user...")
        sample_ratio = params.get('cat_sample_ratio', 0.7)  # Default 70% samples per user
        
        # Load finetune training data
        parquet_loader = ParquetLoader(
            feature_map=feature_map,
            data_path=query_path
        )
        
        # Apply cat_sample_select on the dataset
        for t in range(20):
            sampled_data_array = parquet_loader.dataset.cat_sample_select("Random")
            
            # Convert back to DataFrame and save
            sampled_df = pd.DataFrame(sampled_data_array, columns=parquet_loader.dataset.all_cols)
            finetune_train_sampled_path = os.path.join(per_user_split_dir, 'finetune_train_sampled.parquet')
            sampled_df.to_parquet(finetune_train_sampled_path, index=False)
            
            logging.info(f"    Sampled train data saved: {finetune_train_sampled_path}")
            logging.info(f"    Samples after selection: {len(sampled_df)} (ratio: {sample_ratio})")
            
            finetune_params = params.copy()
            finetune_params['train_data'] = finetune_train_sampled_path  # Use sampled data
            finetune_params['valid_data'] = None
            finetune_params['test_data'] = test_path
            
            # Create new model
            finetune_model = model_class(feature_map, **finetune_params)
            
            # Transfer pretrained parameters (skip user embeddings)
            finetune_model = transfer_weights_except_user_embeddings(
                pretrain_checkpoint, finetune_model, user_features
            )
            finetune_model.count_parameters()
            
            # Fine-tune
            train_gen, _ = RankDataLoader(feature_map, stage='train', **finetune_params).make_iterator()
            finetune_model.fit_selection(train_gen,**finetune_params)
        
        test_result = {}
        if finetune_params["test_data"]:
            logging.info('******** Test data evaluation (new users) ********')
            test_gen = RankDataLoader(feature_map, stage='test', **finetune_params).make_iterator()
            test_result = finetune_model.evaluate(test_gen)
        
        result_filename = Path(args['config']).name.replace(".yaml", "") + '_transfer_learning.csv'
        with open(result_filename, 'a+') as fw:
            fw.write(' {},[command] python {},[exp_id] {},[dataset_id] {},[mode] transfer_learning,[test] {}\n' \
                .format(datetime.now().strftime('%Y%m%d-%H%M%S'), 
                        ' '.join(sys.argv), experiment_id, params['dataset_id'], print_to_list(test_result)))
        logging.info("\n" + "="*80)
        logging.info("Transfer learning completed!")
        logging.info("="*80)
        
    else:
        # ========== Original normal training process ==========
        # If data re-splitting is enabled, merge and re-split data first
        if params.get('resplit_data', False):
            logging.info("Re-splitting datasets enabled")
            splitter = DatasetReSplitter(
                feature_map=feature_map,
                train_path=params.get('train_data'),
                valid_path=params.get('valid_data'),
                test_path=params.get('test_data'),
                data_format=params.get('data_format', 'npz'),
                split_mode=params.get('split_mode', 'user_split'),  # Default: split within each user
                user_col=params.get('user_col', None),  # User column name, None means auto-detect
                min_records=params.get('min_records', 5)  # Filter out users with too few records
            )
            
            # Re-split data
            train_data, valid_data, test_data = splitter.split(
                split_ratios=params.get('split_ratios', [0.7, 0.15, 0.15]),
                random_seed=params.get('seed', 2024),
                shuffle=params.get('resplit_shuffle', True)
            )
            
            # Save re-split data
            resplit_dir = os.path.join(data_dir, 'resplit')
            new_train, new_valid, new_test = splitter.save(
                train_data, valid_data, test_data,
                output_dir=resplit_dir,
                prefix='resplit'
            )
            
        # Update data paths in parameters
        if params.get('use_split', False):
            resplit_dir = os.path.join(data_dir, 'resplit')
            if not os.path.exists(resplit_dir):
                os.makedirs(resplit_dir)
            new_train, new_valid, new_test = (os.path.join(resplit_dir, 'resplit_train.parquet'), 
                                            os.path.join(resplit_dir, 'resplit_valid.parquet'), 
                                            os.path.join(resplit_dir, 'resplit_test.parquet'))
            params['train_data'] = new_train
            params['valid_data'] = new_valid
            params['test_data'] = new_test
        
        model_class = getattr(src, params['model'])
        model = model_class(feature_map, **params)
        model.count_parameters() # print number of parameters used in model

        train_gen, valid_gen = RankDataLoader(feature_map, stage='train', **params).make_iterator()
        model.fit(train_gen, validation_data=valid_gen, **params)

        logging.info('****** Validation evaluation ******')
        valid_result = model.evaluate(valid_gen)
        del train_gen, valid_gen
        gc.collect()
        
        test_result = {}
        if params["test_data"]:
            logging.info('******** Test evaluation ********')
            test_gen = RankDataLoader(feature_map, stage='test', **params).make_iterator()
            test_result = model.evaluate(test_gen)
        
        result_filename = Path(args['config']).name.replace(".yaml", "") + '.csv'
        with open(result_filename, 'a+') as fw:
            fw.write(' {},[command] python {},[exp_id] {},[dataset_id] {},[train] {},[val] {},[test] {}\n' \
                .format(datetime.now().strftime('%Y%m%d-%H%M%S'), 
                        ' '.join(sys.argv), experiment_id, params['dataset_id'],
                        "N.A.", print_to_list(valid_result), print_to_list(test_result)))
