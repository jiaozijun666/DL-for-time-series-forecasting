import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import os
from models.DLinear_model import DLinearAnalyzer

dataset_params = {
    'air_quality': {
        'input_window': 24,        
        'output_window': 6,      
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        'learning_rate': 0.0005,
        'num_epochs': 200,
        'batch_size': 64        
    },
    'energy': {
        'input_window': 144,       
        'output_window': 12,       
        'train_ratio': 0.7,        
        'val_ratio': 0.15,         
        'learning_rate': 0.0003,
        'num_epochs': 250,
        'batch_size': 64
    },
    'gait': {
        'input_window': 1000,      
        'output_window': 200,   
        'train_ratio': 0.8,
        'val_ratio': 0.1,
        'learning_rate': 0.0001,
        'num_epochs': 75,         
        'batch_size': 16                
    },
    'metro': {
        'input_window': 168,      
        'output_window': 24,    
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        'learning_rate': 0.0003,
        'num_epochs': 200,
        'batch_size': 64
    },
    'productivity': {
        'input_window': 20,    
        'output_window': 5,        
        'train_ratio': 0.7,
        'val_ratio': 0.2,         
        'learning_rate': 0.001,    
        'num_epochs': 300,         
        'batch_size': 16,          
        'early_stop_patience': 30  

    }
}
analyzer = DLinearAnalyzer(
    metrics_dir='./results/metric_csv/DLinear',
    predictions_dir='./results/prediction_png/DLinear',
    loss_curves_dir='./results/loss_png/DLinear',
    comparisons_dir='./results/comparisons_png/DLinear'
)

datasets = [
    './data/data_clean/productivity.csv',
    './data/data_clean/air_quality.csv',
    './data/data_clean/energy.csv',
    './data/data_clean/metro.csv',
    './data/data_clean/gait.csv'
]

metrics, models = analyzer.batch_analyze(
    file_paths=datasets,
    params_dict=dataset_params)