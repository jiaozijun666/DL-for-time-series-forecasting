import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
import os
import warnings
import time
import psutil
import gc
from utils.metrics import evaluate_all_metrics
warnings.filterwarnings('ignore')

# random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class TimeSeriesDataset(Dataset):
    def __init__(self, data, input_window, output_window):
        self.data = data
        self.input_window = input_window
        self.output_window = output_window
        self.length = len(data) - input_window - output_window + 1
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        x = self.data[idx:idx+self.input_window]
        y = self.data[idx+self.input_window:idx+self.input_window+self.output_window]
        return torch.FloatTensor(x), torch.FloatTensor(y)
    
class DLinear(nn.Module):
    def __init__(self, input_size, output_size):
        super(DLinear, self).__init__()
        self.Linear_Trend = nn.Linear(input_size, output_size)
        self.Linear_Seasonal = nn.Linear(input_size, output_size)
    
    def series_decompose(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)

        mean = torch.mean(x, dim=1, keepdim=True)
        trend = mean.repeat(1, x.shape[1], 1)
        seasonal = x - trend
        return trend, seasonal
    
    def forward(self, x):
        batch_size = x.size(0)
        if x.dim() == 3:  
            x = x.squeeze(-1)  
        
        trend, seasonal = self.series_decompose(x.unsqueeze(-1))
        trend = trend.squeeze(-1)
        seasonal = seasonal.squeeze(-1)
        trend_output = self.Linear_Trend(trend)
        seasonal_output = self.Linear_Seasonal(seasonal)
        x = trend_output + seasonal_output
        return x

def preprocess_data(file_path, value_col_name=None, date_col_name=None):
    df = pd.read_csv(file_path)   
    if date_col_name is None or value_col_name is None:
        # simple heuristic to guess the columns
        potential_date_cols = [col for col in df.columns if any(kw in col.lower() for kw in ['date', 'time', 'timestamp'])]
        if date_col_name is None:
            if potential_date_cols:
                date_col_name = potential_date_cols[0]
            else:
                date_col_name = df.columns[0]

        if value_col_name is None:
            non_date_cols = [col for col in df.columns if col != date_col_name]
            if non_date_cols:
                value_col_name = non_date_cols[0]
            else:
                raise ValueError("could not find a suitable value column")
    
    print(f"Use '{date_col_name}' as data column, '{value_col_name}' as value column")
    
    try:
        df[date_col_name] = pd.to_datetime(df[date_col_name])
    except:
        print(f"warning: '{date_col_name}' cannot be converted to datetime, using the first column as date")
 
    df = df.sort_values(by=date_col_name)
    if df[value_col_name].isnull().any():
        print(f"warning: '{value_col_name}' contains NaN values, filling with forward fill")
        df[value_col_name] = df[value_col_name].fillna(method='ffill')

    return df[value_col_name].values, df[date_col_name].values

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / (1024 * 1024)  # Convert to MB

def train_dlinear(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, patience=10):
    model.to(device)
    train_losses = []
    val_losses = []
    
    best_val_loss = float('inf')
    no_improve_epochs = 0
    best_model = None
    early_stopped = False
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(x) 
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()         
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                x = x.unsqueeze(-1) 
                y_pred = model(x).squeeze(-1)
                loss = criterion(y_pred, y)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_epochs = 0
            best_model = model.state_dict().copy()
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print(f'Early stopping at epoch {epoch+1}, no improvement for {patience} epochs')
                if best_model is not None:
                    model.load_state_dict(best_model) 
                early_stopped = True
                break
    
    total_time = time.time() - start_time
    final_epoch = epoch + 1
    
    return train_losses, val_losses, total_time, early_stopped, final_epoch

def forecast(model, data, input_window, output_window, forecast_steps, device):
    model.eval()
    predictions = []

    x = data[-input_window:].copy()

    for _ in range(0, forecast_steps, output_window):
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x[-input_window:]).unsqueeze(0).unsqueeze(-1).to(device)
            y_pred = model(x_tensor).squeeze(-1).squeeze(0).cpu().numpy()
            
            # limit the number of steps to output_window
            steps = min(output_window, forecast_steps - len(predictions))
            predictions.extend(y_pred[:steps])
            x = np.append(x, y_pred[:steps])
            
        if len(predictions) >= forecast_steps:
            break
    
    return np.array(predictions)
'''
def plot_results(dataset_name, y_true, y_pred, train_losses, val_losses, output_dir, 
                 training_time, memory_usage, early_stopped, final_epoch):
    os.makedirs(output_dir, exist_ok=True)
    
    # fig1: prediction plot
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='true value', color='blue')
    plt.plot(y_pred, label='DLinear forecast', color='red', linestyle='--')
    plt.title(f'{dataset_name} - DLinear forecast')
    plt.xlabel('time step')
    plt.ylabel('value')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'prediction_{dataset_name}.png'))

    
    # fig2: loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='training loss')
    plt.plot(val_losses, label='validation loss')
    plt.title(f'{dataset_name} - DLinear model loss curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    # Add training info as text on the plot
    early_stop_text = "Early stopped" if early_stopped else "Completed all epochs"
    plt.figtext(0.5, 0.01, 
                f"Training time: {training_time:.2f}s | Memory: {memory_usage:.1f}MB | {early_stop_text} | Final epoch: {final_epoch}",
                ha="center", fontsize=9, bbox={"facecolor":"white", "alpha":0.5, "pad":5})
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'loss_{dataset_name}.png'))
    '''
    
class DLinearAnalyzer:
  
    def __init__(self, input_window=24, output_window=12, train_ratio=0.7, val_ratio=0.1,
                 batch_size=32, num_epochs=100, learning_rate=0.001, 
                 metrics_dir='metrics', predictions_dir='predictions', 
                 loss_curves_dir='loss_curves', comparisons_dir='comparisons'):
        self.input_window = input_window
        self.output_window = output_window
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        
        # 设置单独的输出目录
        self.metrics_dir = metrics_dir
        self.predictions_dir = predictions_dir
        self.loss_curves_dir = loss_curves_dir
        self.comparisons_dir = comparisons_dir
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"device: {self.device}")
        
        # 创建各个输出目录
        os.makedirs(self.metrics_dir, exist_ok=True)
        os.makedirs(self.predictions_dir, exist_ok=True)
        os.makedirs(self.loss_curves_dir, exist_ok=True)
        os.makedirs(self.comparisons_dir, exist_ok=True)
    
    def analyze_dataset(self, file_path, forecast_steps=None, date_col=None, value_col=None):
        # Record initial memory usage
        initial_memory = get_memory_usage()
        
        dataset_name = os.path.splitext(os.path.basename(file_path))[0]
        print(f"\n---analyzing dataset: {dataset_name}---")
        
        if forecast_steps is None:
            forecast_steps = self.output_window
        
        values, dates = preprocess_data(file_path, value_col, date_col)
        
        n = len(values)
        train_size = int(self.train_ratio * n)
        val_size = int(self.val_ratio * n)
        test_size = n - train_size - val_size
        
        train_data = values[:train_size]
        val_data = values[train_size:train_size+val_size]
        test_data = values[train_size+val_size:]
        test_dates = dates[train_size+val_size:]
        
        print(f"dataset size: {n}, train size: {train_size}, val size: {val_size}, test size: {test_size}")
        
        scaler = StandardScaler()
        train_data_scaled = scaler.fit_transform(train_data.reshape(-1, 1)).flatten()
        val_data_scaled = scaler.transform(val_data.reshape(-1, 1)).flatten()
        test_data_scaled = scaler.transform(test_data.reshape(-1, 1)).flatten()
        
        # create datasets and dataloaders
        train_dataset = TimeSeriesDataset(train_data_scaled, self.input_window, self.output_window)
        val_dataset = TimeSeriesDataset(val_data_scaled, self.input_window, self.output_window)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # initialize model, loss function and optimizer
        model = DLinear(self.input_window, self.output_window)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        
        print("starting training...")
        train_losses, val_losses, training_time, early_stopped, final_epoch = train_dlinear(
            model, train_loader, val_loader, criterion, optimizer, 
            self.num_epochs, self.device
        )
        print(f"training finished in {training_time:.2f} seconds. Early stopped: {early_stopped}")
        
        # Get peak memory usage after training
        peak_memory = get_memory_usage()
        memory_used = peak_memory - initial_memory
        print(f"Memory usage: {memory_used:.1f} MB")
        
        print("starting prediction...")
        all_data_scaled = np.concatenate([train_data_scaled, val_data_scaled])
        prediction_scaled = forecast(
            model, all_data_scaled, self.input_window, self.output_window, 
            test_size, self.device
        )
        
        prediction = scaler.inverse_transform(prediction_scaled.reshape(-1, 1)).flatten()
        prediction = prediction[:len(test_data)]

        print("evaluating model...")
        metrics = evaluate_all_metrics(test_data, prediction)
 
        print("\n--- evaluation metrics ---")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        # Add performance metrics to the metrics dict
        metrics['training_time(s)'] = training_time
        metrics['memory_usage_mb(MB)'] = memory_used
        metrics['early_stopped'] = 1 if early_stopped else 0
        metrics['final_epoch'] = final_epoch
        # drop metrics['Runtime_Seconds']
        metrics.pop('Runtime_Seconds', None)
        
        print("creating visualization plots and saving results...")
        
        # Create a two-row CSV in the metrics folder
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        with open(os.path.join(self.metrics_dir, f'metrics_{dataset_name}.csv'), 'w') as f:
            # Write header row (metric names)
            f.write(','.join(metric_names) + '\n')
            # Write values row
            values_str = []
            for val in metric_values:
                if isinstance(val, (int, bool)):
                    values_str.append(str(val))
                elif isinstance(val, float):
                    values_str.append(f"{val:.4f}")
                else:
                    values_str.append(str(val))
            f.write(','.join(values_str) + '\n')
        
        # Save prediction plot to predictions folder
        plt.figure(figsize=(12, 6))
        plt.plot(test_data, label='true value', color='blue')
        plt.plot(prediction, label='DLinear forecast', color='red', linestyle='--')
        plt.title(f'{dataset_name} - DLinear forecast')
        plt.xlabel('time step')
        plt.ylabel('value')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.predictions_dir, f'prediction_{dataset_name}.png'))
        plt.close()
        
        # Save loss curves plot to loss_curves folder
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='training loss')
        plt.plot(val_losses, label='validation loss')
        plt.title(f'{dataset_name} - DLinear model loss curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.loss_curves_dir, f'loss_{dataset_name}.png'))
        plt.close()
        
        print(f"Results saved to folders")    
        gc.collect()
        return metrics, model, prediction
    
    def batch_analyze(self, file_paths, params_dict=None, **kwargs):
        
        all_metrics = {}
        all_models = {}
        
        if params_dict is None:
            params_dict = {}
        
        for file_path in file_paths:
            dataset_name = os.path.splitext(os.path.basename(file_path))[0]
            
            # Get dataset-specific parameters or use default
            dataset_params = params_dict.get(dataset_name, {})
            
            # Create a temporary analyzer with custom parameters if needed
            if dataset_params:
                temp_analyzer = DLinearAnalyzer(
                    input_window=dataset_params.get('input_window', self.input_window),
                    output_window=dataset_params.get('output_window', self.output_window),
                    train_ratio=dataset_params.get('train_ratio', self.train_ratio),
                    val_ratio=dataset_params.get('val_ratio', self.val_ratio),
                    batch_size=dataset_params.get('batch_size', self.batch_size),
                    num_epochs=dataset_params.get('num_epochs', self.num_epochs),
                    learning_rate=dataset_params.get('learning_rate', self.learning_rate),
                    metrics_dir=self.metrics_dir,
                    predictions_dir=self.predictions_dir,
                    loss_curves_dir=self.loss_curves_dir,
                    comparisons_dir=self.comparisons_dir
                )
                
                print(f"\nUsing custom parameters for {dataset_name}:")
                for param, value in dataset_params.items():
                    print(f"  - {param}: {value}")
                
                # Use the temporary analyzer for this dataset
                metrics, model, _ = temp_analyzer.analyze_dataset(
                    file_path, 
                    forecast_steps=dataset_params.get('forecast_steps', None),
                    date_col=dataset_params.get('date_col', None),
                    value_col=dataset_params.get('value_col', None)
                )
            else:
                # Use the default analyzer parameters
                metrics, model, _ = self.analyze_dataset(file_path, **kwargs)
            
            all_metrics[dataset_name] = metrics
            all_models[dataset_name] = model
        
        # Create comparison charts for different metrics
        metrics_to_plot = ['MAE', 'RMSE', 'MAPE', 'sMAPE', 'Directional_Accuracy', 
                          'Threshold_Accuracy', 'training_time(s)', 'memory_usage_mb(MB)']
        
        for metric in metrics_to_plot:
            plt.figure(figsize=(10, 6))
            datasets = list(all_metrics.keys())
            values = [all_metrics[ds][metric] for ds in datasets]
            
            # Create bar chart
            bars = plt.bar(datasets, values, color='#1f77b4')
            
            # Add value labels on top of bars
            for bar, value in zip(bars, values):
                if isinstance(value, float):
                    label_text = f"{value:.3f}"
                else:
                    label_text = f"{value}"
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        label_text, ha='center', va='bottom')
            
            # Add grid lines for readability
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Set chart title and labels
            plt.title(f'{metric} per dataset', fontsize=16, fontweight='bold')
            plt.ylabel(metric, fontsize=12)
            plt.xticks(rotation=45)
            
            # Adjust layout and save figure
            plt.tight_layout()
            plt.savefig(os.path.join(self.comparisons_dir, f'comparison_{metric}.png'))
            plt.close()
        
        print("\n=== Dataset Analysis Complete ===")

        return all_metrics, all_models
        