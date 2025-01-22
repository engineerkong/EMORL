import numpy as np
import time
import psutil
import torch
from typing import List, Dict, Optional
import matplotlib.pyplot as plt

class ComputationMetrics:
    def __init__(self):
        self.start_time = time.time()
        self.epoch_times = []
        self.loss_history = []
        self.gpu_utilization = []
        self.memory_usage = []
        
    def log_epoch(self, loss: float):
        """Log metrics for current epoch"""
        self.epoch_times.append(time.time() - self.start_time)
        self.loss_history.append(loss)
        
        if torch.cuda.is_available():
            self.gpu_utilization.append(torch.cuda.utilization())
            self.memory_usage.append(torch.cuda.memory_allocated())
    
    def calculate_convergence_metrics(self, 
                                   convergence_threshold: float = 0.01,
                                   window_size: int = 5) -> Dict:
        """
        Calculate convergence-related metrics
        
        Args:
            convergence_threshold: Threshold for considering model converged
            window_size: Window size for calculating moving average
            
        Returns:
            Dictionary containing convergence metrics
        """
        metrics = {}
        
        # Time to convergence
        for i in range(len(self.loss_history)):
            if i >= window_size:
                window = self.loss_history[i-window_size:i]
                if np.std(window) < convergence_threshold:
                    metrics['time_to_convergence'] = self.epoch_times[i]
                    metrics['epochs_to_convergence'] = i
                    break
        
        # Convergence rate
        if len(self.loss_history) > 1:
            metrics['convergence_rate'] = (self.loss_history[0] - self.loss_history[-1]) / len(self.loss_history)
        
        return metrics
    
    def calculate_resource_metrics(self) -> Dict:
        """Calculate resource utilization metrics"""
        metrics = {}
        
        if self.gpu_utilization:
            metrics['avg_gpu_utilization'] = np.mean(self.gpu_utilization)
            metrics['peak_gpu_utilization'] = np.max(self.gpu_utilization)
            metrics['gpu_utilization_std'] = np.std(self.gpu_utilization)
            
        if self.memory_usage:
            metrics['avg_memory_usage'] = np.mean(self.memory_usage) / 1e9  # Convert to GB
            metrics['peak_memory_usage'] = np.max(self.memory_usage) / 1e9
            metrics['memory_efficiency'] = np.mean(self.memory_usage) / np.max(self.memory_usage)
            
        # CPU metrics
        metrics['cpu_usage'] = psutil.cpu_percent()
        metrics['ram_usage'] = psutil.virtual_memory().percent
        
        return metrics
    
    def calculate_scalability_metrics(self, 
                                    batch_sizes: List[int], 
                                    throughputs: List[float]) -> Dict:
        """
        Calculate scalability metrics based on throughput at different batch sizes
        
        Args:
            batch_sizes: List of batch sizes tested
            throughputs: List of corresponding throughputs (samples/second)
            
        Returns:
            Dictionary containing scalability metrics
        """
        metrics = {}
        
        # Linear scaling efficiency
        reference_throughput = throughputs[0]
        reference_batch = batch_sizes[0]
        
        scaling_efficiency = []
        for batch, throughput in zip(batch_sizes, throughputs):
            expected_throughput = (batch / reference_batch) * reference_throughput
            efficiency = throughput / expected_throughput
            scaling_efficiency.append(efficiency)
            
        metrics['avg_scaling_efficiency'] = np.mean(scaling_efficiency)
        metrics['scaling_efficiency_curve'] = scaling_efficiency
        
        # Throughput analysis
        metrics['max_throughput'] = max(throughputs)
        metrics['throughput_batch_size_ratio'] = [t/b for t, b in zip(throughputs, batch_sizes)]
        
        return metrics
    
    def plot_metrics(self, save_path: Optional[str] = None):
        """Plot training metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curve
        axes[0,0].plot(self.epoch_times, self.loss_history)
        axes[0,0].set_title('Loss vs Time')
        axes[0,0].set_xlabel('Time (s)')
        axes[0,0].set_ylabel('Loss')
        
        # GPU utilization
        if self.gpu_utilization:
            axes[0,1].plot(self.epoch_times, self.gpu_utilization)
            axes[0,1].set_title('GPU Utilization vs Time')
            axes[0,1].set_xlabel('Time (s)')
            axes[0,1].set_ylabel('GPU Utilization (%)')
            
        # Memory usage
        if self.memory_usage:
            memory_gb = [m/1e9 for m in self.memory_usage]
            axes[1,0].plot(self.epoch_times, memory_gb)
            axes[1,0].set_title('Memory Usage vs Time')
            axes[1,0].set_xlabel('Time (s)')
            axes[1,0].set_ylabel('Memory Usage (GB)')
            
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

# Example usage:
def example_usage():
    metrics = ComputationMetrics()
    
    # Simulate training loop
    for epoch in range(10):
        # Simulate epoch training
        time.sleep(0.5)
        fake_loss = 1.0 / (epoch + 1)
        metrics.log_epoch(fake_loss)
    
    # Calculate metrics
    convergence_metrics = metrics.calculate_convergence_metrics()
    resource_metrics = metrics.calculate_resource_metrics()
    
    # Example scalability data
    batch_sizes = [32, 64, 128, 256]
    throughputs = [100, 180, 320, 580]
    scalability_metrics = metrics.calculate_scalability_metrics(batch_sizes, throughputs)
    
    # Plot results
    metrics.plot_metrics("training_metrics.png")
    
    return {
        'convergence': convergence_metrics,
        'resources': resource_metrics,
        'scalability': scalability_metrics
    }

if __name__ == "__main__":
    results = example_usage()
    print("Metrics:", results)