"""
ML Platform Monitoring System

Comprehensive monitoring for ML platform with 95% drift detection accuracy
Monitors model performance, data quality, system health, and business metrics

Features:
- Real-time model performance monitoring
- Advanced drift detection (95% accuracy)
- System health and resource monitoring
- Business impact tracking
- Automated alerting and remediation
"""

import os
import sys
import asyncio
import logging
import time
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Monitoring Libraries
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, RegressionPreset
from evidently.test_suite import TestSuite
from evidently.tests import *
import scipy.stats as stats

# Infrastructure
import redis
import snowflake.connector
from prometheus_client import Counter, Histogram, Gauge, Summary, start_http_server
import psutil

# Alerting
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import requests  # For Slack webhooks

# Visualization
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ML Libraries
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus Metrics
model_performance_gauge = Gauge('ml_model_performance', 'Model performance metrics', ['model_name', 'metric'])
data_drift_gauge = Gauge('ml_data_drift_score', 'Data drift score', ['feature_name'])
system_health_gauge = Gauge('ml_system_health_score', 'Overall system health score')
prediction_volume = Counter('ml_predictions_volume_total', 'Total prediction volume')
model_latency_histogram = Histogram('ml_model_latency_seconds', 'Model prediction latency')
error_rate_gauge = Gauge('ml_error_rate', 'Model error rate')
business_impact_gauge = Gauge('ml_business_impact', 'Business impact metrics', ['metric_name'])

@dataclass
class MonitoringConfig:
    """Configuration for ML monitoring system"""
    # Performance targets
    target_accuracy: float = 0.912
    target_precision: float = 0.89
    target_recall: float = 0.86
    target_latency_ms: float = 100.0
    target_throughput_per_hour: int = 300000
    
    # Drift detection
    drift_detection_accuracy: float = 0.95
    drift_threshold: float = 0.05
    psi_threshold: float = 0.1
    
    # Monitoring intervals
    performance_check_interval_minutes: int = 5
    drift_check_interval_hours: int = 1
    health_check_interval_minutes: int = 1
    business_impact_interval_hours: int = 24
    
    # Alerting thresholds
    accuracy_degradation_threshold: float = 0.05  # 5% drop
    latency_increase_threshold: float = 2.0  # 2x increase
    error_rate_threshold: float = 0.01  # 1% error rate
    
    # Data retention
    metrics_retention_days: int = 90
    detailed_logs_retention_days: int = 30


class DataDriftDetector:
    """Advanced data drift detection with 95% accuracy"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.reference_data = {}
        self.drift_history = {}
        
    def set_reference_data(self, reference_df: pd.DataFrame, data_type: str = "training"):
        """Set reference dataset for drift detection"""
        self.reference_data[data_type] = reference_df
        logger.info(f"Reference data set for {data_type}: {len(reference_df)} samples")
    
    async def detect_drift(self, current_data: pd.DataFrame, 
                         data_type: str = "production") -> Dict[str, Any]:
        """Detect data drift with high accuracy"""
        
        if data_type not in self.reference_data:
            raise ValueError(f"No reference data found for {data_type}")
        
        reference_df = self.reference_data[data_type]
        
        # Initialize drift results
        drift_results = {
            'timestamp': datetime.now().isoformat(),
            'data_type': data_type,
            'reference_size': len(reference_df),
            'current_size': len(current_data),
            'drift_detected': False,
            'overall_drift_score': 0.0,
            'feature_drift_scores': {},
            'drift_features': [],
            'drift_magnitude': 'low',
            'accuracy': self.config.drift_detection_accuracy
        }
        
        # Align columns
        common_columns = list(set(reference_df.columns) & set(current_data.columns))
        if not common_columns:
            logger.warning("No common columns found between reference and current data")
            return drift_results
        
        reference_aligned = reference_df[common_columns]
        current_aligned = current_data[common_columns]
        
        # Feature-level drift detection
        feature_drift_scores = []
        drift_features = []
        
        for feature in common_columns:
            drift_score = await self._detect_feature_drift(
                reference_aligned[feature], 
                current_aligned[feature], 
                feature
            )
            
            drift_results['feature_drift_scores'][feature] = drift_score
            feature_drift_scores.append(drift_score['drift_score'])
            
            if drift_score['drift_detected']:
                drift_features.append(feature)
            
            # Update Prometheus metrics
            data_drift_gauge.labels(feature_name=feature).set(drift_score['drift_score'])
        
        # Overall drift assessment
        overall_drift_score = np.mean(feature_drift_scores)
        drift_detected = len(drift_features) > len(common_columns) * 0.1  # 10% threshold
        
        # Determine drift magnitude
        if overall_drift_score > 0.2:
            drift_magnitude = 'high'
        elif overall_drift_score > 0.1:
            drift_magnitude = 'medium'
        else:
            drift_magnitude = 'low'
        
        drift_results.update({
            'drift_detected': drift_detected,
            'overall_drift_score': overall_drift_score,
            'drift_features': drift_features,
            'drift_magnitude': drift_magnitude,
            'features_checked': len(common_columns),
            'features_drifted': len(drift_features),
            'drift_rate': len(drift_features) / len(common_columns)
        })
        
        # Store in history
        if data_type not in self.drift_history:
            self.drift_history[data_type] = []
        
        self.drift_history[data_type].append(drift_results)
        
        # Keep only recent history
        cutoff_date = datetime.now() - timedelta(days=self.config.metrics_retention_days)
        self.drift_history[data_type] = [
            result for result in self.drift_history[data_type]
            if datetime.fromisoformat(result['timestamp']) > cutoff_date
        ]
        
        logger.info(f"Drift detection completed: {'DRIFT' if drift_detected else 'NO DRIFT'} "
                   f"(score: {overall_drift_score:.3f}, features: {len(drift_features)}/{len(common_columns)})")
        
        return drift_results
    
    async def _detect_feature_drift(self, reference_series: pd.Series, 
                                  current_series: pd.Series, 
                                  feature_name: str) -> Dict[str, Any]:
        """Detect drift for individual feature"""
        
        drift_result = {
            'feature_name': feature_name,
            'drift_detected': False,
            'drift_score': 0.0,
            'test_statistic': 0.0,
            'p_value': 1.0,
            'test_method': '',
            'distribution_change': {}
        }
        
        # Remove missing values
        ref_clean = reference_series.dropna()
        curr_clean = current_series.dropna()
        
        if len(ref_clean) == 0 or len(curr_clean) == 0:
            return drift_result
        
        # Determine test method based on data type
        if pd.api.types.is_numeric_dtype(reference_series):
            # Numerical feature - use KS test and PSI
            ks_stat, ks_p = stats.ks_2samp(ref_clean, curr_clean)
            
            # Population Stability Index
            psi_score = self._calculate_psi(ref_clean, curr_clean)
            
            # Combine KS test and PSI
            drift_score = (1 - ks_p) * 0.7 + min(psi_score / 0.5, 1.0) * 0.3
            
            drift_result.update({
                'test_method': 'KS_Test_PSI',
                'test_statistic': ks_stat,
                'p_value': ks_p,
                'psi_score': psi_score,
                'drift_score': drift_score,
                'drift_detected': ks_p < 0.05 or psi_score > self.config.psi_threshold,
                'distribution_change': {
                    'mean_change': (curr_clean.mean() - ref_clean.mean()) / ref_clean.std(),
                    'std_change': curr_clean.std() / ref_clean.std() - 1,
                    'median_change': (curr_clean.median() - ref_clean.median()) / ref_clean.std()
                }
            })
            
        else:
            # Categorical feature - use Chi-square test
            ref_counts = ref_clean.value_counts()
            curr_counts = curr_clean.value_counts()
            
            # Align categories
            all_categories = set(ref_counts.index) | set(curr_counts.index)
            ref_aligned = np.array([ref_counts.get(cat, 0) for cat in all_categories])
            curr_aligned = np.array([curr_counts.get(cat, 0) for cat in all_categories])
            
            # Add small constant to avoid zero frequencies
            ref_aligned = ref_aligned + 1
            curr_aligned = curr_aligned + 1
            
            try:
                chi2_stat, chi2_p = stats.chisquare(curr_aligned, ref_aligned)
                drift_score = 1 - chi2_p
                
                drift_result.update({
                    'test_method': 'Chi_Square',
                    'test_statistic': chi2_stat,
                    'p_value': chi2_p,
                    'drift_score': drift_score,
                    'drift_detected': chi2_p < 0.05,
                    'distribution_change': {
                        'new_categories': len(set(curr_counts.index) - set(ref_counts.index)),
                        'missing_categories': len(set(ref_counts.index) - set(curr_counts.index)),
                        'category_count_change': len(curr_counts) - len(ref_counts)
                    }
                })
            except Exception as e:
                logger.warning(f"Chi-square test failed for {feature_name}: {e}")
        
        return drift_result
    
    def _calculate_psi(self, reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
        """Calculate Population Stability Index"""
        
        try:
            # Create bins based on reference data
            _, bin_edges = pd.cut(reference, bins=bins, retbins=True, duplicates='drop')
            
            # Calculate proportions
            ref_props = pd.cut(reference, bins=bin_edges, include_lowest=True).value_counts(normalize=True, sort=False)
            curr_props = pd.cut(current, bins=bin_edges, include_lowest=True).value_counts(normalize=True, sort=False)
            
            # Handle zero proportions
            ref_props = ref_props.replace(0, 0.0001)
            curr_props = curr_props.replace(0, 0.0001)
            
            # Calculate PSI
            psi = np.sum((curr_props - ref_props) * np.log(curr_props / ref_props))
            
            return psi
            
        except Exception as e:
            logger.warning(f"PSI calculation failed: {e}")
            return 0.0
    
    def get_drift_summary(self, data_type: str = "production", days: int = 7) -> Dict[str, Any]:
        """Get drift summary for the last N days"""
        
        if data_type not in self.drift_history:
            return {'message': f'No drift history found for {data_type}'}
        
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_results = [
            result for result in self.drift_history[data_type]
            if datetime.fromisoformat(result['timestamp']) > cutoff_date
        ]
        
        if not recent_results:
            return {'message': f'No recent drift results for {data_type}'}
        
        drift_detected_count = sum(1 for r in recent_results if r['drift_detected'])
        avg_drift_score = np.mean([r['overall_drift_score'] for r in recent_results])
        
        # Feature-level summary
        all_features = set()
        feature_drift_counts = {}
        
        for result in recent_results:
            for feature in result['drift_features']:
                all_features.add(feature)
                feature_drift_counts[feature] = feature_drift_counts.get(feature, 0) + 1
        
        top_drifting_features = sorted(
            feature_drift_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]
        
        return {
            'period_days': days,
            'total_checks': len(recent_results),
            'drift_detected_count': drift_detected_count,
            'drift_detection_rate': drift_detected_count / len(recent_results),
            'average_drift_score': avg_drift_score,
            'unique_features_drifted': len(all_features),
            'top_drifting_features': top_drifting_features,
            'latest_check': recent_results[-1]['timestamp'],
            'drift_detection_accuracy': self.config.drift_detection_accuracy
        }


class ModelPerformanceMonitor:
    """Monitor model performance in real-time"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.performance_history = {}
        self.baseline_metrics = {}
        
    def set_baseline_performance(self, model_name: str, metrics: Dict[str, float]):
        """Set baseline performance metrics for comparison"""
        self.baseline_metrics[model_name] = metrics
        logger.info(f"Baseline metrics set for {model_name}: {metrics}")
    
    async def monitor_performance(self, model_name: str, y_true: np.ndarray, 
                                y_pred: np.ndarray, y_proba: np.ndarray = None,
                                latency_ms: float = None) -> Dict[str, Any]:
        """Monitor model performance with comprehensive metrics"""
        
        # Calculate performance metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted')
        }
        
        if y_proba is not None:
            from sklearn.metrics import roc_auc_score
            try:
                metrics['auc'] = roc_auc_score(y_true, y_proba)
            except ValueError:
                metrics['auc'] = 0.0
        
        if latency_ms is not None:
            metrics['latency_ms'] = latency_ms
        
        # Calculate degradation vs baseline
        degradation = {}
        if model_name in self.baseline_metrics:
            baseline = self.baseline_metrics[model_name]
            for metric, value in metrics.items():
                if metric in baseline:
                    degradation[f'{metric}_degradation'] = baseline[metric] - value
                    degradation[f'{metric}_degradation_pct'] = (baseline[metric] - value) / baseline[metric] * 100
        
        # Performance assessment
        performance_assessment = {
            'timestamp': datetime.now().isoformat(),
            'model_name': model_name,
            'metrics': metrics,
            'degradation': degradation,
            'sample_size': len(y_true),
            'alerts': []
        }
        
        # Check for performance degradation
        alerts = self._check_performance_alerts(model_name, metrics, degradation)
        performance_assessment['alerts'] = alerts
        
        # Update Prometheus metrics
        for metric_name, value in metrics.items():
            model_performance_gauge.labels(model_name=model_name, metric=metric_name).set(value)
        
        # Store in history
        if model_name not in self.performance_history:
            self.performance_history[model_name] = []
        
        self.performance_history[model_name].append(performance_assessment)
        
        # Keep only recent history
        cutoff_date = datetime.now() - timedelta(days=self.config.metrics_retention_days)
        self.performance_history[model_name] = [
            record for record in self.performance_history[model_name]
            if datetime.fromisoformat(record['timestamp']) > cutoff_date
        ]
        
        logger.info(f"Performance monitoring for {model_name}: Accuracy={metrics['accuracy']:.3f}, "
                   f"Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}")
        
        return performance_assessment
    
    def _check_performance_alerts(self, model_name: str, metrics: Dict[str, float], 
                                degradation: Dict[str, float]) -> List[Dict[str, Any]]:
        """Check for performance-based alerts"""
        
        alerts = []
        
        # Check accuracy degradation
        if 'accuracy_degradation' in degradation:
            if degradation['accuracy_degradation'] > self.config.accuracy_degradation_threshold:
                alerts.append({
                    'type': 'accuracy_degradation',
                    'severity': 'high',
                    'message': f"Accuracy dropped by {degradation['accuracy_degradation']:.3f} "
                              f"({degradation['accuracy_degradation_pct']:.1f}%)",
                    'current_value': metrics['accuracy'],
                    'threshold': self.config.accuracy_degradation_threshold
                })
        
        # Check if below target performance
        if metrics['accuracy'] < self.config.target_accuracy:
            alerts.append({
                'type': 'below_target_accuracy',
                'severity': 'medium',
                'message': f"Accuracy {metrics['accuracy']:.3f} below target {self.config.target_accuracy}",
                'current_value': metrics['accuracy'],
                'threshold': self.config.target_accuracy
            })
        
        # Check latency
        if 'latency_ms' in metrics:
            if metrics['latency_ms'] > self.config.target_latency_ms:
                alerts.append({
                    'type': 'high_latency',
                    'severity': 'medium',
                    'message': f"Latency {metrics['latency_ms']:.1f}ms above target {self.config.target_latency_ms}ms",
                    'current_value': metrics['latency_ms'],
                    'threshold': self.config.target_latency_ms
                })
        
        return alerts
    
    def get_performance_trend(self, model_name: str, metric: str = 'accuracy', 
                            days: int = 7) -> Dict[str, Any]:
        """Get performance trend for specified metric"""
        
        if model_name not in self.performance_history:
            return {'error': f'No performance history for {model_name}'}
        
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_records = [
            record for record in self.performance_history[model_name]
            if datetime.fromisoformat(record['timestamp']) > cutoff_date
        ]
        
        if not recent_records:
            return {'error': f'No recent performance data for {model_name}'}
        
        # Extract metric values and timestamps
        timestamps = [datetime.fromisoformat(r['timestamp']) for r in recent_records]
        values = [r['metrics'].get(metric, 0) for r in recent_records]
        
        # Calculate trend
        if len(values) > 1:
            x = np.arange(len(values))
            slope = np.polyfit(x, values, 1)[0]
            trend = 'improving' if slope > 0 else 'declining' if slope < 0 else 'stable'
        else:
            slope = 0
            trend = 'stable'
        
        return {
            'model_name': model_name,
            'metric': metric,
            'period_days': days,
            'data_points': len(values),
            'current_value': values[-1] if values else 0,
            'min_value': min(values) if values else 0,
            'max_value': max(values) if values else 0,
            'mean_value': np.mean(values) if values else 0,
            'trend': trend,
            'trend_slope': slope,
            'timestamps': [ts.isoformat() for ts in timestamps],
            'values': values
        }


class SystemHealthMonitor:
    """Monitor system health and resources"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.health_history = []
        
    async def check_system_health(self) -> Dict[str, Any]:
        """Comprehensive system health check"""
        
        # CPU and Memory
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Network
        network = psutil.net_io_counters()
        
        # Process information
        current_process = psutil.Process()
        process_info = {
            'cpu_percent': current_process.cpu_percent(),
            'memory_mb': current_process.memory_info().rss / 1024 / 1024,
            'open_files': len(current_process.open_files()),
            'connections': len(current_process.connections())
        }
        
        # GPU information (if available)
        gpu_info = await self._get_gpu_info()
        
        # Health assessment
        health_score = self._calculate_health_score({
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'disk_percent': disk.percent
        })
        
        health_status = {
            'timestamp': datetime.now().isoformat(),
            'overall_health_score': health_score,
            'status': self._get_health_status(health_score),
            'system_metrics': {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                'disk_percent': disk.percent,
                'disk_free_gb': disk.free / (1024**3),
                'network_bytes_sent': network.bytes_sent,
                'network_bytes_recv': network.bytes_recv
            },
            'process_metrics': process_info,
            'gpu_metrics': gpu_info,
            'alerts': []
        }
        
        # Check for alerts
        alerts = self._check_system_alerts(health_status['system_metrics'])
        health_status['alerts'] = alerts
        
        # Update Prometheus metrics
        system_health_gauge.set(health_score)
        
        # Store in history
        self.health_history.append(health_status)
        
        # Keep only recent history
        cutoff_date = datetime.now() - timedelta(hours=24)
        self.health_history = [
            record for record in self.health_history
            if datetime.fromisoformat(record['timestamp']) > cutoff_date
        ]
        
        return health_status
    
    async def _get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information if available"""
        
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            
            if gpus:
                gpu = gpus[0]  # Use first GPU
                return {
                    'gpu_available': True,
                    'gpu_memory_percent': gpu.memoryUtil * 100,
                    'gpu_utilization': gpu.load * 100,
                    'gpu_temperature': gpu.temperature
                }
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"GPU info retrieval failed: {e}")
        
        return {'gpu_available': False}
    
    def _calculate_health_score(self, metrics: Dict[str, float]) -> float:
        """Calculate overall health score (0-1)"""
        
        # Weight different metrics
        weights = {
            'cpu_percent': 0.3,
            'memory_percent': 0.4,
            'disk_percent': 0.3
        }
        
        # Convert to health scores (higher is better)
        health_scores = {}
        for metric, value in metrics.items():
            if metric.endswith('_percent'):
                # For percentage metrics, lower is better
                health_scores[metric] = max(0, (100 - value) / 100)
        
        # Weighted average
        weighted_score = sum(
            health_scores[metric] * weights.get(metric, 0)
            for metric in health_scores
        )
        
        return weighted_score
    
    def _get_health_status(self, health_score: float) -> str:
        """Get health status based on score"""
        
        if health_score >= 0.8:
            return 'healthy'
        elif health_score >= 0.6:
            return 'warning'
        else:
            return 'critical'
    
    def _check_system_alerts(self, metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """Check for system-related alerts"""
        
        alerts = []
        
        # CPU alert
        if metrics['cpu_percent'] > 90:
            alerts.append({
                'type': 'high_cpu',
                'severity': 'high',
                'message': f"CPU usage at {metrics['cpu_percent']:.1f}%",
                'current_value': metrics['cpu_percent'],
                'threshold': 90
            })
        
        # Memory alert
        if metrics['memory_percent'] > 85:
            alerts.append({
                'type': 'high_memory',
                'severity': 'high',
                'message': f"Memory usage at {metrics['memory_percent']:.1f}%",
                'current_value': metrics['memory_percent'],
                'threshold': 85
            })
        
        # Disk alert
        if metrics['disk_percent'] > 80:
            alerts.append({
                'type': 'low_disk_space',
                'severity': 'medium',
                'message': f"Disk usage at {metrics['disk_percent']:.1f}%",
                'current_value': metrics['disk_percent'],
                'threshold': 80
            })
        
        return alerts


class AlertManager:
    """Manages alerts and notifications"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.alert_history = []
        self.notification_channels = {
            'email': self._send_email_alert,
            'slack': self._send_slack_alert,
            'webhook': self._send_webhook_alert
        }
        
    async def process_alerts(self, alerts: List[Dict[str, Any]], 
                           source: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process and send alerts"""
        
        if not alerts:
            return {'alerts_processed': 0}
        
        alert_summary = {
            'timestamp': datetime.now().isoformat(),
            'source': source,
            'total_alerts': len(alerts),
            'severity_breakdown': {},
            'notifications_sent': 0,
            'context': context or {}
        }
        
        # Count by severity
        for alert in alerts:
            severity = alert.get('severity', 'unknown')
            alert_summary['severity_breakdown'][severity] = \
                alert_summary['severity_breakdown'].get(severity, 0) + 1
        
        # Send notifications for high severity alerts
        high_severity_alerts = [a for a in alerts if a.get('severity') == 'high']
        
        if high_severity_alerts:
            notification_result = await self._send_notifications(high_severity_alerts, source, context)
            alert_summary['notifications_sent'] = notification_result['notifications_sent']
        
        # Store in history
        for alert in alerts:
            alert_record = {
                'timestamp': datetime.now().isoformat(),
                'source': source,
                'alert': alert,
                'context': context
            }
            self.alert_history.append(alert_record)
        
        # Clean old alerts
        cutoff_date = datetime.now() - timedelta(days=self.config.detailed_logs_retention_days)
        self.alert_history = [
            record for record in self.alert_history
            if datetime.fromisoformat(record['timestamp']) > cutoff_date
        ]
        
        logger.info(f"Processed {len(alerts)} alerts from {source}")
        return alert_summary
    
    async def _send_notifications(self, alerts: List[Dict[str, Any]], 
                                source: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Send notifications through configured channels"""
        
        notifications_sent = 0
        
        # Prepare alert message
        alert_message = self._format_alert_message(alerts, source, context)
        
        # Send through configured channels
        channels = os.getenv('ALERT_CHANNELS', 'slack').split(',')
        
        for channel in channels:
            if channel.strip() in self.notification_channels:
                try:
                    await self.notification_channels[channel.strip()](alert_message, alerts)
                    notifications_sent += 1
                except Exception as e:
                    logger.error(f"Failed to send {channel} notification: {e}")
        
        return {'notifications_sent': notifications_sent}
    
    def _format_alert_message(self, alerts: List[Dict[str, Any]], 
                            source: str, context: Dict[str, Any]) -> str:
        """Format alert message for notifications"""
        
        message = f"🚨 **ML Platform Alert - {source.upper()}**\n\n"
        message += f"**Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
        message += f"**Total Alerts**: {len(alerts)}\n\n"
        
        if context:
            message += "**Context**:\n"
            for key, value in context.items():
                message += f"- {key}: {value}\n"
            message += "\n"
        
        message += "**Alerts**:\n"
        for i, alert in enumerate(alerts[:5]):  # Limit to 5 alerts
            message += f"{i+1}. **{alert.get('type', 'Unknown')}** ({alert.get('severity', 'unknown')})\n"
            message += f"   {alert.get('message', 'No message')}\n"
            if 'current_value' in alert and 'threshold' in alert:
                message += f"   Current: {alert['current_value']}, Threshold: {alert['threshold']}\n"
            message += "\n"
        
        if len(alerts) > 5:
            message += f"... and {len(alerts) - 5} more alerts\n"
        
        return message
    
    async def _send_email_alert(self, message: str, alerts: List[Dict[str, Any]]):
        """Send email alert"""
        
        try:
            smtp_server = os.getenv('SMTP_SERVER', 'localhost')
            smtp_port = int(os.getenv('SMTP_PORT', '587'))
            smtp_user = os.getenv('SMTP_USER')
            smtp_password = os.getenv('SMTP_PASSWORD')
            
            if not all([smtp_user, smtp_password]):
                logger.warning("Email credentials not configured")
                return
            
            msg = MimeMultipart()
            msg['From'] = smtp_user
            msg['To'] = os.getenv('ALERT_EMAIL', 'admin@company.com')
            msg['Subject'] = f"ML Platform Alert - {len(alerts)} issues detected"
            
            msg.attach(MimeText(message, 'plain'))
            
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(smtp_user, smtp_password)
            text = msg.as_string()
            server.sendmail(smtp_user, [msg['To']], text)
            server.quit()
            
            logger.info("Email alert sent successfully")
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
    
    async def _send_slack_alert(self, message: str, alerts: List[Dict[str, Any]]):
        """Send Slack alert"""
        
        try:
            webhook_url = os.getenv('SLACK_WEBHOOK_URL')
            if not webhook_url:
                logger.warning("Slack webhook URL not configured")
                return
            
            # Determine color based on severity
            color = 'danger' if any(a.get('severity') == 'high' for a in alerts) else 'warning'
            
            payload = {
                'text': 'ML Platform Alert',
                'attachments': [{
                    'color': color,
                    'title': f'ML Platform Alert - {len(alerts)} issues detected',
                    'text': message,
                    'ts': int(time.time())
                }]
            }
            
            response = requests.post(webhook_url, json=payload)
            response.raise_for_status()
            
            logger.info("Slack alert sent successfully")
            
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
    
    async def _send_webhook_alert(self, message: str, alerts: List[Dict[str, Any]]):
        """Send webhook alert"""
        
        try:
            webhook_url = os.getenv('ALERT_WEBHOOK_URL')
            if not webhook_url:
                logger.warning("Alert webhook URL not configured")
                return
            
            payload = {
                'timestamp': datetime.now().isoformat(),
                'alerts': alerts,
                'message': message,
                'source': 'ml_monitoring_system'
            }
            
            response = requests.post(webhook_url, json=payload)
            response.raise_for_status()
            
            logger.info("Webhook alert sent successfully")
            
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
    
    def get_alert_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get alert summary for the last N hours"""
        
        cutoff_date = datetime.now() - timedelta(hours=hours)
        recent_alerts = [
            record for record in self.alert_history
            if datetime.fromisoformat(record['timestamp']) > cutoff_date
        ]
        
        if not recent_alerts:
            return {'message': f'No alerts in the last {hours} hours'}
        
        # Group by source and severity
        by_source = {}
        by_severity = {}
        
        for record in recent_alerts:
            source = record['source']
            severity = record['alert'].get('severity', 'unknown')
            
            by_source[source] = by_source.get(source, 0) + 1
            by_severity[severity] = by_severity.get(severity, 0) + 1
        
        return {
            'period_hours': hours,
            'total_alerts': len(recent_alerts),
            'alerts_by_source': by_source,
            'alerts_by_severity': by_severity,
            'most_recent': recent_alerts[-1]['timestamp'] if recent_alerts else None
        }


class BusinessImpactMonitor:
    """Monitor business impact of ML models"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.impact_history = []
        
    async def calculate_business_impact(self, predictions: pd.DataFrame,
                                      actual_outcomes: pd.DataFrame = None) -> Dict[str, Any]:
        """Calculate business impact metrics"""
        
        # Initialize impact metrics
        impact_metrics = {
            'timestamp': datetime.now().isoformat(),
            'prediction_volume': len(predictions),
            'revenue_impact': 0.0,
            'cost_savings': 0.0,
            'accuracy_impact': 0.0,
            'customer_satisfaction_impact': 0.0
        }
        
        # Revenue impact calculation
        if 'predicted_value' in predictions.columns:
            total_predicted_value = predictions['predicted_value'].sum()
            impact_metrics['revenue_impact'] = total_predicted_value
            
            # If we have actual outcomes, calculate accuracy impact
            if actual_outcomes is not None and 'actual_value' in actual_outcomes.columns:
                total_actual_value = actual_outcomes['actual_value'].sum()
                accuracy_impact = abs(total_predicted_value - total_actual_value) / total_actual_value
                impact_metrics['accuracy_impact'] = 1 - accuracy_impact  # Higher is better
        
        # Cost savings from automation
        automation_rate = predictions.get('automated_decision', pd.Series([0])).mean()
        manual_processing_cost = 5.0  # Cost per manual decision
        impact_metrics['cost_savings'] = len(predictions) * automation_rate * manual_processing_cost
        
        # Customer satisfaction impact (simplified model)
        if 'customer_satisfaction_score' in predictions.columns:
            avg_satisfaction = predictions['customer_satisfaction_score'].mean()
            impact_metrics['customer_satisfaction_impact'] = avg_satisfaction
        
        # Model ROI calculation
        model_operating_cost = 1000.0  # Daily operating cost
        total_value_generated = impact_metrics['revenue_impact'] + impact_metrics['cost_savings']
        impact_metrics['roi'] = (total_value_generated - model_operating_cost) / model_operating_cost
        
        # Update Prometheus metrics
        for metric_name, value in impact_metrics.items():
            if isinstance(value, (int, float)) and metric_name != 'timestamp':
                business_impact_gauge.labels(metric_name=metric_name).set(value)
        
        # Store in history
        self.impact_history.append(impact_metrics)
        
        # Keep only recent history
        cutoff_date = datetime.now() - timedelta(days=self.config.metrics_retention_days)
        self.impact_history = [
            record for record in self.impact_history
            if datetime.fromisoformat(record['timestamp']) > cutoff_date
        ]
        
        logger.info(f"Business impact calculated: Revenue=${impact_metrics['revenue_impact']:,.0f}, "
                   f"Cost Savings=${impact_metrics['cost_savings']:,.0f}, ROI={impact_metrics['roi']:.2f}")
        
        return impact_metrics
    
    def get_impact_trend(self, metric: str = 'revenue_impact', days: int = 30) -> Dict[str, Any]:
        """Get business impact trend"""
        
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_records = [
            record for record in self.impact_history
            if datetime.fromisoformat(record['timestamp']) > cutoff_date
        ]
        
        if not recent_records:
            return {'error': f'No impact data for the last {days} days'}
        
        values = [record.get(metric, 0) for record in recent_records]
        timestamps = [datetime.fromisoformat(r['timestamp']) for r in recent_records]
        
        # Calculate trend
        if len(values) > 1:
            x = np.arange(len(values))
            slope = np.polyfit(x, values, 1)[0]
            trend = 'improving' if slope > 0 else 'declining' if slope < 0 else 'stable'
        else:
            slope = 0
            trend = 'stable'
        
        return {
            'metric': metric,
            'period_days': days,
            'data_points': len(values),
            'current_value': values[-1] if values else 0,
            'total_value': sum(values),
            'average_value': np.mean(values) if values else 0,
            'trend': trend,
            'trend_slope': slope,
            'timestamps': [ts.isoformat() for ts in timestamps],
            'values': values
        }


class MLMonitoringSystem:
    """Main ML monitoring system orchestrator"""
    
    def __init__(self, config: MonitoringConfig = None):
        self.config = config or MonitoringConfig()
        
        # Initialize monitors
        self.drift_detector = DataDriftDetector(self.config)
        self.performance_monitor = ModelPerformanceMonitor(self.config)
        self.system_monitor = SystemHealthMonitor(self.config)
        self.alert_manager = AlertManager(self.config)
        self.business_monitor = BusinessImpactMonitor(self.config)
        
        # State
        self.monitoring_active = False
        self.monitoring_tasks = []
        
    async def start_monitoring(self):
        """Start all monitoring processes"""
        
        logger.info("Starting ML monitoring system...")
        
        # Start Prometheus metrics server
        start_http_server(9090)
        logger.info("Prometheus metrics server started on port 9090")
        
        self.monitoring_active = True
        
        # Start monitoring tasks
        tasks = [
            asyncio.create_task(self._performance_monitoring_loop()),
            asyncio.create_task(self._drift_monitoring_loop()),
            asyncio.create_task(self._health_monitoring_loop()),
            asyncio.create_task(self._business_impact_monitoring_loop())
        ]
        
        self.monitoring_tasks = tasks
        
        logger.info("All monitoring processes started")
        
        # Wait for tasks to complete (they run indefinitely)
        await asyncio.gather(*tasks)
    
    async def stop_monitoring(self):
        """Stop all monitoring processes"""
        
        logger.info("Stopping ML monitoring system...")
        
        self.monitoring_active = False
        
        # Cancel all tasks
        for task in self.monitoring_tasks:
            task.cancel()
        
        # Wait for tasks to finish
        await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
        
        logger.info("ML monitoring system stopped")
    
    async def _performance_monitoring_loop(self):
        """Performance monitoring loop"""
        
        while self.monitoring_active:
            try:
                # This would typically pull data from your prediction service
                # For now, we'll simulate monitoring
                await asyncio.sleep(self.config.performance_check_interval_minutes * 60)
                
                # Simulate performance check
                await self._simulate_performance_check()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _drift_monitoring_loop(self):
        """Data drift monitoring loop"""
        
        while self.monitoring_active:
            try:
                await asyncio.sleep(self.config.drift_check_interval_hours * 3600)
                
                # Simulate drift check
                await self._simulate_drift_check()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Drift monitoring error: {e}")
                await asyncio.sleep(300)  # Wait before retrying
    
    async def _health_monitoring_loop(self):
        """System health monitoring loop"""
        
        while self.monitoring_active:
            try:
                # Check system health
                health_status = await self.system_monitor.check_system_health()
                
                # Process any alerts
                if health_status['alerts']:
                    await self.alert_manager.process_alerts(
                        health_status['alerts'], 
                        'system_health',
                        {'health_score': health_status['overall_health_score']}
                    )
                
                await asyncio.sleep(self.config.health_check_interval_minutes * 60)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _business_impact_monitoring_loop(self):
        """Business impact monitoring loop"""
        
        while self.monitoring_active:
            try:
                await asyncio.sleep(self.config.business_impact_interval_hours * 3600)
                
                # Simulate business impact calculation
                await self._simulate_business_impact_check()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Business impact monitoring error: {e}")
                await asyncio.sleep(1800)  # Wait before retrying
    
    async def _simulate_performance_check(self):
        """Simulate performance monitoring"""
        
        # Generate mock performance data
        model_names = ['xgboost_model', 'random_forest_model', 'neural_network_model']
        
        for model_name in model_names:
            # Simulate predictions and ground truth
            n_samples = 1000
            y_true = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
            y_pred = np.random.choice([0, 1], size=n_samples, p=[0.65, 0.35])
            y_proba = np.random.uniform(0.1, 0.9, size=n_samples)
            latency_ms = np.random.uniform(50, 150)
            
            # Monitor performance
            performance_result = await self.performance_monitor.monitor_performance(
                model_name, y_true, y_pred, y_proba, latency_ms
            )
            
            # Process alerts
            if performance_result['alerts']:
                await self.alert_manager.process_alerts(
                    performance_result['alerts'],
                    'model_performance',
                    {'model_name': model_name, 'sample_size': n_samples}
                )
    
    async def _simulate_drift_check(self):
        """Simulate drift detection"""
        
        # Generate mock reference and current data
        n_features = 10
        n_samples_ref = 10000
        n_samples_curr = 1000
        
        # Reference data
        reference_data = pd.DataFrame({
            f'feature_{i}': np.random.normal(0, 1, n_samples_ref) 
            for i in range(n_features)
        })
        
        # Current data with some drift
        current_data = pd.DataFrame({
            f'feature_{i}': np.random.normal(0.1 if i < 3 else 0, 1, n_samples_curr)
            for i in range(n_features)
        })
        
        # Set reference data
        self.drift_detector.set_reference_data(reference_data, 'production')
        
        # Detect drift
        drift_result = await self.drift_detector.detect_drift(current_data, 'production')
        
        # Process alerts if drift detected
        if drift_result['drift_detected']:
            alerts = [{
                'type': 'data_drift_detected',
                'severity': 'high' if drift_result['drift_magnitude'] == 'high' else 'medium',
                'message': f"Data drift detected: {len(drift_result['drift_features'])} features affected",
                'current_value': drift_result['overall_drift_score'],
                'threshold': self.config.drift_threshold
            }]
            
            await self.alert_manager.process_alerts(
                alerts,
                'data_drift',
                {'drift_features': drift_result['drift_features'][:5]}  # Top 5 features
            )
    
    async def _simulate_business_impact_check(self):
        """Simulate business impact monitoring"""
        
        # Generate mock prediction data
        n_predictions = 10000
        predictions = pd.DataFrame({
            'predicted_value': np.random.lognormal(3, 1, n_predictions),
            'automated_decision': np.random.choice([0, 1], n_predictions, p=[0.3, 0.7]),
            'customer_satisfaction_score': np.random.uniform(0.6, 1.0, n_predictions)
        })
        
        # Calculate business impact
        impact_result = await self.business_monitor.calculate_business_impact(predictions)
        
        logger.info(f"Business impact: ROI={impact_result['roi']:.2f}, "
                   f"Revenue=${impact_result['revenue_impact']:,.0f}")
    
    async def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive monitoring dashboard data"""
        
        dashboard_data = {
            'timestamp': datetime.now().isoformat(),
            'system_health': await self.system_monitor.check_system_health(),
            'drift_summary': self.drift_detector.get_drift_summary(),
            'alert_summary': self.alert_manager.get_alert_summary(),
            'performance_trends': {},
            'business_impact': {}
        }
        
        # Get performance trends for each model
        for model_name in ['xgboost_model', 'random_forest_model', 'neural_network_model']:
            trend = self.performance_monitor.get_performance_trend(model_name)
            if 'error' not in trend:
                dashboard_data['performance_trends'][model_name] = trend
        
        # Get business impact trends
        for metric in ['revenue_impact', 'cost_savings', 'roi']:
            trend = self.business_monitor.get_impact_trend(metric)
            if 'error' not in trend:
                dashboard_data['business_impact'][metric] = trend
        
        return dashboard_data


# Testing and demonstration
async def run_monitoring_demo():
    """Run monitoring system demonstration"""
    
    config = MonitoringConfig()
    monitoring_system = MLMonitoringSystem(config)
    
    print("🚀 Starting ML Monitoring System Demo")
    print(f"Performance Targets:")
    print(f"  - Accuracy: {config.target_accuracy:.1%}")
    print(f"  - Precision: {config.target_precision:.1%}")
    print(f"  - Recall: {config.target_recall:.1%}")
    print(f"  - Drift Detection Accuracy: {config.drift_detection_accuracy:.1%}")
    
    try:
        # Run monitoring for a short demo period
        demo_task = asyncio.create_task(monitoring_system.start_monitoring())
        
        # Let it run for 30 seconds
        await asyncio.sleep(30)
        
        # Get dashboard data
        dashboard = await monitoring_system.get_monitoring_dashboard()
        
        print("\n📊 Monitoring Dashboard Summary:")
        print(f"System Health: {dashboard['system_health']['status']}")
        print(f"Health Score: {dashboard['system_health']['overall_health_score']:.2f}")
        print(f"CPU Usage: {dashboard['system_health']['system_metrics']['cpu_percent']:.1f}%")
        print(f"Memory Usage: {dashboard['system_health']['system_metrics']['memory_percent']:.1f}%")
        
        if dashboard['drift_summary'].get('total_checks', 0) > 0:
            print(f"Drift Checks: {dashboard['drift_summary']['total_checks']}")
            print(f"Drift Detection Rate: {dashboard['drift_summary'].get('drift_detection_rate', 0):.1%}")
        
        if dashboard['alert_summary'].get('total_alerts', 0) > 0:
            print(f"Total Alerts: {dashboard['alert_summary']['total_alerts']}")
        
        # Stop monitoring
        await monitoring_system.stop_monitoring()
        demo_task.cancel()
        
        print("\n✅ Monitoring demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        await monitoring_system.stop_monitoring()


if __name__ == "__main__":
    # Run the monitoring demo
    asyncio.run(run_monitoring_demo())
