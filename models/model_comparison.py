"""
Model Comparison Module
Provides functionality to compare different ML models side by side
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

class ModelComparison:
    """Model comparison and evaluation class"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
    
    def add_model(self, name, model):
        """
        Add a model to the comparison
        
        Args:
            name: Model name
            model: Model instance
        """
        self.models[name] = model
    
    def compare_models(self, data, target_column='close'):
        """
        Compare multiple models on the same data
        
        Args:
            data: DataFrame with time series data
            target_column: Column to predict
            
        Returns:
            comparison_results: Dictionary with comparison results
        """
        comparison_results = {}
        
        for name, model in self.models.items():
            try:
                if hasattr(model, 'is_trained') and model.is_trained:
                    # Get model predictions
                    if hasattr(model, 'predict'):
                        predictions = model.predict(data)
                    else:
                        predictions = None
                    
                    # Get trading signal
                    if hasattr(model, 'get_signal'):
                        signal, confidence = model.get_signal(data)
                    else:
                        signal, confidence = 'hold', 0.0
                    
                    # Evaluate model if evaluation method exists
                    if hasattr(model, 'evaluate'):
                        try:
                            metrics = model.evaluate(data, target_column)
                        except:
                            metrics = {}
                    else:
                        metrics = {}
                    
                    comparison_results[name] = {
                        'signal': signal,
                        'confidence': confidence,
                        'predictions': predictions,
                        'metrics': metrics,
                        'status': 'success'
                    }
                    
                else:
                    comparison_results[name] = {
                        'signal': 'hold',
                        'confidence': 0.0,
                        'predictions': None,
                        'metrics': {},
                        'status': 'not_trained'
                    }
                    
            except Exception as e:
                comparison_results[name] = {
                    'signal': 'hold',
                    'confidence': 0.0,
                    'predictions': None,
                    'metrics': {},
                    'status': f'error: {str(e)}'
                }
        
        return comparison_results
    
    def create_comparison_chart(self, data, comparison_results):
        """
        Create comparison chart for model predictions
        
        Args:
            data: DataFrame with time series data
            comparison_results: Results from compare_models
            
        Returns:
            fig: Plotly figure
        """
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Price and Predictions', 'Model Signals'),
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3]
        )
        
        # Plot actual price
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['close'],
                mode='lines',
                name='Actual Price',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        # Plot predictions for each model
        colors = ['red', 'green', 'orange', 'purple', 'brown']
        for i, (name, result) in enumerate(comparison_results.items()):
            if result['predictions'] is not None and len(result['predictions']) > 0:
                # Align predictions with data index
                pred_data = result['predictions']
                if len(pred_data) == len(data):
                    pred_index = data.index
                else:
                    # Assume predictions start from a certain point
                    pred_index = data.index[-len(pred_data):]
                
                fig.add_trace(
                    go.Scatter(
                        x=pred_index,
                        y=pred_data,
                        mode='lines',
                        name=f'{name} Prediction',
                        line=dict(color=colors[i % len(colors)], dash='dash')
                    ),
                    row=1, col=1
                )
        
        # Plot signals
        signal_colors = {'buy': 'green', 'sell': 'red', 'hold': 'gray'}
        model_names = list(comparison_results.keys())
        signals = [comparison_results[name]['signal'] for name in model_names]
        confidences = [comparison_results[name]['confidence'] for name in model_names]
        
        fig.add_trace(
            go.Bar(
                x=model_names,
                y=confidences,
                name='Signal Confidence',
                marker_color=[signal_colors.get(signal, 'gray') for signal in signals],
                text=signals,
                textposition='auto'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title='Model Comparison Dashboard',
            height=600,
            showlegend=True
        )
        
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_xaxes(title_text="Models", row=2, col=1)
        fig.update_yaxes(title_text="Confidence", row=2, col=1)
        
        return fig
    
    def create_metrics_table(self, comparison_results):
        """
        Create metrics comparison table
        
        Args:
            comparison_results: Results from compare_models
            
        Returns:
            df: DataFrame with metrics comparison
        """
        metrics_data = []
        
        for name, result in comparison_results.items():
            row = {'Model': name}
            
            # Add signal information
            row['Signal'] = result['signal']
            row['Confidence'] = f"{result['confidence']:.3f}"
            row['Status'] = result['status']
            
            # Add model metrics if available
            if result['metrics']:
                for metric_name, metric_value in result['metrics'].items():
                    if isinstance(metric_value, (int, float)):
                        row[metric_name] = f"{metric_value:.4f}"
                    else:
                        row[metric_name] = str(metric_value)
            
            metrics_data.append(row)
        
        return pd.DataFrame(metrics_data)
    
    def get_consensus_signal(self, comparison_results):
        """
        Get consensus signal from all models
        
        Args:
            comparison_results: Results from compare_models
            
        Returns:
            consensus_signal: Consensus trading signal
            consensus_confidence: Confidence in consensus
        """
        signals = []
        confidences = []
        
        for name, result in comparison_results.items():
            if result['status'] == 'success':
                signals.append(result['signal'])
                confidences.append(result['confidence'])
        
        if not signals:
            return 'hold', 0.0
        
        # Count votes
        signal_counts = {}
        weighted_counts = {}
        
        for signal, confidence in zip(signals, confidences):
            signal_counts[signal] = signal_counts.get(signal, 0) + 1
            weighted_counts[signal] = weighted_counts.get(signal, 0) + confidence
        
        # Get consensus based on weighted votes
        consensus_signal = max(weighted_counts, key=weighted_counts.get)
        
        # Calculate consensus confidence
        total_weight = sum(weighted_counts.values())
        consensus_confidence = weighted_counts[consensus_signal] / total_weight if total_weight > 0 else 0.0
        
        return consensus_signal, consensus_confidence
    
    def create_performance_summary(self, comparison_results):
        """
        Create performance summary for all models
        
        Args:
            comparison_results: Results from compare_models
            
        Returns:
            summary: Performance summary dictionary
        """
        summary = {
            'total_models': len(comparison_results),
            'trained_models': 0,
            'active_signals': {'buy': 0, 'sell': 0, 'hold': 0},
            'average_confidence': 0.0,
            'model_status': {}
        }
        
        total_confidence = 0
        active_models = 0
        
        for name, result in comparison_results.items():
            summary['model_status'][name] = result['status']
            
            if result['status'] == 'success':
                summary['trained_models'] += 1
                summary['active_signals'][result['signal']] += 1
                total_confidence += result['confidence']
                active_models += 1
        
        if active_models > 0:
            summary['average_confidence'] = total_confidence / active_models
        
        return summary
    
    def display_comparison_dashboard(self, data, comparison_results):
        """
        Display complete comparison dashboard in Streamlit
        
        Args:
            data: DataFrame with time series data
            comparison_results: Results from compare_models
        """
        st.subheader("Model Comparison Dashboard")
        
        # Performance summary
        summary = self.create_performance_summary(comparison_results)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Models", summary['total_models'])
        with col2:
            st.metric("Trained Models", summary['trained_models'])
        with col3:
            st.metric("Average Confidence", f"{summary['average_confidence']:.3f}")
        with col4:
            consensus_signal, consensus_confidence = self.get_consensus_signal(comparison_results)
            st.metric("Consensus Signal", f"{consensus_signal.upper()}")
        
        # Signal distribution
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Signal Distribution")
            signal_df = pd.DataFrame(list(summary['active_signals'].items()), 
                                   columns=['Signal', 'Count'])
            st.bar_chart(signal_df.set_index('Signal'))
        
        with col2:
            st.subheader("Consensus Analysis")
            st.write(f"**Consensus Signal:** {consensus_signal.upper()}")
            st.write(f"**Consensus Confidence:** {consensus_confidence:.3f}")
            
            # Show individual model signals
            for name, result in comparison_results.items():
                if result['status'] == 'success':
                    st.write(f"• {name}: {result['signal'].upper()} ({result['confidence']:.3f})")
        
        # Comparison chart
        fig = self.create_comparison_chart(data, comparison_results)
        st.plotly_chart(fig, use_container_width=True)
        
        # Metrics table
        st.subheader("Model Metrics Comparison")
        metrics_df = self.create_metrics_table(comparison_results)
        st.dataframe(metrics_df, use_container_width=True)
        
        # Model status details
        st.subheader("Model Status Details")
        for name, status in summary['model_status'].items():
            if status == 'success':
                st.success(f"✅ {name}: Ready")
            elif status == 'not_trained':
                st.warning(f"⚠️ {name}: Not trained")
            else:
                st.error(f"❌ {name}: {status}")
