"""
Explainability Module
Provides SHAP-based model interpretability and rule-based decision logging
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime
import json

# Try to import SHAP, use fallback if not available
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    st.warning("SHAP not available. Some explainability features will be limited.")

class ExplainabilityEngine:
    """Engine for model explainability and decision transparency"""
    
    def __init__(self):
        self.explainers = {}
        self.decision_log = []
        
    def create_shap_explainer(self, model, X_train, model_type='tree'):
        """
        Create SHAP explainer for a model
        
        Args:
            model: Trained model
            X_train: Training data
            model_type: Type of model ('tree', 'linear', 'deep')
            
        Returns:
            explainer: SHAP explainer object or None if SHAP not available
        """
        if not SHAP_AVAILABLE:
            st.warning("SHAP not installed. Install it to enable model explainability features.")
            return None
            
        try:
            if model_type == 'tree':
                explainer = shap.TreeExplainer(model)
            elif model_type == 'linear':
                explainer = shap.LinearExplainer(model, X_train)
            elif model_type == 'deep':
                explainer = shap.DeepExplainer(model, X_train)
            else:
                # Use KernelExplainer as fallback
                explainer = shap.KernelExplainer(model.predict, X_train)
            
            return explainer
            
        except Exception as e:
            raise Exception(f"Error creating SHAP explainer: {str(e)}")
    
    def explain_prediction(self, model, explainer, X_sample, feature_names=None):
        """
        Explain a single prediction using SHAP
        
        Args:
            model: Trained model
            explainer: SHAP explainer
            X_sample: Sample to explain
            feature_names: Names of features
            
        Returns:
            explanation: Dictionary with explanation data
        """
        try:
            # Get SHAP values
            shap_values = explainer.shap_values(X_sample)
            
            # Handle different output formats
            if isinstance(shap_values, list):
                # Multi-class classification
                shap_values = shap_values[0]  # Take first class for now
            
            # Get expected value
            expected_value = explainer.expected_value
            if isinstance(expected_value, np.ndarray):
                expected_value = expected_value[0]
            
            # Create explanation
            explanation = {
                'shap_values': shap_values.tolist() if isinstance(shap_values, np.ndarray) else shap_values,
                'expected_value': float(expected_value),
                'feature_names': feature_names or [f'feature_{i}' for i in range(len(X_sample))],
                'feature_values': X_sample.tolist() if isinstance(X_sample, np.ndarray) else X_sample,
                'prediction': model.predict(X_sample.reshape(1, -1))[0] if hasattr(model, 'predict') else None
            }
            
            return explanation
            
        except Exception as e:
            raise Exception(f"Error explaining prediction: {str(e)}")
    
    def create_feature_importance_plot(self, shap_values, feature_names, title="Feature Importance"):
        """
        Create feature importance plot using SHAP values
        
        Args:
            shap_values: SHAP values
            feature_names: Names of features
            title: Plot title
            
        Returns:
            fig: Plotly figure
        """
        try:
            # Calculate mean absolute SHAP values
            mean_shap = np.mean(np.abs(shap_values), axis=0)
            
            # Sort by importance
            sorted_idx = np.argsort(mean_shap)[::-1]
            sorted_features = [feature_names[i] for i in sorted_idx]
            sorted_importance = mean_shap[sorted_idx]
            
            # Create plot
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=sorted_importance[:20],  # Top 20 features
                y=sorted_features[:20],
                orientation='h',
                marker_color='steelblue'
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title="Mean |SHAP value|",
                yaxis_title="Features",
                height=600,
                yaxis={'categoryorder': 'total ascending'}
            )
            
            return fig
            
        except Exception as e:
            raise Exception(f"Error creating feature importance plot: {str(e)}")
    
    def create_waterfall_plot(self, explanation, max_features=10):
        """
        Create waterfall plot for a single prediction
        
        Args:
            explanation: Explanation dictionary
            max_features: Maximum number of features to show
            
        Returns:
            fig: Plotly figure
        """
        try:
            shap_values = np.array(explanation['shap_values'])
            feature_names = explanation['feature_names']
            feature_values = explanation['feature_values']
            expected_value = explanation['expected_value']
            
            # Sort by absolute SHAP value
            sorted_idx = np.argsort(np.abs(shap_values))[::-1]
            top_idx = sorted_idx[:max_features]
            
            # Get top features
            top_shap = shap_values[top_idx]
            top_names = [feature_names[i] for i in top_idx]
            top_values = [feature_values[i] for i in top_idx]
            
            # Create waterfall data
            cumulative = [expected_value]
            for val in top_shap:
                cumulative.append(cumulative[-1] + val)
            
            # Create plot
            fig = go.Figure()
            
            # Add bars
            x_pos = list(range(len(top_names) + 2))
            
            # Expected value bar
            fig.add_trace(go.Bar(
                x=[0],
                y=[expected_value],
                name="Expected Value",
                marker_color='gray',
                text=[f"Expected: {expected_value:.3f}"],
                textposition='auto'
            ))
            
            # Feature contribution bars
            for i, (name, shap_val, feat_val) in enumerate(zip(top_names, top_shap, top_values)):
                color = 'green' if shap_val > 0 else 'red'
                fig.add_trace(go.Bar(
                    x=[i + 1],
                    y=[shap_val],
                    name=f"{name}",
                    marker_color=color,
                    text=[f"{name}={feat_val:.3f}<br>SHAP={shap_val:.3f}"],
                    textposition='auto'
                ))
            
            # Final prediction bar
            final_pred = cumulative[-1]
            fig.add_trace(go.Bar(
                x=[len(top_names) + 1],
                y=[final_pred],
                name="Final Prediction",
                marker_color='blue',
                text=[f"Prediction: {final_pred:.3f}"],
                textposition='auto'
            ))
            
            fig.update_layout(
                title="Prediction Explanation (Waterfall)",
                xaxis_title="Features",
                yaxis_title="Contribution",
                showlegend=False,
                height=500
            )
            
            # Update x-axis labels
            fig.update_xaxes(
                tickvals=x_pos,
                ticktext=["Expected"] + top_names + ["Prediction"]
            )
            
            return fig
            
        except Exception as e:
            raise Exception(f"Error creating waterfall plot: {str(e)}")
    
    def log_decision(self, model_name, symbol, prediction, confidence, explanation_data, features_used):
        """
        Log a trading decision with explanation
        
        Args:
            model_name: Name of the model
            symbol: Trading symbol
            prediction: Model prediction
            confidence: Confidence score
            explanation_data: SHAP explanation data
            features_used: List of features used
        """
        try:
            decision_entry = {
                'timestamp': datetime.now().isoformat(),
                'model_name': model_name,
                'symbol': symbol,
                'prediction': prediction,
                'confidence': confidence,
                'explanation': explanation_data,
                'features_used': features_used,
                'decision_id': len(self.decision_log) + 1
            }
            
            self.decision_log.append(decision_entry)
            
        except Exception as e:
            print(f"Error logging decision: {str(e)}")
    
    def get_decision_history(self, model_name=None, symbol=None, limit=100):
        """
        Get decision history
        
        Args:
            model_name: Filter by model name
            symbol: Filter by symbol
            limit: Maximum number of decisions to return
            
        Returns:
            decisions: List of decision entries
        """
        try:
            decisions = self.decision_log.copy()
            
            # Apply filters
            if model_name:
                decisions = [d for d in decisions if d['model_name'] == model_name]
            
            if symbol:
                decisions = [d for d in decisions if d['symbol'] == symbol]
            
            # Sort by timestamp (newest first)
            decisions.sort(key=lambda x: x['timestamp'], reverse=True)
            
            return decisions[:limit]
            
        except Exception as e:
            print(f"Error getting decision history: {str(e)}")
            return []
    
    def create_decision_summary(self, decisions):
        """
        Create summary of decisions
        
        Args:
            decisions: List of decision entries
            
        Returns:
            summary: Decision summary dictionary
        """
        try:
            if not decisions:
                return {'total_decisions': 0, 'summary': 'No decisions found'}
            
            summary = {
                'total_decisions': len(decisions),
                'models_used': list(set([d['model_name'] for d in decisions])),
                'symbols_traded': list(set([d['symbol'] for d in decisions])),
                'predictions': {
                    'buy': len([d for d in decisions if d['prediction'] == 'buy']),
                    'sell': len([d for d in decisions if d['prediction'] == 'sell']),
                    'hold': len([d for d in decisions if d['prediction'] == 'hold'])
                },
                'average_confidence': np.mean([d['confidence'] for d in decisions]),
                'time_range': {
                    'start': min([d['timestamp'] for d in decisions]),
                    'end': max([d['timestamp'] for d in decisions])
                }
            }
            
            return summary
            
        except Exception as e:
            return {'error': f"Error creating decision summary: {str(e)}"}
    
    def explain_ensemble_decision(self, model_predictions, model_explanations):
        """
        Explain ensemble model decision
        
        Args:
            model_predictions: Dictionary of model predictions
            model_explanations: Dictionary of model explanations
            
        Returns:
            ensemble_explanation: Combined explanation
        """
        try:
            ensemble_explanation = {
                'individual_predictions': model_predictions,
                'individual_explanations': model_explanations,
                'consensus_analysis': {},
                'feature_importance_consensus': {},
                'decision_rationale': []
            }
            
            # Analyze consensus
            predictions = list(model_predictions.values())
            unique_predictions = list(set(predictions))
            
            for pred in unique_predictions:
                count = predictions.count(pred)
                ensemble_explanation['consensus_analysis'][pred] = {
                    'count': count,
                    'percentage': count / len(predictions) * 100
                }
            
            # Combine feature importance
            all_features = set()
            for exp in model_explanations.values():
                if 'feature_names' in exp:
                    all_features.update(exp['feature_names'])
            
            feature_importance_sum = {}
            for feature in all_features:
                importance_sum = 0
                count = 0
                
                for exp in model_explanations.values():
                    if 'feature_names' in exp and feature in exp['feature_names']:
                        idx = exp['feature_names'].index(feature)
                        if 'shap_values' in exp and idx < len(exp['shap_values']):
                            importance_sum += abs(exp['shap_values'][idx])
                            count += 1
                
                if count > 0:
                    feature_importance_sum[feature] = importance_sum / count
            
            ensemble_explanation['feature_importance_consensus'] = feature_importance_sum
            
            # Generate decision rationale
            for model_name, prediction in model_predictions.items():
                rationale = f"{model_name} predicts {prediction}"
                if model_name in model_explanations:
                    exp = model_explanations[model_name]
                    if 'confidence' in exp:
                        rationale += f" with confidence {exp['confidence']:.3f}"
                
                ensemble_explanation['decision_rationale'].append(rationale)
            
            return ensemble_explanation
            
        except Exception as e:
            return {'error': f"Error explaining ensemble decision: {str(e)}"}
    
    def create_explainability_dashboard(self, model_name, explanation, symbol):
        """
        Create explainability dashboard in Streamlit
        
        Args:
            model_name: Name of the model
            explanation: Explanation data
            symbol: Trading symbol
        """
        try:
            st.subheader(f"Model Explainability - {model_name}")
            
            # Basic prediction info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Prediction", explanation.get('prediction', 'N/A'))
            with col2:
                st.metric("Expected Value", f"{explanation.get('expected_value', 0):.4f}")
            with col3:
                st.metric("Symbol", symbol)
            
            # Feature importance
            if 'shap_values' in explanation and 'feature_names' in explanation:
                shap_values = np.array(explanation['shap_values'])
                feature_names = explanation['feature_names']
                
                # Create feature importance plot
                fig_importance = self.create_feature_importance_plot(
                    shap_values.reshape(1, -1), 
                    feature_names, 
                    f"Feature Importance - {model_name}"
                )
                st.plotly_chart(fig_importance, use_container_width=True)
                
                # Create waterfall plot
                fig_waterfall = self.create_waterfall_plot(explanation)
                st.plotly_chart(fig_waterfall, use_container_width=True)
                
                # Feature values table
                st.subheader("Feature Values and Contributions")
                if 'feature_values' in explanation:
                    feature_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Value': explanation['feature_values'],
                        'SHAP Value': shap_values,
                        'Contribution': ['Positive' if x > 0 else 'Negative' for x in shap_values]
                    })
                    
                    # Sort by absolute SHAP value
                    feature_df['Abs_SHAP'] = np.abs(feature_df['SHAP Value'])
                    feature_df = feature_df.sort_values('Abs_SHAP', ascending=False)
                    feature_df = feature_df.drop('Abs_SHAP', axis=1)
                    
                    st.dataframe(feature_df, use_container_width=True)
            
            # Decision rationale
            st.subheader("Decision Rationale")
            if 'decision_rationale' in explanation:
                for rationale in explanation['decision_rationale']:
                    st.write(f"• {rationale}")
            else:
                st.write("Detailed rationale not available for this prediction.")
            
        except Exception as e:
            st.error(f"Error creating explainability dashboard: {str(e)}")
    
    def save_decision_log(self, filepath):
        """Save decision log to file"""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.decision_log, f, indent=2, default=str)
            return True
        except Exception as e:
            print(f"Error saving decision log: {str(e)}")
            return False
    
    def load_decision_log(self, filepath):
        """Load decision log from file"""
        try:
            with open(filepath, 'r') as f:
                self.decision_log = json.load(f)
            return True
        except Exception as e:
            print(f"Error loading decision log: {str(e)}")
            return False
