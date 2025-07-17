"""
Tool Ranking Framework
Evaluates and ranks trading tools based on timeliness, accuracy, transparency, accessibility, and market integration
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime
import json

class ToolRanking:
    """Framework for ranking and evaluating trading tools"""
    
    def __init__(self):
        self.ranking_criteria = {
            'timeliness': {
                'weight': 0.25,
                'description': 'Speed of data delivery and signal generation',
                'subcriteria': ['data_latency', 'signal_speed', 'real_time_capability']
            },
            'accuracy': {
                'weight': 0.30,
                'description': 'Prediction accuracy and signal quality',
                'subcriteria': ['prediction_accuracy', 'signal_quality', 'backtest_performance']
            },
            'transparency': {
                'weight': 0.20,
                'description': 'Explainability and decision transparency',
                'subcriteria': ['explainability', 'methodology_clarity', 'decision_logging']
            },
            'accessibility': {
                'weight': 0.15,
                'description': 'Ease of use and user experience',
                'subcriteria': ['user_interface', 'learning_curve', 'documentation']
            },
            'market_integration': {
                'weight': 0.10,
                'description': 'Cross-market support and data coverage',
                'subcriteria': ['market_coverage', 'data_sources', 'cross_asset_analysis']
            }
        }
        
        self.trading_tools = self._initialize_tools()
    
    def _initialize_tools(self):
        """Initialize trading tools for evaluation"""
        return {
            'AI_Trading_Agent': {
                'name': 'AI Trading Agent (This Tool)',
                'type': 'AI-Powered',
                'description': 'Comprehensive AI trading agent with ML models and explainability',
                'features': ['LSTM', 'Random Forest', 'XGBoost', 'SHAP Explainability', 'Cross-Market Analysis'],
                'scores': {
                    'timeliness': {
                        'data_latency': 8,
                        'signal_speed': 9,
                        'real_time_capability': 7
                    },
                    'accuracy': {
                        'prediction_accuracy': 8,
                        'signal_quality': 8,
                        'backtest_performance': 9
                    },
                    'transparency': {
                        'explainability': 10,
                        'methodology_clarity': 9,
                        'decision_logging': 10
                    },
                    'accessibility': {
                        'user_interface': 9,
                        'learning_curve': 7,
                        'documentation': 8
                    },
                    'market_integration': {
                        'market_coverage': 9,
                        'data_sources': 8,
                        'cross_asset_analysis': 10
                    }
                }
            },
            'TradingView': {
                'name': 'TradingView',
                'type': 'Technical Analysis Platform',
                'description': 'Popular charting and analysis platform',
                'features': ['Advanced Charts', 'Technical Indicators', 'Social Trading', 'Alerts'],
                'scores': {
                    'timeliness': {
                        'data_latency': 9,
                        'signal_speed': 8,
                        'real_time_capability': 9
                    },
                    'accuracy': {
                        'prediction_accuracy': 6,
                        'signal_quality': 7,
                        'backtest_performance': 6
                    },
                    'transparency': {
                        'explainability': 5,
                        'methodology_clarity': 7,
                        'decision_logging': 4
                    },
                    'accessibility': {
                        'user_interface': 9,
                        'learning_curve': 8,
                        'documentation': 8
                    },
                    'market_integration': {
                        'market_coverage': 9,
                        'data_sources': 9,
                        'cross_asset_analysis': 7
                    }
                }
            },
            'MetaTrader': {
                'name': 'MetaTrader 5',
                'type': 'Trading Platform',
                'description': 'Professional trading platform with algorithmic trading',
                'features': ['Algorithmic Trading', 'Technical Analysis', 'Market Depth', 'Strategy Tester'],
                'scores': {
                    'timeliness': {
                        'data_latency': 8,
                        'signal_speed': 9,
                        'real_time_capability': 8
                    },
                    'accuracy': {
                        'prediction_accuracy': 7,
                        'signal_quality': 7,
                        'backtest_performance': 8
                    },
                    'transparency': {
                        'explainability': 4,
                        'methodology_clarity': 6,
                        'decision_logging': 5
                    },
                    'accessibility': {
                        'user_interface': 6,
                        'learning_curve': 5,
                        'documentation': 7
                    },
                    'market_integration': {
                        'market_coverage': 8,
                        'data_sources': 7,
                        'cross_asset_analysis': 6
                    }
                }
            },
            'Coinigy': {
                'name': 'Coinigy',
                'type': 'Crypto Trading Platform',
                'description': 'Cryptocurrency trading and portfolio management',
                'features': ['Multi-Exchange', 'Portfolio Tracking', 'Technical Analysis', 'Alerts'],
                'scores': {
                    'timeliness': {
                        'data_latency': 7,
                        'signal_speed': 7,
                        'real_time_capability': 8
                    },
                    'accuracy': {
                        'prediction_accuracy': 6,
                        'signal_quality': 6,
                        'backtest_performance': 5
                    },
                    'transparency': {
                        'explainability': 3,
                        'methodology_clarity': 5,
                        'decision_logging': 3
                    },
                    'accessibility': {
                        'user_interface': 7,
                        'learning_curve': 7,
                        'documentation': 6
                    },
                    'market_integration': {
                        'market_coverage': 6,
                        'data_sources': 7,
                        'cross_asset_analysis': 5
                    }
                }
            },
            'QuantConnect': {
                'name': 'QuantConnect',
                'type': 'Algorithmic Trading Platform',
                'description': 'Cloud-based algorithmic trading platform',
                'features': ['Backtesting', 'Algorithm Development', 'Data Library', 'Live Trading'],
                'scores': {
                    'timeliness': {
                        'data_latency': 7,
                        'signal_speed': 8,
                        'real_time_capability': 7
                    },
                    'accuracy': {
                        'prediction_accuracy': 8,
                        'signal_quality': 8,
                        'backtest_performance': 9
                    },
                    'transparency': {
                        'explainability': 6,
                        'methodology_clarity': 8,
                        'decision_logging': 7
                    },
                    'accessibility': {
                        'user_interface': 6,
                        'learning_curve': 4,
                        'documentation': 9
                    },
                    'market_integration': {
                        'market_coverage': 8,
                        'data_sources': 9,
                        'cross_asset_analysis': 7
                    }
                }
            },
            'Alpaca': {
                'name': 'Alpaca',
                'type': 'Commission-Free Trading API',
                'description': 'API-first stock trading platform',
                'features': ['Commission-Free Trading', 'API Access', 'Paper Trading', 'Real-time Data'],
                'scores': {
                    'timeliness': {
                        'data_latency': 8,
                        'signal_speed': 9,
                        'real_time_capability': 8
                    },
                    'accuracy': {
                        'prediction_accuracy': 7,
                        'signal_quality': 7,
                        'backtest_performance': 7
                    },
                    'transparency': {
                        'explainability': 5,
                        'methodology_clarity': 7,
                        'decision_logging': 6
                    },
                    'accessibility': {
                        'user_interface': 7,
                        'learning_curve': 6,
                        'documentation': 8
                    },
                    'market_integration': {
                        'market_coverage': 6,
                        'data_sources': 7,
                        'cross_asset_analysis': 4
                    }
                }
            }
        }
    
    def calculate_weighted_score(self, tool_scores):
        """Calculate weighted score for a tool"""
        try:
            total_score = 0
            
            for criterion, criterion_data in self.ranking_criteria.items():
                criterion_score = 0
                subcriteria_count = len(criterion_data['subcriteria'])
                
                for subcriterion in criterion_data['subcriteria']:
                    if subcriterion in tool_scores[criterion]:
                        criterion_score += tool_scores[criterion][subcriterion]
                
                # Average the subcriteria scores
                if subcriteria_count > 0:
                    criterion_score /= subcriteria_count
                
                # Apply weight
                weighted_score = criterion_score * criterion_data['weight']
                total_score += weighted_score
            
            return total_score
            
        except Exception as e:
            print(f"Error calculating weighted score: {str(e)}")
            return 0
    
    def rank_tools(self):
        """Rank all tools based on weighted scores"""
        try:
            rankings = []
            
            for tool_id, tool_data in self.trading_tools.items():
                weighted_score = self.calculate_weighted_score(tool_data['scores'])
                
                # Calculate individual criterion scores
                criterion_scores = {}
                for criterion, criterion_data in self.ranking_criteria.items():
                    criterion_score = 0
                    subcriteria_count = len(criterion_data['subcriteria'])
                    
                    for subcriterion in criterion_data['subcriteria']:
                        if subcriterion in tool_data['scores'][criterion]:
                            criterion_score += tool_data['scores'][criterion][subcriterion]
                    
                    if subcriteria_count > 0:
                        criterion_scores[criterion] = criterion_score / subcriteria_count
                    else:
                        criterion_scores[criterion] = 0
                
                rankings.append({
                    'tool_id': tool_id,
                    'name': tool_data['name'],
                    'type': tool_data['type'],
                    'description': tool_data['description'],
                    'features': tool_data['features'],
                    'weighted_score': weighted_score,
                    'criterion_scores': criterion_scores
                })
            
            # Sort by weighted score (descending)
            rankings.sort(key=lambda x: x['weighted_score'], reverse=True)
            
            return rankings
            
        except Exception as e:
            st.error(f"Error ranking tools: {str(e)}")
            return []
    
    def create_ranking_chart(self, rankings):
        """Create ranking visualization chart"""
        try:
            if not rankings:
                return None
            
            # Prepare data for plotting
            names = [r['name'] for r in rankings]
            scores = [r['weighted_score'] for r in rankings]
            types = [r['type'] for r in rankings]
            
            # Create bar chart
            fig = go.Figure()
            
            # Color mapping for different types
            color_map = {
                'AI-Powered': '#1f77b4',
                'Technical Analysis Platform': '#ff7f0e',
                'Trading Platform': '#2ca02c',
                'Crypto Trading Platform': '#d62728',
                'Algorithmic Trading Platform': '#9467bd',
                'Commission-Free Trading API': '#8c564b'
            }
            
            colors = [color_map.get(t, '#7f7f7f') for t in types]
            
            fig.add_trace(go.Bar(
                x=names,
                y=scores,
                marker_color=colors,
                text=[f'{score:.2f}' for score in scores],
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>Score: %{y:.2f}<extra></extra>'
            ))
            
            fig.update_layout(
                title='Trading Tool Rankings',
                xaxis_title='Trading Tools',
                yaxis_title='Weighted Score',
                xaxis_tickangle=-45,
                height=500,
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating ranking chart: {str(e)}")
            return None
    
    def create_radar_chart(self, rankings):
        """Create radar chart for top tools"""
        try:
            if not rankings:
                return None
            
            # Take top 4 tools
            top_tools = rankings[:4]
            
            # Prepare data
            criteria = list(self.ranking_criteria.keys())
            criteria_labels = [c.replace('_', ' ').title() for c in criteria]
            
            fig = go.Figure()
            
            for tool in top_tools:
                values = [tool['criterion_scores'][c] for c in criteria]
                
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=criteria_labels,
                    fill='toself',
                    name=tool['name']
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 10]
                    )
                ),
                title='Top Trading Tools - Criteria Comparison',
                showlegend=True,
                height=600
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating radar chart: {str(e)}")
            return None
    
    def create_detailed_comparison(self, rankings):
        """Create detailed comparison table"""
        try:
            if not rankings:
                return pd.DataFrame()
            
            comparison_data = []
            
            for tool in rankings:
                row = {
                    'Rank': rankings.index(tool) + 1,
                    'Tool': tool['name'],
                    'Type': tool['type'],
                    'Overall Score': f"{tool['weighted_score']:.2f}",
                    'Timeliness': f"{tool['criterion_scores']['timeliness']:.1f}",
                    'Accuracy': f"{tool['criterion_scores']['accuracy']:.1f}",
                    'Transparency': f"{tool['criterion_scores']['transparency']:.1f}",
                    'Accessibility': f"{tool['criterion_scores']['accessibility']:.1f}",
                    'Market Integration': f"{tool['criterion_scores']['market_integration']:.1f}"
                }
                comparison_data.append(row)
            
            return pd.DataFrame(comparison_data)
            
        except Exception as e:
            st.error(f"Error creating detailed comparison: {str(e)}")
            return pd.DataFrame()
    
    def get_tool_analysis(self, tool_id):
        """Get detailed analysis for a specific tool"""
        try:
            if tool_id not in self.trading_tools:
                return None
            
            tool = self.trading_tools[tool_id]
            
            # Calculate scores
            weighted_score = self.calculate_weighted_score(tool['scores'])
            
            # Calculate strengths and weaknesses
            strengths = []
            weaknesses = []
            
            for criterion, criterion_data in self.ranking_criteria.items():
                criterion_score = 0
                subcriteria_count = len(criterion_data['subcriteria'])
                
                for subcriterion in criterion_data['subcriteria']:
                    if subcriterion in tool['scores'][criterion]:
                        score = tool['scores'][criterion][subcriterion]
                        criterion_score += score
                        
                        if score >= 8:
                            strengths.append(f"{subcriterion.replace('_', ' ').title()} ({score}/10)")
                        elif score <= 4:
                            weaknesses.append(f"{subcriterion.replace('_', ' ').title()} ({score}/10)")
                
                avg_criterion_score = criterion_score / subcriteria_count if subcriteria_count > 0 else 0
                
                if avg_criterion_score >= 8:
                    strengths.append(f"{criterion.replace('_', ' ').title()} (Overall: {avg_criterion_score:.1f}/10)")
                elif avg_criterion_score <= 4:
                    weaknesses.append(f"{criterion.replace('_', ' ').title()} (Overall: {avg_criterion_score:.1f}/10)")
            
            analysis = {
                'tool_data': tool,
                'weighted_score': weighted_score,
                'strengths': strengths,
                'weaknesses': weaknesses,
                'recommendation': self._generate_recommendation(weighted_score, strengths, weaknesses)
            }
            
            return analysis
            
        except Exception as e:
            st.error(f"Error getting tool analysis: {str(e)}")
            return None
    
    def _generate_recommendation(self, score, strengths, weaknesses):
        """Generate recommendation based on analysis"""
        try:
            if score >= 8:
                return "Highly Recommended - Excellent overall performance with strong capabilities across all criteria."
            elif score >= 7:
                return "Recommended - Good performance with notable strengths, minor areas for improvement."
            elif score >= 6:
                return "Conditionally Recommended - Decent performance but has some limitations to consider."
            elif score >= 5:
                return "Limited Recommendation - Average performance with significant room for improvement."
            else:
                return "Not Recommended - Below average performance with major limitations."
                
        except Exception:
            return "Unable to generate recommendation."
    
    def get_comprehensive_ranking(self):
        """Get comprehensive ranking analysis"""
        try:
            rankings = self.rank_tools()
            
            # Calculate market insights
            market_insights = {
                'total_tools': len(rankings),
                'average_score': np.mean([r['weighted_score'] for r in rankings]),
                'best_tool': rankings[0] if rankings else None,
                'criteria_averages': {}
            }
            
            # Calculate average scores by criteria
            for criterion in self.ranking_criteria.keys():
                scores = [r['criterion_scores'][criterion] for r in rankings]
                market_insights['criteria_averages'][criterion] = np.mean(scores)
            
            return {
                'rankings': rankings,
                'market_insights': market_insights,
                'criteria_info': self.ranking_criteria
            }
            
        except Exception as e:
            st.error(f"Error getting comprehensive ranking: {str(e)}")
            return {}
    
    def display_ranking_dashboard(self, ranking_results):
        """Display comprehensive ranking dashboard"""
        try:
            if not ranking_results:
                st.error("No ranking results available")
                return
            
            rankings = ranking_results['rankings']
            market_insights = ranking_results['market_insights']
            
            # Market overview
            st.subheader("📊 Market Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Tools Analyzed", market_insights['total_tools'])
            
            with col2:
                st.metric("Average Score", f"{market_insights['average_score']:.2f}")
            
            with col3:
                if market_insights['best_tool']:
                    st.metric("Top Tool", market_insights['best_tool']['name'])
            
            with col4:
                if market_insights['best_tool']:
                    st.metric("Best Score", f"{market_insights['best_tool']['weighted_score']:.2f}")
            
            # Ranking chart
            st.subheader("🏆 Tool Rankings")
            ranking_chart = self.create_ranking_chart(rankings)
            if ranking_chart:
                st.plotly_chart(ranking_chart, use_container_width=True)
            
            # Radar chart
            st.subheader("🎯 Criteria Comparison")
            radar_chart = self.create_radar_chart(rankings)
            if radar_chart:
                st.plotly_chart(radar_chart, use_container_width=True)
            
            # Detailed comparison
            st.subheader("📋 Detailed Comparison")
            comparison_df = self.create_detailed_comparison(rankings)
            if not comparison_df.empty:
                st.dataframe(comparison_df, use_container_width=True)
            
            # Tool analysis
            st.subheader("🔍 Tool Analysis")
            selected_tool = st.selectbox(
                "Select Tool for Detailed Analysis",
                options=list(self.trading_tools.keys()),
                format_func=lambda x: self.trading_tools[x]['name']
            )
            
            if selected_tool:
                analysis = self.get_tool_analysis(selected_tool)
                if analysis:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Overall Score:** {analysis['weighted_score']:.2f}/10")
                        st.write(f"**Type:** {analysis['tool_data']['type']}")
                        st.write(f"**Description:** {analysis['tool_data']['description']}")
                        
                        st.write("**Key Features:**")
                        for feature in analysis['tool_data']['features']:
                            st.write(f"• {feature}")
                    
                    with col2:
                        st.write("**Strengths:**")
                        for strength in analysis['strengths']:
                            st.success(f"✅ {strength}")
                        
                        st.write("**Areas for Improvement:**")
                        for weakness in analysis['weaknesses']:
                            st.warning(f"⚠️ {weakness}")
                        
                        st.write("**Recommendation:**")
                        st.info(analysis['recommendation'])
            
            # Criteria explanation
            st.subheader("📖 Ranking Criteria")
            for criterion, criterion_data in ranking_results['criteria_info'].items():
                with st.expander(f"{criterion.replace('_', ' ').title()} (Weight: {criterion_data['weight']:.0%})"):
                    st.write(f"**Description:** {criterion_data['description']}")
                    st.write("**Subcriteria:**")
                    for subcriterion in criterion_data['subcriteria']:
                        st.write(f"• {subcriterion.replace('_', ' ').title()}")
                    st.write(f"**Average Score:** {market_insights['criteria_averages'][criterion]:.1f}/10")
            
        except Exception as e:
            st.error(f"Error displaying ranking dashboard: {str(e)}")
