"""
Generate ERD (Entity-Relationship Diagram) for IntelliTradeAI
Shows data entities and relationships in the trading system
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

# Ensure diagrams directory exists
os.makedirs('diagrams', exist_ok=True)

def create_erd():
    """Create comprehensive ERD diagram"""
    
    fig, ax = plt.subplots(figsize=(20, 14))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # Define entities with their attributes
    entities = {
        'Cryptocurrency': {
            'pos': (2, 11),
            'attributes': [
                'PK: symbol (VARCHAR)',
                'name (VARCHAR)',
                'cmc_id (INT)',
                'rank (INT)',
                'yahoo_symbol (VARCHAR)',
                'market_cap (DECIMAL)',
                'last_updated (TIMESTAMP)'
            ],
            'color': '#E3F2FD'
        },
        'OHLCV_Data': {
            'pos': (2, 6.5),
            'attributes': [
                'PK: id (INT)',
                'FK: symbol (VARCHAR)',
                'date (DATE)',
                'open (DECIMAL)',
                'high (DECIMAL)',
                'low (DECIMAL)',
                'close (DECIMAL)',
                'volume (BIGINT)',
                'source (VARCHAR)'
            ],
            'color': '#FFF3E0'
        },
        'Technical_Indicators': {
            'pos': (7, 6.5),
            'attributes': [
                'PK: id (INT)',
                'FK: ohlcv_id (INT)',
                'rsi (DECIMAL)',
                'macd (DECIMAL)',
                'macd_signal (DECIMAL)',
                'bb_upper (DECIMAL)',
                'bb_middle (DECIMAL)',
                'bb_lower (DECIMAL)',
                'ema_12 (DECIMAL)',
                'ema_26 (DECIMAL)',
                'volume_ratio (DECIMAL)'
            ],
            'color': '#E8F5E9'
        },
        'ML_Models': {
            'pos': (12, 11),
            'attributes': [
                'PK: model_id (INT)',
                'FK: symbol (VARCHAR)',
                'model_type (VARCHAR)',
                'version (VARCHAR)',
                'accuracy (DECIMAL)',
                'precision (DECIMAL)',
                'recall (DECIMAL)',
                'f1_score (DECIMAL)',
                'roc_auc (DECIMAL)',
                'trained_date (TIMESTAMP)',
                'model_path (VARCHAR)'
            ],
            'color': '#F3E5F5'
        },
        'Training_Sessions': {
            'pos': (12, 6.5),
            'attributes': [
                'PK: session_id (INT)',
                'FK: model_id (INT)',
                'train_start (TIMESTAMP)',
                'train_end (TIMESTAMP)',
                'train_samples (INT)',
                'test_samples (INT)',
                'hyperparameters (JSON)',
                'status (VARCHAR)'
            ],
            'color': '#FCE4EC'
        },
        'Predictions': {
            'pos': (17, 11),
            'attributes': [
                'PK: prediction_id (INT)',
                'FK: model_id (INT)',
                'FK: symbol (VARCHAR)',
                'prediction_date (TIMESTAMP)',
                'signal (VARCHAR)',
                'confidence (DECIMAL)',
                'predicted_direction (INT)',
                'actual_direction (INT)',
                'target_price (DECIMAL)'
            ],
            'color': '#FFF9C4'
        },
        'Portfolio_Performance': {
            'pos': (17, 6.5),
            'attributes': [
                'PK: performance_id (INT)',
                'FK: symbol (VARCHAR)',
                'period_start (DATE)',
                'period_end (DATE)',
                'total_return_pct (DECIMAL)',
                'volatility_pct (DECIMAL)',
                'sharpe_ratio (DECIMAL)',
                'max_drawdown (DECIMAL)',
                'win_rate (DECIMAL)'
            ],
            'color': '#E0F7FA'
        },
        'API_Cache': {
            'pos': (2, 2),
            'attributes': [
                'PK: cache_id (INT)',
                'cache_key (VARCHAR)',
                'data (JSON)',
                'created_at (TIMESTAMP)',
                'expires_at (TIMESTAMP)',
                'source (VARCHAR)'
            ],
            'color': '#EFEBE9'
        },
        'Feature_Engineering': {
            'pos': (7, 2),
            'attributes': [
                'PK: feature_id (INT)',
                'FK: ohlcv_id (INT)',
                'momentum_features (JSON)',
                'volatility_features (JSON)',
                'pattern_features (JSON)',
                'lagged_features (JSON)',
                'target_variable (INT)'
            ],
            'color': '#E1F5FE'
        },
        'Backtest_Results': {
            'pos': (12, 2),
            'attributes': [
                'PK: backtest_id (INT)',
                'FK: model_id (INT)',
                'start_date (DATE)',
                'end_date (DATE)',
                'initial_capital (DECIMAL)',
                'final_capital (DECIMAL)',
                'total_trades (INT)',
                'winning_trades (INT)',
                'losing_trades (INT)',
                'profit_factor (DECIMAL)'
            ],
            'color': '#F1F8E9'
        }
    }
    
    # Draw entities
    for entity_name, entity_info in entities.items():
        x, y = entity_info['pos']
        attrs = entity_info['attributes']
        color = entity_info['color']
        
        # Calculate box height based on number of attributes
        height = 0.35 + (len(attrs) * 0.18)
        
        # Draw entity box
        box = FancyBboxPatch(
            (x - 1.8, y - height/2), 3.6, height,
            boxstyle="round,pad=0.1",
            edgecolor='black',
            facecolor=color,
            linewidth=2
        )
        ax.add_patch(box)
        
        # Draw entity name (header)
        header_box = FancyBboxPatch(
            (x - 1.8, y + height/2 - 0.35), 3.6, 0.35,
            boxstyle="round,pad=0.05",
            edgecolor='black',
            facecolor='#37474F',
            linewidth=2
        )
        ax.add_patch(header_box)
        
        ax.text(x, y + height/2 - 0.175, entity_name,
               ha='center', va='center', fontsize=11,
               weight='bold', color='white')
        
        # Draw attributes
        y_offset = y + height/2 - 0.35 - 0.15
        for attr in attrs:
            if attr.startswith('PK:'):
                ax.text(x - 1.7, y_offset, attr, ha='left', va='center',
                       fontsize=8, weight='bold', color='#D32F2F',
                       family='monospace')
            elif attr.startswith('FK:'):
                ax.text(x - 1.7, y_offset, attr, ha='left', va='center',
                       fontsize=8, weight='bold', color='#1976D2',
                       family='monospace')
            else:
                ax.text(x - 1.7, y_offset, attr, ha='left', va='center',
                       fontsize=8, family='monospace')
            y_offset -= 0.18
    
    # Define relationships
    relationships = [
        # One-to-Many relationships
        {
            'from': 'Cryptocurrency',
            'to': 'OHLCV_Data',
            'label': '1:N\nHas historical\ndata',
            'from_pos': (2, 10.25),
            'to_pos': (2, 8.3),
            'color': '#1976D2'
        },
        {
            'from': 'OHLCV_Data',
            'to': 'Technical_Indicators',
            'label': '1:N\nGenerates\nindicators',
            'from_pos': (3.8, 6.5),
            'to_pos': (5.2, 6.5),
            'color': '#388E3C'
        },
        {
            'from': 'OHLCV_Data',
            'to': 'Feature_Engineering',
            'label': '1:N\nEngineers\nfeatures',
            'from_pos': (2, 5.0),
            'to_pos': (7, 3.8),
            'color': '#F57C00'
        },
        {
            'from': 'Cryptocurrency',
            'to': 'ML_Models',
            'label': '1:N\nHas trained\nmodels',
            'from_pos': (3.8, 11),
            'to_pos': (10.2, 11),
            'color': '#7B1FA2'
        },
        {
            'from': 'ML_Models',
            'to': 'Training_Sessions',
            'label': '1:N\nTrained in\nsessions',
            'from_pos': (12, 10.25),
            'to_pos': (12, 8.3),
            'color': '#C2185B'
        },
        {
            'from': 'ML_Models',
            'to': 'Predictions',
            'label': '1:N\nGenerates\npredictions',
            'from_pos': (13.8, 11),
            'to_pos': (15.2, 11),
            'color': '#FBC02D'
        },
        {
            'from': 'ML_Models',
            'to': 'Backtest_Results',
            'label': '1:N\nBacktested\nwith results',
            'from_pos': (12, 10.25),
            'to_pos': (12, 3.8),
            'color': '#689F38'
        },
        {
            'from': 'Cryptocurrency',
            'to': 'Portfolio_Performance',
            'label': '1:N\nHas performance\nmetrics',
            'from_pos': (3.8, 11),
            'to_pos': (15.2, 6.5),
            'color': '#00ACC1'
        },
        {
            'from': 'Cryptocurrency',
            'to': 'API_Cache',
            'label': '1:N\nCached\ndata',
            'from_pos': (2, 10.25),
            'to_pos': (2, 3.8),
            'color': '#5D4037'
        },
        {
            'from': 'Predictions',
            'to': 'Cryptocurrency',
            'label': 'N:1\nFor symbol',
            'from_pos': (17, 10.25),
            'to_pos': (3.8, 11),
            'color': '#455A64',
            'style': 'dashed'
        }
    ]
    
    # Draw relationships
    for rel in relationships:
        from_pos = rel['from_pos']
        to_pos = rel['to_pos']
        
        arrow = FancyArrowPatch(
            from_pos, to_pos,
            arrowstyle='-|>',
            mutation_scale=20,
            linewidth=2,
            color=rel['color'],
            linestyle=rel.get('style', 'solid'),
            alpha=0.7
        )
        ax.add_patch(arrow)
        
        # Add relationship label
        mid_x = (from_pos[0] + to_pos[0]) / 2
        mid_y = (from_pos[1] + to_pos[1]) / 2
        
        ax.text(mid_x, mid_y, rel['label'],
               ha='center', va='center',
               fontsize=7, style='italic',
               bbox=dict(boxstyle='round,pad=0.3',
                        facecolor='white',
                        edgecolor=rel['color'],
                        linewidth=1.5))
    
    # Add title
    ax.text(10, 13.5, 'IntelliTradeAI - Entity Relationship Diagram (ERD)',
           ha='center', va='center', fontsize=18, weight='bold',
           bbox=dict(boxstyle='round,pad=0.5',
                    facecolor='#1565C0',
                    edgecolor='black',
                    linewidth=2),
           color='white')
    
    # Add legend
    legend_x = 0.5
    legend_y = 0.5
    
    ax.text(legend_x, legend_y + 1.2, 'Legend:',
           ha='left', va='top', fontsize=11, weight='bold')
    
    # PK/FK legend
    ax.text(legend_x, legend_y + 0.9, 'PK: Primary Key',
           ha='left', va='top', fontsize=9, color='#D32F2F', weight='bold')
    ax.text(legend_x, legend_y + 0.6, 'FK: Foreign Key',
           ha='left', va='top', fontsize=9, color='#1976D2', weight='bold')
    
    # Relationship types
    ax.text(legend_x, legend_y + 0.3, '1:N = One-to-Many',
           ha='left', va='top', fontsize=9, style='italic')
    ax.text(legend_x, legend_y, 'N:1 = Many-to-One',
           ha='left', va='top', fontsize=9, style='italic')
    
    # Add metadata
    ax.text(19.5, 0.3, 'Created: November 19, 2025\nStorage: File-based (JSON)\nDatabase: PostgreSQL (future)',
           ha='right', va='bottom', fontsize=8, style='italic',
           color='gray')
    
    plt.tight_layout()
    plt.savefig('diagrams/erd_diagram.png', dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    print("✅ ERD diagram saved: diagrams/erd_diagram.png")
    plt.close()


def create_simplified_erd():
    """Create simplified ERD with main entities only"""
    
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Define simplified entities
    entities = {
        'Assets': {
            'pos': (3, 7.5),
            'attributes': [
                'symbol (PK)',
                'name',
                'rank',
                'market_cap'
            ],
            'color': '#BBDEFB'
        },
        'Price_Data': {
            'pos': (3, 4.5),
            'attributes': [
                'id (PK)',
                'symbol (FK)',
                'date',
                'OHLCV',
                'volume'
            ],
            'color': '#FFCCBC'
        },
        'Models': {
            'pos': (8, 7.5),
            'attributes': [
                'model_id (PK)',
                'symbol (FK)',
                'type',
                'accuracy',
                'status'
            ],
            'color': '#E1BEE7'
        },
        'Predictions': {
            'pos': (8, 4.5),
            'attributes': [
                'id (PK)',
                'model_id (FK)',
                'date',
                'signal',
                'confidence'
            ],
            'color': '#FFF9C4'
        },
        'Performance': {
            'pos': (13, 6),
            'attributes': [
                'id (PK)',
                'symbol (FK)',
                'returns',
                'volatility',
                'sharpe_ratio'
            ],
            'color': '#B2DFDB'
        }
    }
    
    # Draw entities
    for entity_name, entity_info in entities.items():
        x, y = entity_info['pos']
        attrs = entity_info['attributes']
        color = entity_info['color']
        
        height = 0.5 + (len(attrs) * 0.25)
        
        # Entity box
        box = FancyBboxPatch(
            (x - 1.5, y - height/2), 3, height,
            boxstyle="round,pad=0.1",
            edgecolor='black',
            facecolor=color,
            linewidth=2.5
        )
        ax.add_patch(box)
        
        # Header
        header = FancyBboxPatch(
            (x - 1.5, y + height/2 - 0.5), 3, 0.5,
            boxstyle="round,pad=0.05",
            edgecolor='black',
            facecolor='#263238',
            linewidth=2.5
        )
        ax.add_patch(header)
        
        ax.text(x, y + height/2 - 0.25, entity_name,
               ha='center', va='center', fontsize=13,
               weight='bold', color='white')
        
        # Attributes
        y_offset = y + height/2 - 0.5 - 0.2
        for attr in attrs:
            ax.text(x - 1.4, y_offset, attr, ha='left', va='center',
                   fontsize=10, family='monospace')
            y_offset -= 0.25
    
    # Draw relationships
    rels = [
        ((3, 7.0), (3, 5.5), '1:N', '#1976D2'),
        ((4.5, 7.5), (6.5, 7.5), '1:N', '#7B1FA2'),
        ((8, 7.0), (8, 5.5), '1:N', '#F57C00'),
        ((9.5, 7.5), (11.5, 6), '1:N', '#00ACC1'),
        ((4.5, 4.5), (6.5, 4.5), 'Uses', '#388E3C')
    ]
    
    for from_pos, to_pos, label, color in rels:
        arrow = FancyArrowPatch(
            from_pos, to_pos,
            arrowstyle='-|>',
            mutation_scale=25,
            linewidth=3,
            color=color,
            alpha=0.8
        )
        ax.add_patch(arrow)
        
        mid_x = (from_pos[0] + to_pos[0]) / 2
        mid_y = (from_pos[1] + to_pos[1]) / 2
        ax.text(mid_x, mid_y, label,
               ha='center', va='center', fontsize=10, weight='bold',
               bbox=dict(boxstyle='round,pad=0.4',
                        facecolor='white',
                        edgecolor=color,
                        linewidth=2))
    
    # Title
    ax.text(8, 9.3, 'IntelliTradeAI - Simplified ERD',
           ha='center', va='center', fontsize=20, weight='bold',
           bbox=dict(boxstyle='round,pad=0.6',
                    facecolor='#0D47A1',
                    edgecolor='black',
                    linewidth=2.5),
           color='white')
    
    # Description
    desc = "Core Data Model: 5 Main Entities with Relationships"
    ax.text(8, 8.5, desc,
           ha='center', va='center', fontsize=12, style='italic')
    
    plt.tight_layout()
    plt.savefig('diagrams/erd_simplified.png', dpi=300, bbox_inches='tight',
               facecolor='white')
    print("✅ Simplified ERD saved: diagrams/erd_simplified.png")
    plt.close()


if __name__ == '__main__':
    print("\n📊 Generating ERD Diagrams for IntelliTradeAI...")
    print("=" * 60)
    
    # Generate comprehensive ERD
    print("\n1. Creating comprehensive ERD (10 entities)...")
    create_erd()
    
    # Generate simplified ERD
    print("\n2. Creating simplified ERD (5 core entities)...")
    create_simplified_erd()
    
    print("\n" + "=" * 60)
    print("✅ All ERD diagrams generated successfully!")
    print("\nGenerated files:")
    print("  • diagrams/erd_diagram.png (comprehensive)")
    print("  • diagrams/erd_simplified.png (simplified)")
