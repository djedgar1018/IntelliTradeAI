"""
Generate ERD (Entity-Relationship Diagram) for IntelliTradeAI
Enhanced version with clearer, easier-to-read diagrams
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ConnectionPatch
import os

# Ensure diagrams directory exists
os.makedirs('diagrams', exist_ok=True)

def create_erd():
    """Create comprehensive ERD diagram with improved readability"""
    
    fig, ax = plt.subplots(figsize=(24, 16))
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 16)
    ax.axis('off')
    
    # Define entities with ONLY key attributes for clarity
    entities = {
        'Cryptocurrency': {
            'pos': (4, 13),
            'key_attrs': ['symbol', 'name', 'rank'],
            'other_count': 4,
            'color': '#E3F2FD',
            'category': 'Master Data'
        },
        'OHLCV_Data': {
            'pos': (4, 9),
            'key_attrs': ['id', 'symbol (FK)', 'date'],
            'other_count': 6,
            'color': '#FFF3E0',
            'category': 'Price Data'
        },
        'Technical_Indicators': {
            'pos': (10, 9),
            'key_attrs': ['id', 'ohlcv_id (FK)'],
            'other_count': 9,
            'color': '#E8F5E9',
            'category': 'Analytics'
        },
        'ML_Models': {
            'pos': (16, 13),
            'key_attrs': ['model_id', 'symbol (FK)', 'type'],
            'other_count': 8,
            'color': '#F3E5F5',
            'category': 'AI Models'
        },
        'Training_Sessions': {
            'pos': (16, 9),
            'key_attrs': ['session_id', 'model_id (FK)'],
            'other_count': 6,
            'color': '#FCE4EC',
            'category': 'AI Models'
        },
        'Predictions': {
            'pos': (22, 13),
            'key_attrs': ['id', 'model_id (FK)', 'signal'],
            'other_count': 6,
            'color': '#FFF9C4',
            'category': 'Outputs'
        },
        'Portfolio_Performance': {
            'pos': (22, 9),
            'key_attrs': ['id', 'symbol (FK)', 'returns'],
            'other_count': 6,
            'color': '#E0F7FA',
            'category': 'Outputs'
        },
        'API_Cache': {
            'pos': (4, 4.5),
            'key_attrs': ['cache_id', 'cache_key'],
            'other_count': 4,
            'color': '#EFEBE9',
            'category': 'Infrastructure'
        },
        'Feature_Engineering': {
            'pos': (10, 4.5),
            'key_attrs': ['feature_id', 'ohlcv_id (FK)'],
            'other_count': 5,
            'color': '#E1F5FE',
            'category': 'ML Pipeline'
        },
        'Backtest_Results': {
            'pos': (16, 4.5),
            'key_attrs': ['backtest_id', 'model_id (FK)'],
            'other_count': 8,
            'color': '#F1F8E9',
            'category': 'Validation'
        }
    }
    
    # Draw entities with improved styling
    for entity_name, entity_info in entities.items():
        x, y = entity_info['pos']
        key_attrs = entity_info['key_attrs']
        other_count = entity_info['other_count']
        color = entity_info['color']
        category = entity_info['category']
        
        # Calculate box dimensions
        width = 4.5
        attr_height = 0.35
        header_height = 0.6
        footer_height = 0.35
        body_height = len(key_attrs) * attr_height + 0.3
        total_height = header_height + body_height + footer_height
        
        # Draw main entity box with shadow
        shadow = FancyBboxPatch(
            (x - width/2 + 0.1, y - total_height/2 - 0.1), width, total_height,
            boxstyle="round,pad=0.15",
            edgecolor='none',
            facecolor='#CCCCCC',
            linewidth=0,
            alpha=0.3
        )
        ax.add_patch(shadow)
        
        box = FancyBboxPatch(
            (x - width/2, y - total_height/2), width, total_height,
            boxstyle="round,pad=0.15",
            edgecolor='#333333',
            facecolor=color,
            linewidth=3
        )
        ax.add_patch(box)
        
        # Draw header
        header = FancyBboxPatch(
            (x - width/2, y + total_height/2 - header_height), width, header_height,
            boxstyle="round,pad=0.1",
            edgecolor='#333333',
            facecolor='#1A237E',
            linewidth=3
        )
        ax.add_patch(header)
        
        # Entity name - larger and clearer
        ax.text(x, y + total_height/2 - header_height/2, entity_name,
               ha='center', va='center', fontsize=15,
               weight='bold', color='white',
               family='sans-serif')
        
        # Category label
        ax.text(x, y - total_height/2 + footer_height/2, f'[{category}]',
               ha='center', va='center', fontsize=9,
               style='italic', color='#555555')
        
        # Draw key attributes - larger font
        y_offset = y + total_height/2 - header_height - 0.25
        for attr in key_attrs:
            if '(FK)' in attr:
                # Foreign key - blue
                clean_attr = attr.replace(' (FK)', '')
                ax.text(x - width/2 + 0.25, y_offset, '🔗 ' + clean_attr,
                       ha='left', va='center',
                       fontsize=11, weight='bold', color='#1565C0',
                       family='sans-serif')
            elif attr == key_attrs[0]:
                # Primary key - red
                ax.text(x - width/2 + 0.25, y_offset, '🔑 ' + attr,
                       ha='left', va='center',
                       fontsize=11, weight='bold', color='#C62828',
                       family='sans-serif')
            else:
                # Regular attribute
                ax.text(x - width/2 + 0.25, y_offset, '• ' + attr,
                       ha='left', va='center',
                       fontsize=11, color='#212121',
                       family='sans-serif')
            y_offset -= attr_height
        
        # Show count of other attributes
        if other_count > 0:
            ax.text(x, y_offset, f'+ {other_count} more fields',
                   ha='center', va='center',
                   fontsize=9, style='italic', color='#757575')
    
    # Define relationships with clearer paths
    relationships = [
        {
            'from': 'Cryptocurrency',
            'to': 'OHLCV_Data',
            'label': '1 : N',
            'desc': 'has price history',
            'from_pos': (4, 11.5),
            'to_pos': (4, 10.5),
            'color': '#1976D2'
        },
        {
            'from': 'OHLCV_Data',
            'to': 'Technical_Indicators',
            'label': '1 : N',
            'desc': 'generates indicators',
            'from_pos': (6.25, 9),
            'to_pos': (7.75, 9),
            'color': '#388E3C'
        },
        {
            'from': 'OHLCV_Data',
            'to': 'Feature_Engineering',
            'label': '1 : N',
            'desc': 'creates features',
            'from_pos': (4, 7.5),
            'to_pos': (10, 6),
            'color': '#F57C00'
        },
        {
            'from': 'Cryptocurrency',
            'to': 'ML_Models',
            'label': '1 : N',
            'desc': 'trains models for',
            'from_pos': (6.25, 13),
            'to_pos': (13.75, 13),
            'color': '#7B1FA2'
        },
        {
            'from': 'ML_Models',
            'to': 'Training_Sessions',
            'label': '1 : N',
            'desc': 'has training sessions',
            'from_pos': (16, 11.5),
            'to_pos': (16, 10.5),
            'color': '#C2185B'
        },
        {
            'from': 'ML_Models',
            'to': 'Predictions',
            'label': '1 : N',
            'desc': 'generates predictions',
            'from_pos': (18.25, 13),
            'to_pos': (19.75, 13),
            'color': '#F9A825'
        },
        {
            'from': 'ML_Models',
            'to': 'Backtest_Results',
            'label': '1 : N',
            'desc': 'backtested with',
            'from_pos': (16, 11.5),
            'to_pos': (16, 6),
            'color': '#689F38'
        },
        {
            'from': 'Cryptocurrency',
            'to': 'Portfolio_Performance',
            'label': '1 : N',
            'desc': 'tracks performance',
            'from_pos': (6.25, 13),
            'to_pos': (19.75, 9),
            'color': '#00ACC1'
        },
        {
            'from': 'Cryptocurrency',
            'to': 'API_Cache',
            'label': '1 : N',
            'desc': 'cached data',
            'from_pos': (4, 11.5),
            'to_pos': (4, 6),
            'color': '#5D4037'
        }
    ]
    
    # Draw relationships with improved styling
    for rel in relationships:
        from_pos = rel['from_pos']
        to_pos = rel['to_pos']
        
        # Draw arrow with better visibility
        arrow = FancyArrowPatch(
            from_pos, to_pos,
            arrowstyle='-|>',
            mutation_scale=30,
            linewidth=3.5,
            color=rel['color'],
            linestyle=rel.get('style', 'solid'),
            alpha=0.85,
            zorder=1
        )
        ax.add_patch(arrow)
        
        # Add relationship label with better background
        mid_x = (from_pos[0] + to_pos[0]) / 2
        mid_y = (from_pos[1] + to_pos[1]) / 2
        
        # Cardinality label (1:N)
        ax.text(mid_x, mid_y + 0.35, rel['label'],
               ha='center', va='center',
               fontsize=12, weight='bold',
               bbox=dict(boxstyle='round,pad=0.5',
                        facecolor='white',
                        edgecolor=rel['color'],
                        linewidth=2.5),
               zorder=2)
        
        # Description label
        ax.text(mid_x, mid_y - 0.25, rel['desc'],
               ha='center', va='center',
               fontsize=9, style='italic',
               bbox=dict(boxstyle='round,pad=0.3',
                        facecolor='#F5F5F5',
                        edgecolor='#CCCCCC',
                        linewidth=1),
               zorder=2)
    
    # Add title with better styling
    title_box = FancyBboxPatch(
        (4, 14.5), 16, 1,
        boxstyle="round,pad=0.3",
        edgecolor='#1A237E',
        facecolor='#1565C0',
        linewidth=4
    )
    ax.add_patch(title_box)
    
    ax.text(12, 15.2, 'IntelliTradeAI',
           ha='center', va='center', fontsize=26, weight='bold',
           color='white', family='sans-serif')
    ax.text(12, 14.75, 'Entity Relationship Diagram',
           ha='center', va='center', fontsize=16,
           color='white', family='sans-serif')
    
    # Add enhanced legend
    legend_x = 1
    legend_y = 2.5
    legend_width = 6
    legend_height = 2
    
    legend_box = FancyBboxPatch(
        (legend_x, legend_y - legend_height), legend_width, legend_height,
        boxstyle="round,pad=0.2",
        edgecolor='#333333',
        facecolor='#FAFAFA',
        linewidth=2.5
    )
    ax.add_patch(legend_box)
    
    ax.text(legend_x + 0.3, legend_y - 0.3, 'LEGEND',
           ha='left', va='top', fontsize=13, weight='bold',
           color='#1A237E')
    
    # Legend items with icons
    y_pos = legend_y - 0.7
    ax.text(legend_x + 0.3, y_pos, '🔑  Primary Key',
           ha='left', va='center', fontsize=11, color='#C62828', weight='bold')
    
    y_pos -= 0.35
    ax.text(legend_x + 0.3, y_pos, '🔗  Foreign Key',
           ha='left', va='center', fontsize=11, color='#1565C0', weight='bold')
    
    y_pos -= 0.35
    ax.text(legend_x + 0.3, y_pos, '1 : N  One-to-Many',
           ha='left', va='center', fontsize=11, color='#424242')
    
    y_pos -= 0.35
    ax.text(legend_x + 0.3, y_pos, '•  Regular Attribute',
           ha='left', va='center', fontsize=11, color='#424242')
    
    # Add info box
    info_x = 17
    info_y = 2.5
    info_width = 6
    info_height = 2
    
    info_box = FancyBboxPatch(
        (info_x, info_y - info_height), info_width, info_height,
        boxstyle="round,pad=0.2",
        edgecolor='#333333',
        facecolor='#FFFDE7',
        linewidth=2.5
    )
    ax.add_patch(info_box)
    
    ax.text(info_x + 0.3, info_y - 0.3, 'SYSTEM INFO',
           ha='left', va='top', fontsize=13, weight='bold',
           color='#F57F17')
    
    info_text = [
        '10 Entities',
        '9 Relationships',
        'PostgreSQL-Ready',
        'Nov 19, 2025'
    ]
    
    y_pos = info_y - 0.75
    for text in info_text:
        ax.text(info_x + 0.3, y_pos, f'• {text}',
               ha='left', va='center', fontsize=10, color='#424242')
        y_pos -= 0.35
    
    plt.tight_layout()
    plt.savefig('diagrams/erd_diagram.png', dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    print("✅ Enhanced ERD diagram saved: diagrams/erd_diagram.png")
    plt.close()


def create_simplified_erd():
    """Create simplified ERD with improved clarity"""
    
    fig, ax = plt.subplots(figsize=(20, 12))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Define simplified entities - larger and clearer
    entities = {
        'Assets\n(Cryptocurrency)': {
            'pos': (4, 9),
            'key_attrs': ['🔑 symbol', '• name', '• rank'],
            'color': '#BBDEFB',
            'icon': '💎'
        },
        'Price Data\n(OHLCV)': {
            'pos': (4, 5),
            'key_attrs': ['🔑 id', '🔗 symbol', '• date', '• OHLCV'],
            'color': '#FFCCBC',
            'icon': '📊'
        },
        'ML Models': {
            'pos': (10, 9),
            'key_attrs': ['🔑 model_id', '🔗 symbol', '• type', '• accuracy'],
            'color': '#E1BEE7',
            'icon': '🤖'
        },
        'Predictions\n(Signals)': {
            'pos': (10, 5),
            'key_attrs': ['🔑 id', '🔗 model_id', '• signal', '• confidence'],
            'color': '#FFF9C4',
            'icon': '⚡'
        },
        'Performance\n(Portfolio)': {
            'pos': (16, 7),
            'key_attrs': ['🔑 id', '🔗 symbol', '• returns', '• volatility'],
            'color': '#B2DFDB',
            'icon': '📈'
        }
    }
    
    # Draw entities with enhanced design
    for entity_name, entity_info in entities.items():
        x, y = entity_info['pos']
        key_attrs = entity_info['key_attrs']
        color = entity_info['color']
        icon = entity_info['icon']
        
        width = 5
        attr_height = 0.4
        header_height = 1.2
        body_height = len(key_attrs) * attr_height + 0.4
        total_height = header_height + body_height
        
        # Shadow effect
        shadow = FancyBboxPatch(
            (x - width/2 + 0.15, y - total_height/2 - 0.15), width, total_height,
            boxstyle="round,pad=0.2",
            edgecolor='none',
            facecolor='#999999',
            alpha=0.25
        )
        ax.add_patch(shadow)
        
        # Main box
        box = FancyBboxPatch(
            (x - width/2, y - total_height/2), width, total_height,
            boxstyle="round,pad=0.2",
            edgecolor='#212121',
            facecolor=color,
            linewidth=3.5
        )
        ax.add_patch(box)
        
        # Header
        header = FancyBboxPatch(
            (x - width/2, y + total_height/2 - header_height), width, header_height,
            boxstyle="round,pad=0.15",
            edgecolor='#212121',
            facecolor='#263238',
            linewidth=3.5
        )
        ax.add_patch(header)
        
        # Icon and entity name
        ax.text(x, y + total_height/2 - 0.3, icon,
               ha='center', va='center', fontsize=28)
        
        ax.text(x, y + total_height/2 - 0.85, entity_name,
               ha='center', va='center', fontsize=14,
               weight='bold', color='white',
               family='sans-serif')
        
        # Attributes with better spacing
        y_offset = y + total_height/2 - header_height - 0.35
        for attr in key_attrs:
            ax.text(x - width/2 + 0.3, y_offset, attr,
                   ha='left', va='center',
                   fontsize=12, family='sans-serif',
                   weight='normal' if '•' in attr else 'bold')
            y_offset -= attr_height
    
    # Draw relationships with labels
    relationships = [
        {
            'from_pos': (4, 7.5),
            'to_pos': (4, 6.5),
            'label': '1 : N',
            'desc': 'contains',
            'color': '#1976D2'
        },
        {
            'from_pos': (6.5, 9),
            'to_pos': (7.5, 9),
            'label': '1 : N',
            'desc': 'trains',
            'color': '#7B1FA2'
        },
        {
            'from_pos': (10, 7.5),
            'to_pos': (10, 6.5),
            'label': '1 : N',
            'desc': 'produces',
            'color': '#F57C00'
        },
        {
            'from_pos': (12.5, 9),
            'to_pos': (13.5, 7),
            'label': '1 : N',
            'desc': 'tracks',
            'color': '#00ACC1'
        },
        {
            'from_pos': (6.5, 5),
            'to_pos': (7.5, 5),
            'label': 'uses',
            'desc': 'data from',
            'color': '#388E3C'
        }
    ]
    
    for rel in relationships:
        arrow = FancyArrowPatch(
            rel['from_pos'], rel['to_pos'],
            arrowstyle='-|>',
            mutation_scale=35,
            linewidth=4,
            color=rel['color'],
            alpha=0.9
        )
        ax.add_patch(arrow)
        
        mid_x = (rel['from_pos'][0] + rel['to_pos'][0]) / 2
        mid_y = (rel['from_pos'][1] + rel['to_pos'][1]) / 2
        
        # Relationship type
        ax.text(mid_x, mid_y + 0.4, rel['label'],
               ha='center', va='center',
               fontsize=13, weight='bold',
               bbox=dict(boxstyle='round,pad=0.5',
                        facecolor='white',
                        edgecolor=rel['color'],
                        linewidth=3))
        
        # Description
        ax.text(mid_x, mid_y - 0.3, rel['desc'],
               ha='center', va='center',
               fontsize=10, style='italic',
               color='#424242')
    
    # Title
    title_box = FancyBboxPatch(
        (3, 10.5), 14, 1.2,
        boxstyle="round,pad=0.3",
        edgecolor='#0D47A1',
        facecolor='#1565C0',
        linewidth=4
    )
    ax.add_patch(title_box)
    
    ax.text(10, 11.3, 'IntelliTradeAI - Core Data Model',
           ha='center', va='center', fontsize=24, weight='bold',
           color='white')
    ax.text(10, 10.85, '5 Essential Entities',
           ha='center', va='center', fontsize=14,
           color='white', style='italic')
    
    # Info panel
    info_box = FancyBboxPatch(
        (0.5, 0.5), 19, 2,
        boxstyle="round,pad=0.2",
        edgecolor='#424242',
        facecolor='#F5F5F5',
        linewidth=2.5
    )
    ax.add_patch(info_box)
    
    info_items = [
        ('💎', 'Assets: Top 10 cryptocurrencies'),
        ('📊', 'Price Data: 1,850+ OHLCV records'),
        ('🤖', 'ML Models: Random Forest, XGBoost, LSTM'),
        ('⚡', 'Predictions: BUY/SELL/HOLD signals'),
        ('📈', 'Performance: Returns & analytics')
    ]
    
    x_spacing = 3.8
    for i, (icon, text) in enumerate(info_items):
        x_pos = 1 + (i * x_spacing)
        ax.text(x_pos, 1.5, icon, ha='left', va='center', fontsize=18)
        ax.text(x_pos + 0.5, 1.5, text, ha='left', va='center',
               fontsize=10, family='sans-serif')
    
    plt.tight_layout()
    plt.savefig('diagrams/erd_simplified.png', dpi=300, bbox_inches='tight',
               facecolor='white')
    print("✅ Enhanced simplified ERD saved: diagrams/erd_simplified.png")
    plt.close()


if __name__ == '__main__':
    print("\n📊 Generating Enhanced ERD Diagrams for IntelliTradeAI...")
    print("=" * 70)
    print("\n🎨 Improvements:")
    print("  • Larger, more readable fonts")
    print("  • Better color contrast and visual hierarchy")
    print("  • Clearer relationship arrows and labels")
    print("  • Icons and shadows for better visual appeal")
    print("  • Simplified attribute display (key fields only)")
    print("  • Enhanced legend and info panels")
    print("=" * 70)
    
    # Generate comprehensive ERD
    print("\n1. Creating comprehensive ERD (10 entities)...")
    create_erd()
    
    # Generate simplified ERD
    print("\n2. Creating simplified ERD (5 core entities)...")
    create_simplified_erd()
    
    print("\n" + "=" * 70)
    print("✅ All ERD diagrams generated successfully!")
    print("\n📁 Generated files:")
    print("  • diagrams/erd_diagram.png (comprehensive, easier to read)")
    print("  • diagrams/erd_simplified.png (simplified, clearer layout)")
    print("\n💡 View the diagrams to see the improved clarity!")
