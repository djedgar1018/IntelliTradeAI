"""
Generate Use Case Diagrams for IntelliTradeAI
Creates professional UML-style diagrams as PNG images
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Ellipse, FancyArrowPatch
import os

# Create diagrams directory
os.makedirs('diagrams', exist_ok=True)

def create_actor(ax, x, y, name, color='lightblue'):
    """Draw a stick figure actor"""
    # Head
    head = plt.Circle((x, y), 0.3, color=color, ec='black', linewidth=2)
    ax.add_patch(head)
    # Body
    ax.plot([x, x], [y-0.3, y-1.0], 'k-', linewidth=2)
    # Arms
    ax.plot([x-0.4, x+0.4], [y-0.6, y-0.6], 'k-', linewidth=2)
    # Legs
    ax.plot([x, x-0.3], [y-1.0, y-1.6], 'k-', linewidth=2)
    ax.plot([x, x+0.3], [y-1.0, y-1.6], 'k-', linewidth=2)
    # Name
    ax.text(x, y-2.0, name, ha='center', va='top', fontsize=10, fontweight='bold')

def create_use_case(ax, x, y, width, height, text, color='lightyellow'):
    """Draw an ellipse use case"""
    ellipse = Ellipse((x, y), width, height, color=color, ec='black', linewidth=2)
    ax.add_patch(ellipse)
    # Wrap text if too long
    words = text.split()
    if len(words) > 3:
        mid = len(words) // 2
        line1 = ' '.join(words[:mid])
        line2 = ' '.join(words[mid:])
        ax.text(x, y+0.1, line1, ha='center', va='center', fontsize=9, wrap=True)
        ax.text(x, y-0.1, line2, ha='center', va='center', fontsize=9, wrap=True)
    else:
        ax.text(x, y, text, ha='center', va='center', fontsize=9, wrap=True)

def draw_connection(ax, x1, y1, x2, y2, style='solid'):
    """Draw a line connecting actor to use case"""
    ax.plot([x1, x2], [y1, y2], 'k-', linewidth=1.5, linestyle=style)

def draw_include_extend(ax, x1, y1, x2, y2, label):
    """Draw dashed arrow with label for include/extend"""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle='->', mutation_scale=20,
                           linestyle='dashed', linewidth=1.5)
    ax.add_patch(arrow)
    mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
    ax.text(mid_x, mid_y + 0.3, f'<<{label}>>', ha='center', fontsize=8, style='italic')

def create_system_boundary(ax, x, y, width, height, title):
    """Draw system boundary box"""
    box = FancyBboxPatch((x-width/2, y-height/2), width, height,
                         boxstyle="round,pad=0.1", 
                         edgecolor='navy', facecolor='none',
                         linewidth=3)
    ax.add_patch(box)
    ax.text(x-width/2+0.5, y+height/2-0.3, title, 
            fontsize=12, fontweight='bold', color='navy')

# ============================================================================
# DIAGRAM 1: Core Trading Operations
# ============================================================================
def create_core_trading_diagram():
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(7, 9.5, 'Use Case Diagram: Core Trading Operations', 
            ha='center', fontsize=16, fontweight='bold')
    
    # System boundary
    create_system_boundary(ax, 7, 5, 10, 7.5, 'IntelliTradeAI System')
    
    # Actors
    create_actor(ax, 1.5, 7, 'Day Trader', 'lightcoral')
    create_actor(ax, 1.5, 4.5, 'Swing Trader', 'lightgreen')
    create_actor(ax, 12.5, 6, 'Long-term\nInvestor', 'lightblue')
    
    # Use cases
    create_use_case(ax, 5, 7.5, 2.5, 1, 'Get Instant Prediction')
    create_use_case(ax, 5, 6, 2.5, 1, 'View Confidence Score')
    create_use_case(ax, 5, 4.5, 2.5, 1, 'Select Asset')
    create_use_case(ax, 8, 7, 2.5, 1, 'View BUY/SELL Signal')
    create_use_case(ax, 8, 5.5, 2.5, 1, 'Monitor Watchlist')
    create_use_case(ax, 8, 4, 2.5, 1, 'View Price Chart')
    create_use_case(ax, 11, 6.5, 2.5, 1, 'Validate Decision')
    create_use_case(ax, 11, 5, 2.5, 1, 'Check Signal History')
    
    # Connections - Day Trader
    draw_connection(ax, 2, 7, 3.8, 7.5)
    draw_connection(ax, 2, 7, 3.8, 6)
    draw_connection(ax, 2, 7, 6.8, 7)
    draw_connection(ax, 2, 6.5, 6.8, 5.5)
    
    # Connections - Swing Trader
    draw_connection(ax, 2, 4.5, 3.8, 4.5)
    draw_connection(ax, 2, 4.8, 6.8, 5.5)
    draw_connection(ax, 2, 4.5, 6.8, 4)
    
    # Connections - Long-term Investor
    draw_connection(ax, 12, 6, 9.8, 6.5)
    draw_connection(ax, 12, 5.8, 9.8, 5)
    draw_connection(ax, 12, 6.2, 9.2, 7)
    
    # Include relationships
    draw_include_extend(ax, 5, 7, 5, 6, 'include')
    draw_include_extend(ax, 5, 6, 5, 4.8, 'include')
    
    plt.tight_layout()
    plt.savefig('diagrams/01_core_trading_use_case.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created: diagrams/01_core_trading_use_case.png")

# ============================================================================
# DIAGRAM 2: Model Management
# ============================================================================
def create_model_management_diagram():
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(7, 9.5, 'Use Case Diagram: Model Management', 
            ha='center', fontsize=16, fontweight='bold')
    
    # System boundary
    create_system_boundary(ax, 7, 5, 10, 7.5, 'IntelliTradeAI System')
    
    # Actors
    create_actor(ax, 1.5, 6.5, 'Data Scientist', 'lavender')
    create_actor(ax, 1.5, 3.5, 'Advanced User', 'lightyellow')
    create_actor(ax, 12.5, 5.5, 'System\nScheduler', 'lightgray')
    
    # Use cases
    create_use_case(ax, 5, 7.5, 2.5, 1, 'Train New Model')
    create_use_case(ax, 5, 6, 2.5, 1, 'Select Algorithm')
    create_use_case(ax, 5, 4.5, 2.5, 1, 'View Training Progress')
    create_use_case(ax, 8, 7.5, 2.5, 1, 'Compare Models')
    create_use_case(ax, 8, 6, 2.5, 1, 'View Accuracy Metrics')
    create_use_case(ax, 8, 4.5, 2.5, 1, 'Manage Model Cache')
    create_use_case(ax, 11, 6.5, 2.5, 1, 'Auto-Retrain Models')
    create_use_case(ax, 11, 5, 2.5, 1, 'Load Cached Model')
    create_use_case(ax, 5, 3, 2.5, 1, 'Fetch Training Data')
    
    # Connections - Data Scientist
    draw_connection(ax, 2, 6.5, 3.8, 7.5)
    draw_connection(ax, 2, 6.5, 3.8, 6)
    draw_connection(ax, 2, 6.5, 6.8, 7.5)
    draw_connection(ax, 2, 6.5, 6.8, 6)
    draw_connection(ax, 2, 6.2, 6.8, 4.5)
    
    # Connections - Advanced User
    draw_connection(ax, 2, 3.8, 3.8, 4.5)
    draw_connection(ax, 2, 3.8, 6.8, 4.5)
    draw_connection(ax, 2, 4, 9.8, 5)
    
    # Connections - System Scheduler
    draw_connection(ax, 12, 5.5, 9.8, 6.5)
    draw_connection(ax, 12, 5.5, 9.8, 5)
    
    # Include/Extend relationships
    draw_include_extend(ax, 5, 7, 5, 6, 'include')
    draw_include_extend(ax, 5, 7, 5, 3.5, 'include')
    draw_include_extend(ax, 8, 7, 8, 6, 'include')
    draw_include_extend(ax, 5, 7.5, 9.8, 5, 'extend')
    
    plt.tight_layout()
    plt.savefig('diagrams/02_model_management_use_case.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created: diagrams/02_model_management_use_case.png")

# ============================================================================
# DIAGRAM 3: Analytics & Risk Management
# ============================================================================
def create_analytics_diagram():
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(7, 9.5, 'Use Case Diagram: Analytics & Risk Management', 
            ha='center', fontsize=16, fontweight='bold')
    
    # System boundary
    create_system_boundary(ax, 7, 5, 10, 7.5, 'IntelliTradeAI System')
    
    # Actors
    create_actor(ax, 1.5, 7, 'Portfolio\nManager', 'lightcoral')
    create_actor(ax, 1.5, 4, 'Financial\nAdvisor', 'lightgreen')
    create_actor(ax, 12.5, 5.5, 'Risk\nAnalyst', 'lightsalmon')
    
    # Use cases
    create_use_case(ax, 5, 7.5, 2.5, 1, 'Run Backtest')
    create_use_case(ax, 5, 6, 2.5, 1, 'View Performance Metrics')
    create_use_case(ax, 5, 4.5, 2.5, 1, 'Analyze Technical Indicators')
    create_use_case(ax, 8, 7.5, 2.5, 1, 'Calculate Risk Metrics')
    create_use_case(ax, 8, 6, 2.5, 1, 'Set Stop-Loss Levels')
    create_use_case(ax, 8, 4.5, 2.5, 1, 'Set Take-Profit Levels')
    create_use_case(ax, 11, 6.5, 2.5, 1, 'Track Portfolio P&L')
    create_use_case(ax, 11, 5, 2.5, 1, 'Generate Reports')
    create_use_case(ax, 5, 3, 2.5, 1, 'View SHAP Analysis')
    
    # Connections - Portfolio Manager
    draw_connection(ax, 2, 7, 3.8, 7.5)
    draw_connection(ax, 2, 7, 3.8, 6)
    draw_connection(ax, 2, 6.8, 6.8, 7.5)
    draw_connection(ax, 2, 6.8, 9.8, 6.5)
    draw_connection(ax, 2, 6.8, 9.8, 5)
    
    # Connections - Financial Advisor
    draw_connection(ax, 2, 4.2, 3.8, 4.5)
    draw_connection(ax, 2, 4.2, 3.8, 3.5)
    draw_connection(ax, 2, 4.5, 6.8, 6)
    draw_connection(ax, 2, 4.5, 6.8, 4.5)
    
    # Connections - Risk Analyst
    draw_connection(ax, 12, 5.5, 9.2, 7.5)
    draw_connection(ax, 12, 5.5, 9.2, 6)
    draw_connection(ax, 12, 5.5, 9.2, 4.5)
    
    # Include relationships
    draw_include_extend(ax, 5, 7, 5, 6, 'include')
    draw_include_extend(ax, 8, 7, 8, 6, 'include')
    draw_include_extend(ax, 8, 6, 8, 4.8, 'extend')
    
    plt.tight_layout()
    plt.savefig('diagrams/03_analytics_risk_use_case.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created: diagrams/03_analytics_risk_use_case.png")

# ============================================================================
# DIAGRAM 4: API & Automation
# ============================================================================
def create_api_automation_diagram():
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(7, 9.5, 'Use Case Diagram: API & Automation', 
            ha='center', fontsize=16, fontweight='bold')
    
    # System boundary
    create_system_boundary(ax, 7, 5, 10, 7.5, 'IntelliTradeAI System')
    
    # Actors
    create_actor(ax, 1.5, 7, 'Algorithm\nDeveloper', 'lavender')
    create_actor(ax, 1.5, 4, 'Trading Bot', 'lightgray')
    create_actor(ax, 12.5, 5.5, 'External\nSystem', 'lightyellow')
    
    # Use cases
    create_use_case(ax, 5, 7.5, 2.5, 1, 'Access REST API')
    create_use_case(ax, 5, 6, 2.5, 1, 'Authenticate API Key')
    create_use_case(ax, 5, 4.5, 2.5, 1, 'Get Predictions via API')
    create_use_case(ax, 8, 7.5, 2.5, 1, 'Trigger Model Retrain')
    create_use_case(ax, 8, 6, 2.5, 1, 'Fetch Market Data')
    create_use_case(ax, 8, 4.5, 2.5, 1, 'Configure Alerts')
    create_use_case(ax, 11, 6.5, 2.5, 1, 'Receive Webhooks')
    create_use_case(ax, 11, 5, 2.5, 1, 'Execute Automated Trades')
    create_use_case(ax, 5, 3, 2.5, 1, 'View API Documentation')
    
    # Connections - Algorithm Developer
    draw_connection(ax, 2, 7, 3.8, 7.5)
    draw_connection(ax, 2, 7, 3.8, 6)
    draw_connection(ax, 2, 6.8, 3.8, 4.5)
    draw_connection(ax, 2, 6.5, 3.8, 3.5)
    draw_connection(ax, 2, 6.8, 6.8, 7.5)
    
    # Connections - Trading Bot
    draw_connection(ax, 2, 4.2, 3.8, 4.5)
    draw_connection(ax, 2, 4.5, 6.8, 6)
    draw_connection(ax, 2, 4.5, 6.8, 4.5)
    draw_connection(ax, 2, 4.3, 9.8, 5)
    
    # Connections - External System
    draw_connection(ax, 12, 5.8, 9.8, 6.5)
    draw_connection(ax, 12, 5.5, 9.8, 5)
    draw_connection(ax, 12, 5.2, 6.8, 4.8)
    
    # Include relationships
    draw_include_extend(ax, 5, 7.2, 5, 6, 'include')
    draw_include_extend(ax, 5, 4.5, 8, 6, 'extend')
    draw_include_extend(ax, 11, 6, 11, 5.5, 'extend')
    
    plt.tight_layout()
    plt.savefig('diagrams/04_api_automation_use_case.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created: diagrams/04_api_automation_use_case.png")

# ============================================================================
# DIAGRAM 5: System Overview (All Actors & Use Cases)
# ============================================================================
def create_system_overview_diagram():
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Title
    ax.text(8, 11.5, 'Complete System Overview: IntelliTradeAI', 
            ha='center', fontsize=18, fontweight='bold')
    
    # System boundary
    create_system_boundary(ax, 8, 6, 12, 9, 'IntelliTradeAI Platform')
    
    # Actors - Left side
    create_actor(ax, 1.5, 9.5, 'Day Trader', 'lightcoral')
    create_actor(ax, 1.5, 7.5, 'Portfolio\nManager', 'lightgreen')
    create_actor(ax, 1.5, 5.5, 'Data\nScientist', 'lavender')
    create_actor(ax, 1.5, 3.5, 'Algorithm\nDeveloper', 'lightyellow')
    
    # Actors - Right side
    create_actor(ax, 14.5, 8.5, 'Financial\nAdvisor', 'lightblue')
    create_actor(ax, 14.5, 6, 'Trading Bot', 'lightgray')
    create_actor(ax, 14.5, 3.5, 'External\nSystem', 'lightsalmon')
    
    # Core Trading Use Cases
    create_use_case(ax, 4.5, 10, 2, 0.8, 'Get Predictions')
    create_use_case(ax, 4.5, 8.8, 2, 0.8, 'View Signals')
    create_use_case(ax, 4.5, 7.6, 2, 0.8, 'Monitor Watchlist')
    
    # Model Management Use Cases
    create_use_case(ax, 7, 9.5, 2, 0.8, 'Train Models')
    create_use_case(ax, 7, 8.3, 2, 0.8, 'Compare Algorithms')
    create_use_case(ax, 7, 7.1, 2, 0.8, 'Manage Cache')
    
    # Analytics Use Cases
    create_use_case(ax, 9.5, 9.5, 2, 0.8, 'Run Backtest')
    create_use_case(ax, 9.5, 8.3, 2, 0.8, 'Risk Analysis')
    create_use_case(ax, 9.5, 7.1, 2, 0.8, 'View Metrics')
    
    # API & Automation Use Cases
    create_use_case(ax, 12, 9, 2, 0.8, 'REST API Access')
    create_use_case(ax, 12, 7.8, 2, 0.8, 'Webhooks')
    create_use_case(ax, 12, 6.6, 2, 0.8, 'Auto-Trading')
    
    # Data & Technical Use Cases
    create_use_case(ax, 5.5, 5.5, 2, 0.8, 'Fetch Market Data')
    create_use_case(ax, 5.5, 4.3, 2, 0.8, 'Calculate Indicators')
    create_use_case(ax, 8, 5.5, 2, 0.8, 'SHAP Analysis')
    create_use_case(ax, 8, 4.3, 2, 0.8, 'Generate Reports')
    create_use_case(ax, 10.5, 5, 2, 0.8, 'Set Alerts')
    create_use_case(ax, 10.5, 3.8, 2, 0.8, 'Track Portfolio')
    
    # Connections - Day Trader
    draw_connection(ax, 2, 9.5, 3.5, 10)
    draw_connection(ax, 2, 9.5, 3.5, 8.8)
    draw_connection(ax, 2, 9.5, 3.5, 7.6)
    
    # Connections - Portfolio Manager
    draw_connection(ax, 2, 7.5, 8.5, 9.5)
    draw_connection(ax, 2, 7.5, 8.5, 8.3)
    draw_connection(ax, 2, 7.2, 9.5, 5)
    
    # Connections - Data Scientist
    draw_connection(ax, 2, 5.5, 6, 9.5)
    draw_connection(ax, 2, 5.5, 6, 8.3)
    draw_connection(ax, 2, 5.5, 7, 5.5)
    
    # Connections - Algorithm Developer
    draw_connection(ax, 2, 3.8, 11, 9)
    draw_connection(ax, 2, 3.8, 11, 7.8)
    draw_connection(ax, 2, 4, 4.5, 5.5)
    
    # Connections - Financial Advisor
    draw_connection(ax, 14, 8.5, 10.5, 9.5)
    draw_connection(ax, 14, 8.3, 10.5, 8.3)
    draw_connection(ax, 14, 8.2, 9, 4.3)
    
    # Connections - Trading Bot
    draw_connection(ax, 14, 6, 13, 6.6)
    draw_connection(ax, 14, 6.2, 13, 7.8)
    draw_connection(ax, 14, 5.8, 9.5, 5)
    
    # Connections - External System
    draw_connection(ax, 14, 3.8, 13, 7.8)
    draw_connection(ax, 14, 4, 11.5, 5)
    
    plt.tight_layout()
    plt.savefig('diagrams/05_system_overview_use_case.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created: diagrams/05_system_overview_use_case.png")

# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    print("\n🎨 Generating Use Case Diagrams for IntelliTradeAI...\n")
    
    create_core_trading_diagram()
    create_model_management_diagram()
    create_analytics_diagram()
    create_api_automation_diagram()
    create_system_overview_diagram()
    
    print("\n✅ All diagrams generated successfully!")
    print("📁 Location: diagrams/")
    print("\nGenerated files:")
    print("  1. 01_core_trading_use_case.png")
    print("  2. 02_model_management_use_case.png")
    print("  3. 03_analytics_risk_use_case.png")
    print("  4. 04_api_automation_use_case.png")
    print("  5. 05_system_overview_use_case.png")
