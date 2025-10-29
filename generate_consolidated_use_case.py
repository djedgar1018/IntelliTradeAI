import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Ellipse, FancyArrowPatch
import numpy as np

class ConsolidatedUseCaseDiagram:
    def __init__(self, width=24, height=18):
        self.fig, self.ax = plt.subplots(figsize=(width, height))
        self.ax.set_xlim(0, width)
        self.ax.set_ylim(0, height)
        self.ax.axis('off')
        
    def add_system_boundary(self, x, y, width, height, name):
        """Add system boundary box"""
        rect = FancyBboxPatch((x, y), width, height,
                             boxstyle="round,pad=0.1",
                             edgecolor='black', facecolor='white',
                             linewidth=2, linestyle='--')
        self.ax.add_patch(rect)
        self.ax.text(x + 0.3, y + height - 0.3, name,
                    fontsize=14, fontweight='bold')
    
    def add_actor(self, x, y, name, color='lightblue'):
        """Add stick figure actor"""
        # Head
        circle = plt.Circle((x, y + 0.6), 0.15, color=color, ec='black', linewidth=1.5)
        self.ax.add_patch(circle)
        # Body
        self.ax.plot([x, x], [y + 0.45, y + 0.1], 'k-', linewidth=2)
        # Arms
        self.ax.plot([x - 0.2, x + 0.2], [y + 0.35, y + 0.35], 'k-', linewidth=2)
        # Legs
        self.ax.plot([x, x - 0.15], [y + 0.1, y - 0.2], 'k-', linewidth=2)
        self.ax.plot([x, x + 0.15], [y + 0.1, y - 0.2], 'k-', linewidth=2)
        # Name
        self.ax.text(x, y - 0.5, name, ha='center', fontsize=9, fontweight='bold')
    
    def add_use_case(self, x, y, name, width=1.8, height=0.5):
        """Add use case ellipse"""
        ellipse = Ellipse((x, y), width, height,
                         edgecolor='black', facecolor='lightyellow',
                         linewidth=1.5)
        self.ax.add_patch(ellipse)
        
        # Wrap text for long names
        words = name.split()
        lines = []
        current_line = []
        for word in words:
            current_line.append(word)
            if len(' '.join(current_line)) > 20:
                lines.append(' '.join(current_line[:-1]))
                current_line = [word]
        if current_line:
            lines.append(' '.join(current_line))
        
        self.ax.text(x, y, '\n'.join(lines), ha='center', va='center',
                    fontsize=8, fontweight='normal')
    
    def add_connection(self, x1, y1, x2, y2, style='solid'):
        """Add connection line between actor and use case"""
        if style == 'solid':
            self.ax.plot([x1, x2], [y1, y2], 'k-', linewidth=1)
        else:
            self.ax.plot([x1, x2], [y1, y2], 'k--', linewidth=1)
    
    def save(self, filename):
        plt.tight_layout()
        plt.savefig(f'diagrams/{filename}', dpi=300, bbox_inches='tight')
        plt.close()

def create_consolidated_diagram():
    """Create one comprehensive use case diagram with all actors and use cases"""
    diagram = ConsolidatedUseCaseDiagram(width=26, height=20)
    
    # System boundary
    diagram.add_system_boundary(3, 1, 18, 17.5, 'IntelliTradeAI System')
    
    # ============ LEFT SIDE ACTORS ============
    # Primary trading actors
    diagram.add_actor(1, 16, 'Day Trader', 'lightblue')
    diagram.add_actor(1, 13.5, 'Swing Trader', 'lightblue')
    diagram.add_actor(1, 11, 'Long-term\nInvestor', 'lightblue')
    diagram.add_actor(1, 8.5, 'Portfolio\nManager', 'lightgreen')
    diagram.add_actor(1, 6, 'Data\nScientist', 'lightcoral')
    diagram.add_actor(1, 3.5, 'Risk\nAnalyst', 'lightyellow')
    
    # ============ RIGHT SIDE ACTORS ============
    # Technical and automated actors
    diagram.add_actor(23, 16, 'Algorithm\nDeveloper', 'lightcoral')
    diagram.add_actor(23, 13.5, 'Trading Bot', 'lightgray')
    diagram.add_actor(23, 11, 'External\nSystem', 'lightgray')
    diagram.add_actor(23, 8.5, 'Financial\nAdvisor', 'lightgreen')
    
    # ============ USE CASES - TOP ROW (Real-time Trading) ============
    uc1_x, uc1_y = 6.5, 16.5
    uc2_x, uc2_y = 10, 16.5
    uc3_x, uc3_y = 13.5, 16.5
    uc4_x, uc4_y = 17, 16.5
    
    diagram.add_use_case(uc1_x, uc1_y, 'Get Instant\nBUY/SELL/HOLD Signals')
    diagram.add_use_case(uc2_x, uc2_y, 'View Real-time\nPrice Charts')
    diagram.add_use_case(uc3_x, uc3_y, 'Set Price Alerts\n& Notifications')
    diagram.add_use_case(uc4_x, uc4_y, 'Execute Automated\nTrading Strategies')
    
    # ============ USE CASES - SECOND ROW (Analysis & Research) ============
    uc5_x, uc5_y = 6.5, 14
    uc6_x, uc6_y = 10, 14
    uc7_x, uc7_y = 13.5, 14
    uc8_x, uc8_y = 17, 14
    
    diagram.add_use_case(uc5_x, uc5_y, 'Analyze Historical\nPerformance')
    diagram.add_use_case(uc6_x, uc6_y, 'Compare Multiple\nAssets')
    diagram.add_use_case(uc7_x, uc7_y, 'View Technical\nIndicators')
    diagram.add_use_case(uc8_x, uc8_y, 'Access Market\nSentiment Data')
    
    # ============ USE CASES - THIRD ROW (Portfolio Management) ============
    uc9_x, uc9_y = 6.5, 11.5
    uc10_x, uc10_y = 10, 11.5
    uc11_x, uc11_y = 13.5, 11.5
    uc12_x, uc12_y = 17, 11.5
    
    diagram.add_use_case(uc9_x, uc9_y, 'Track Portfolio\nPerformance')
    diagram.add_use_case(uc10_x, uc10_y, 'Assess Risk\nExposure')
    diagram.add_use_case(uc11_x, uc11_y, 'Generate Portfolio\nRecommendations')
    diagram.add_use_case(uc12_x, uc12_y, 'Rebalance Portfolio\nAutomatically')
    
    # ============ USE CASES - FOURTH ROW (Model Management) ============
    uc13_x, uc13_y = 6.5, 9
    uc14_x, uc14_y = 10, 9
    uc15_x, uc15_y = 13.5, 9
    uc16_x, uc16_y = 17, 9
    
    diagram.add_use_case(uc13_x, uc13_y, 'Train Custom\nML Models')
    diagram.add_use_case(uc14_x, uc14_y, 'Compare Model\nPerformance')
    diagram.add_use_case(uc15_x, uc15_y, 'Tune Model\nHyperparameters')
    diagram.add_use_case(uc16_x, uc16_y, 'View Model\nExplainability')
    
    # ============ USE CASES - FIFTH ROW (Backtesting & Validation) ============
    uc17_x, uc17_y = 6.5, 6.5
    uc18_x, uc18_y = 10, 6.5
    uc19_x, uc19_y = 13.5, 6.5
    uc20_x, uc20_y = 17, 6.5
    
    diagram.add_use_case(uc17_x, uc17_y, 'Run Backtests on\nHistorical Data')
    diagram.add_use_case(uc18_x, uc18_y, 'Validate Trading\nStrategies')
    diagram.add_use_case(uc19_x, uc19_y, 'Calculate Sharpe\nRatio & Metrics')
    diagram.add_use_case(uc20_x, uc20_y, 'Analyze Drawdown\n& Risk')
    
    # ============ USE CASES - SIXTH ROW (API & Integration) ============
    uc21_x, uc21_y = 6.5, 4
    uc22_x, uc22_y = 10, 4
    uc23_x, uc23_y = 13.5, 4
    uc24_x, uc24_y = 17, 4
    
    diagram.add_use_case(uc21_x, uc21_y, 'Access REST API\nEndpoints')
    diagram.add_use_case(uc22_x, uc22_y, 'Authenticate with\nAPI Keys')
    diagram.add_use_case(uc23_x, uc23_y, 'Integrate with\nExternal Platforms')
    diagram.add_use_case(uc24_x, uc24_y, 'Stream Real-time\nPredictions')
    
    # ============ USE CASES - BOTTOM ROW (System Management) ============
    uc25_x, uc25_y = 8.5, 2
    uc26_x, uc26_y = 12, 2
    uc27_x, uc27_y = 15.5, 2
    
    diagram.add_use_case(uc25_x, uc25_y, 'Monitor System\nHealth')
    diagram.add_use_case(uc26_x, uc26_y, 'Export Trading\nReports')
    diagram.add_use_case(uc27_x, uc27_y, 'Configure Alert\nPreferences')
    
    # ============ CONNECTIONS - Day Trader ============
    diagram.add_connection(1.2, 16.2, uc1_x - 0.9, uc1_y)
    diagram.add_connection(1.2, 16.2, uc2_x - 0.9, uc2_y)
    diagram.add_connection(1.2, 16.2, uc3_x - 0.9, uc3_y)
    diagram.add_connection(1.2, 16.2, uc7_x - 0.9, uc7_y)
    
    # ============ CONNECTIONS - Swing Trader ============
    diagram.add_connection(1.2, 13.7, uc1_x - 0.9, uc1_y)
    diagram.add_connection(1.2, 13.7, uc5_x - 0.9, uc5_y)
    diagram.add_connection(1.2, 13.7, uc7_x - 0.9, uc7_y)
    diagram.add_connection(1.2, 13.7, uc3_x - 0.9, uc3_y)
    
    # ============ CONNECTIONS - Long-term Investor ============
    diagram.add_connection(1.2, 11.2, uc6_x - 0.9, uc6_y)
    diagram.add_connection(1.2, 11.2, uc9_x - 0.9, uc9_y)
    diagram.add_connection(1.2, 11.2, uc11_x - 0.9, uc11_y)
    
    # ============ CONNECTIONS - Portfolio Manager ============
    diagram.add_connection(1.2, 8.7, uc9_x - 0.9, uc9_y)
    diagram.add_connection(1.2, 8.7, uc10_x - 0.9, uc10_y)
    diagram.add_connection(1.2, 8.7, uc11_x - 0.9, uc11_y)
    diagram.add_connection(1.2, 8.7, uc12_x - 0.9, uc12_y)
    diagram.add_connection(1.2, 8.7, uc26_x - 0.9, uc26_y)
    
    # ============ CONNECTIONS - Data Scientist ============
    diagram.add_connection(1.2, 6.2, uc13_x - 0.9, uc13_y)
    diagram.add_connection(1.2, 6.2, uc14_x - 0.9, uc14_y)
    diagram.add_connection(1.2, 6.2, uc15_x - 0.9, uc15_y)
    diagram.add_connection(1.2, 6.2, uc16_x - 0.9, uc16_y)
    diagram.add_connection(1.2, 6.2, uc17_x - 0.9, uc17_y)
    
    # ============ CONNECTIONS - Risk Analyst ============
    diagram.add_connection(1.2, 3.7, uc10_x - 0.9, uc10_y)
    diagram.add_connection(1.2, 3.7, uc19_x - 0.9, uc19_y)
    diagram.add_connection(1.2, 3.7, uc20_x - 0.9, uc20_y)
    
    # ============ CONNECTIONS - Algorithm Developer ============
    diagram.add_connection(22.8, 16.2, uc21_x + 0.9, uc21_y)
    diagram.add_connection(22.8, 16.2, uc22_x + 0.9, uc22_y)
    diagram.add_connection(22.8, 16.2, uc13_x + 0.9, uc13_y)
    diagram.add_connection(22.8, 16.2, uc15_x + 0.9, uc15_y)
    diagram.add_connection(22.8, 16.2, uc25_x + 0.9, uc25_y)
    
    # ============ CONNECTIONS - Trading Bot ============
    diagram.add_connection(22.8, 13.7, uc4_x + 0.9, uc4_y)
    diagram.add_connection(22.8, 13.7, uc21_x + 0.9, uc21_y)
    diagram.add_connection(22.8, 13.7, uc22_x + 0.9, uc22_y)
    diagram.add_connection(22.8, 13.7, uc24_x + 0.9, uc24_y)
    
    # ============ CONNECTIONS - External System ============
    diagram.add_connection(22.8, 11.2, uc23_x + 0.9, uc23_y)
    diagram.add_connection(22.8, 11.2, uc21_x + 0.9, uc21_y)
    diagram.add_connection(22.8, 11.2, uc22_x + 0.9, uc22_y)
    
    # ============ CONNECTIONS - Financial Advisor ============
    diagram.add_connection(22.8, 8.7, uc11_x + 0.9, uc11_y)
    diagram.add_connection(22.8, 8.7, uc10_x + 0.9, uc10_y)
    diagram.add_connection(22.8, 8.7, uc26_x + 0.9, uc26_y)
    
    diagram.save('use_case_consolidated.png')

# Generate the diagram
print("🎨 Generating Consolidated Use Case Diagram...")
create_consolidated_diagram()
print("✅ Consolidated use case diagram generated successfully!")
print("📁 Location: diagrams/use_case_consolidated.png")
