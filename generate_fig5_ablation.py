import matplotlib.pyplot as plt
import numpy as np

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

components = ['Full System\n(Tri-Signal)', 'w/o ML\nSignal', 'w/o Pattern\nRecognition', 'w/o News\nIntelligence']
accuracies = [85.2, 68.4, 79.1, 82.8]
colors = ['#2ecc71', '#e74c3c', '#f39c12', '#3498db']

bars1 = ax1.bar(components, accuracies, color=colors, edgecolor='black', linewidth=1.2)
ax1.axhline(y=85.2, color='#2ecc71', linestyle='--', alpha=0.7, label='Full System (85.2%)')
ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax1.set_title('(a) Network Ablation', fontsize=14, fontweight='bold')
ax1.set_ylim(0, 100)
ax1.legend(loc='lower right')

for bar, acc in zip(bars1, accuracies):
    height = bar.get_height()
    ax1.annotate(f'{acc}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=11, fontweight='bold')

features = ['RSI (14)', 'MACD\nHistogram', 'Volume\nChange %', '50-day\nSMA Cross', 'Bollinger\n%B']
shap_values = [0.142, 0.128, 0.115, 0.098, 0.087]
feature_colors = ['#e74c3c', '#e67e22', '#f1c40f', '#3498db', '#9b59b6']

bars2 = ax2.barh(features, shap_values, color=feature_colors, edgecolor='black', linewidth=1.2)
ax2.set_xlabel('Mean SHAP Value', fontsize=12, fontweight='bold')
ax2.set_title('(b) Feature Importance (SHAP Analysis)', fontsize=14, fontweight='bold')
ax2.set_xlim(0, 0.18)

for bar, val in zip(bars2, shap_values):
    width = bar.get_width()
    ax2.annotate(f'{val:.3f}',
                xy=(width, bar.get_y() + bar.get_height() / 2),
                xytext=(3, 0),
                textcoords="offset points",
                ha='left', va='center', fontsize=11, fontweight='bold')

ax2.invert_yaxis()

plt.tight_layout()
plt.savefig('docs/figures/fig5.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("fig5.png generated successfully!")
