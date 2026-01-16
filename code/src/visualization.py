import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from .config import OUTPUT_DIR, PLOT_STYLE, ACADEMIC_COLORS

def set_style():
    """Sets the global style for plots."""
    plt.style.use(PLOT_STYLE)
    
def plot_jcurve(model, df):
    """Generates the J-Curve plot with 95% CI - Academic Style."""
    set_style()
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    gamma1 = model.params['TechIntensity']
    gamma2 = model.params['Tech_Sq']
    const = model.params['const']
    min_point = -gamma1 / (2 * gamma2)
    
    x = np.linspace(0, 0.40, 200)
    y = const + gamma1 * x + gamma2 * x**2
    
    # Confidence interval - academic gray
    se1, se2 = model.bse['TechIntensity'], model.bse['Tech_Sq']
    y_upper = const + (gamma1 + 1.96*se1) * x + (gamma2 - 1.96*se2) * x**2
    y_lower = const + (gamma1 - 1.96*se1) * x + (gamma2 + 1.96*se2) * x**2
    
    # Academic palette: dark gray for curve, light gray for CI
    ax.fill_between(x, y_lower, y_upper, alpha=0.25, color='#4a4a4a', label='95% Confidence Interval')
    ax.plot(x, y, color='#1a1a1a', linewidth=2.5, label='Estimated J-Curve')
    
    y_min = const + gamma1 * min_point + gamma2 * min_point**2
    # Dark academic red for minimum
    ax.axvline(x=min_point, color='#8B0000', linestyle='--', linewidth=1.5, alpha=0.8, 
               label=f'Turning Point ({min_point:.1%})')
    ax.scatter([min_point], [y_min], color='#8B0000', s=80, zorder=5, edgecolors='white', linewidths=1.5)
    
    mean_tech = df['TechIntensity'].mean()
    # Navy blue for the mean
    ax.axvline(x=mean_tech, color='#000080', linestyle=':', linewidth=1.5, alpha=0.8, 
               label=f'Sample Mean ({mean_tech:.1%})')
    
    ax.set_xlabel('Intangible Asset Intensity', fontsize=12, fontweight='medium')
    ax.set_ylabel('Estimated TFP (ω̂)', fontsize=12, fontweight='medium')
    ax.set_title('The Productivity J-Curve for Italian SMEs\n(Balanced Panel, Fixed Effects Estimation)', 
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10, framealpha=0.95, edgecolor='#cccccc')
    ax.set_xlim(0, 0.40)
    ax.tick_params(axis='both', labelsize=10)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Annotation box with academic style
    ax.annotate(f'γ₁ = {gamma1:.3f}\nγ₂ = {gamma2:.3f}', 
                xy=(0.30, y.min() + 0.02), fontsize=11,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#999999', alpha=0.95))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig_jcurve.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved: {OUTPUT_DIR / 'fig_jcurve.png'}")


def plot_geo_boxplots(df):
    """TFP and Labor Productivity Boxplots by geographic area - Academic Style (Binary: Center-North vs South)."""
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.patch.set_facecolor('white')
    
    # Create binary region classification
    df = df.copy()
    df['Region'] = df['MacroArea'].apply(lambda x: 'South' if x == 'Sud' else 'Center-North')
    
    order = ['Center-North', 'South']
    palette = {'Center-North': ACADEMIC_COLORS[0], 'South': ACADEMIC_COLORS[2]}
    
    # Plot 1: TFP
    ax1 = axes[0]
    ax1.set_facecolor('white')
    sns.boxplot(data=df, x='Region', y='TFP', order=order, palette=palette, ax=ax1,
                      linewidth=1.5, flierprops={'marker': 'o', 'markersize': 3, 'alpha': 0.5})
    ax1.set_xlabel('Geographic Area', fontsize=12, fontweight='medium')
    ax1.set_ylabel('Total Factor Productivity (TFP)', fontsize=12, fontweight='medium')
    ax1.set_title('TFP Distribution by Region', fontsize=13, fontweight='bold')
    ax1.tick_params(axis='both', labelsize=10)
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, axis='y')
    
    # Plot 2: Labor Productivity
    ax2 = axes[1]
    ax2.set_facecolor('white')
    sns.boxplot(data=df, x='Region', y='LaborProd', order=order, palette=palette, ax=ax2,
                      linewidth=1.5, flierprops={'marker': 'o', 'markersize': 3, 'alpha': 0.5})
    ax2.set_xlabel('Geographic Area', fontsize=12, fontweight='medium')
    ax2.set_ylabel('Labor Productivity (Y/L)', fontsize=12, fontweight='medium')
    ax2.set_title('Labor Productivity Distribution by Region', fontsize=13, fontweight='bold')
    ax2.tick_params(axis='both', labelsize=10)
    ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, axis='y')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig_geo_boxplots.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved: {OUTPUT_DIR / 'fig_geo_boxplots.png'}")


def plot_sector_analysis(df):
    """Sector Analysis (ATECO) - Academic Style."""
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('white')
    
    # Desired sector order
    desired_order = ['Computer & Electronics', 'Electrical Equipment', 'Machinery']
    
    # Calculate aggregate statistics
    sector_stats = df.groupby('Sector').agg({
        'TechIntensity': 'mean',
        'TFP': 'mean',
        'firm_id': 'nunique'
    }).rename(columns={'firm_id': 'N_firms'})
    
    # Filter and sort only present sectors
    valid_order = [s for s in desired_order if s in sector_stats.index]
    sector_stats = sector_stats.loc[valid_order]
    
    print("\n--- Sector Stats (ATECO) ---")
    print(sector_stats)
    
    colors = ACADEMIC_COLORS
    
    # Plot 1: Tech Intensity Bar
    ax1 = axes[0]
    ax1.set_facecolor('white')
    bars1 = ax1.bar(sector_stats.index, sector_stats['TechIntensity'] * 100, 
                    color=colors[:len(valid_order)], edgecolor='#1a1a1a', linewidth=1.2)
    ax1.set_xlabel('ATECO Sector', fontsize=12, fontweight='medium')
    ax1.set_ylabel('Mean Technological Intensity (%)', fontsize=12, fontweight='medium')
    ax1.set_title('Technological Intensity by Sector', fontsize=13, fontweight='bold')
    ax1.tick_params(axis='x', labelsize=10, rotation=15)
    ax1.tick_params(axis='y', labelsize=10)
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, axis='y')
    ax1.set_ylim(0, max(sector_stats['TechIntensity'] * 100) * 1.2)
    
    for bar, val in zip(bars1, sector_stats['TechIntensity'] * 100):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='medium')
    
    # Plot 2: TFP Bar
    ax2 = axes[1]
    ax2.set_facecolor('white')
    bars2 = ax2.bar(sector_stats.index, sector_stats['TFP'], 
                    color=colors[:len(valid_order)], edgecolor='#1a1a1a', linewidth=1.2)
    ax2.set_xlabel('ATECO Sector', fontsize=12, fontweight='medium')
    ax2.set_ylabel('Mean TFP', fontsize=12, fontweight='medium')
    ax2.set_title('Total Factor Productivity by Sector', fontsize=13, fontweight='bold')
    ax2.tick_params(axis='x', labelsize=10, rotation=15)
    ax2.tick_params(axis='y', labelsize=10)
    ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, axis='y')
    # TFP range zoom
    tfp_min = sector_stats['TFP'].min() * 0.95
    tfp_max = sector_stats['TFP'].max() * 1.05
    ax2.set_ylim(tfp_min, tfp_max)
    
    for bar, val in zip(bars2, sector_stats['TFP']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, 
                f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='medium')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig_sector_bar.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved: {OUTPUT_DIR / 'fig_sector_bar.png'}")
