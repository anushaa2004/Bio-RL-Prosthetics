import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_twin_critic_diagram():
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # --- 1. THE TWIN CRITICS ---
    # Critic 1 Box (Optimist)
    critic1 = patches.FancyBboxPatch((0.1, 0.6), 0.25, 0.2, boxstyle="round,pad=0.05", 
                                     fc='#e6f2ff', ec='#3366cc', lw=2)
    ax.add_patch(critic1)
    ax.text(0.225, 0.72, "Critic 1\n($Q_{\\theta_1}$)", ha='center', va='center', fontsize=14, fontweight='bold', color='#3366cc')
    ax.text(0.225, 0.65, "Estimate: 100\n(Optimistic)", ha='center', va='center', fontsize=10, style='italic')

    # Critic 2 Box (Pessimist)
    critic2 = patches.FancyBboxPatch((0.1, 0.2), 0.25, 0.2, boxstyle="round,pad=0.05", 
                                     fc='#ffe6e6', ec='#cc3333', lw=2)
    ax.add_patch(critic2)
    ax.text(0.225, 0.32, "Critic 2\n($Q_{\\theta_2}$)", ha='center', va='center', fontsize=14, fontweight='bold', color='#cc3333')
    ax.text(0.225, 0.25, "Estimate: 80\n(Realistic)", ha='center', va='center', fontsize=10, style='italic')

    # --- 2. THE SAFETY BRAKE (MIN Operator) ---
    min_box = patches.FancyBboxPatch((0.5, 0.4), 0.2, 0.2, boxstyle="round,pad=0.05", 
                                     fc='#fff2cc', ec='#d6b656', lw=2)
    ax.add_patch(min_box)
    ax.text(0.6, 0.53, "MIN( )", ha='center', va='center', fontsize=16, fontweight='bold', color='#997a00')
    ax.text(0.6, 0.47, "Safety Brake", ha='center', va='center', fontsize=10, style='italic')

    # --- 3. THE RESULT (Target Value) ---
    # Target Box
    target_box = patches.FancyBboxPatch((0.8, 0.45), 0.15, 0.1, boxstyle="round,pad=0.02", 
                                        fc='white', ec='green', lw=2)
    ax.add_patch(target_box)
    ax.text(0.875, 0.5, "y = 80", ha='center', va='center', fontsize=14, fontweight='bold', color='green')

    # --- 4. ARROWS ---
    # From Critic 1 to Min
    ax.arrow(0.35, 0.7, 0.15, -0.15, head_width=0.03, head_length=0.03, fc='k', ec='k', length_includes_head=True)
    # From Critic 2 to Min
    ax.arrow(0.35, 0.3, 0.15, 0.15, head_width=0.03, head_length=0.03, fc='k', ec='k', length_includes_head=True)
    # From Min to Target
    ax.arrow(0.7, 0.5, 0.1, 0, head_width=0.03, head_length=0.03, fc='g', ec='g', length_includes_head=True)

    # --- 5. LABELS ---
    ax.text(0.6, 0.35, "Takes the\nLower Value", ha='center', fontsize=10, color='#997a00', style='italic')
    ax.set_title("TD3 'Twin Critic' Logic: Censoring Overestimation", fontsize=16, pad=20)

    # Setup layout
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    plt.tight_layout()
    plt.show()

draw_twin_critic_diagram()
