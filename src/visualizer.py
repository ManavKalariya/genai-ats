import matplotlib.pyplot as plt
import numpy as np

def plot_match_score(score: float):
    """
    Visualize match score as a gauge (0–100)
    """
    fig, ax = plt.subplots(figsize=(6, 3), subplot_kw={'projection': 'polar'})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 100)

    # Remove labels and ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Draw gauge arc
    theta = np.linspace(0, np.pi, 100)
    radii = np.ones(100) * 100
    colors = plt.cm.RdYlGn(np.linspace(0, 1, 100))
    ax.bar(theta, radii, width=np.pi/100, bottom=0, color=colors, alpha=0.8)

    # Needle
    angle = (1 - score/100) * np.pi
    ax.arrow(angle, 0, 0, 70,
             width=0.03, head_width=0.1, head_length=10,
             fc='black', ec='black')

    # Center text
    ax.text(0, -10, f"Match: {score:.2f}%", ha='center', va='center',
            fontsize=14, fontweight='bold')

    plt.show()
