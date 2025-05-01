import numpy as np


def debug_plot_apf_field(start, goal, obstacles, k_att=1.0, k_rep=5.0, influence_radius=10.0, margin=500):
    from APF_Path_Planner import compute_total_potential
    import matplotlib.pyplot as plt

    min_x = min(start[0], goal[0]) - margin
    max_x = max(start[0], goal[0]) + margin
    min_y = min(start[1], goal[1]) - margin
    max_y = max(start[1], goal[1]) + margin

    x_range = np.arange(min_x, max_x, 5.0)
    y_range = np.arange(min_y, max_y, 5.0)
    X, Y = np.meshgrid(x_range, y_range)

    U_total = compute_total_potential(X, Y, goal, obstacles, k_att, k_rep, influence_radius)

    plt.figure(figsize=(10, 8))
    contour = plt.contourf(X, Y, U_total, levels=200, cmap='plasma')
    plt.colorbar(contour, label='Total Potential Value')
    plt.plot(start[0], start[1], 'go', markersize=10, label='Start')
    plt.plot(goal[0], goal[1], 'r*', markersize=15, label='Goal')

    for obs in obstacles:
        circle = plt.Circle(obs, 20, color='gray', alpha=0.3)
        plt.gca().add_patch(circle)

    plt.title("APF Repulsive Field from Obstacle Wall")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()
