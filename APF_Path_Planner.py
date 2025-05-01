import numpy as np
import matplotlib.pyplot as plt


def compute_attractive_field(X, Y, goal, k_att):
    d = np.sqrt((X - goal[0]) ** 2 + (Y - goal[1]) ** 2)
    return 0.5 * k_att * d #** 2


def compute_repulsive_field(X, Y, obstacles, k_rep, influence_radius, epsilon=1e-3):
    U_rep = np.zeros_like(X)
    for obs in obstacles:
        dx = X - obs[0]
        dy = Y - obs[1]
        d = np.sqrt(dx ** 2 + dy ** 2)
        d = np.maximum(d, epsilon)

        mask = d < influence_radius # mask selects only the points within the influence radius
        U_rep_local = np.zeros_like(d)
        U_rep_local[mask] = 0.5 * k_rep * ((influence_radius - d[mask]) ** 2)
        # Other formula which explodes as you get close to the obstacle (stronger gradient), starts very weak at edge but gets very strong fast
        #U_rep_local[mask] = 0.5 * k_rep * ((1 / d[mask]) - (1 / influence_radius)) ** 2

        U_rep += U_rep_local

    return U_rep



def compute_total_potential(X, Y, goal, obstacles, k_att, k_rep, influence_radius):
    U_att = compute_attractive_field(X, Y, goal, k_att)
    U_rep = compute_repulsive_field(X, Y, obstacles, k_rep, influence_radius)
    return U_att + U_rep

# APF PARAMETERS
# alpha: Step size in the gradient direction, controls how fast the agent moves
# momentum_beta: Adds momentum so the agent can "push through" flat or high potential areas
# max_iters: Max number of iterations
# threshold: Distance from actual goal where it's considered reached
# k_att: Strength of attraction towards goal
# k_rep: Strength of repulsion from obstacles
# influence_radius: Range around obstacles where repulsive force applies
# margin: Extra space around the environment to include in potential field grid

# POSSIBLE MILESTONES INSTEAD OF DIRECT GOAL
# Instead of just goal
#waypoints = [intermediate_1, intermediate_2, ..., final_goal]
#for wp in waypoints:
#    path_segment = apf(start, wp, ...)
#    path += path_segment

def create_path_using_apf(start, goal, obstacles, alpha=0.3, max_iters=3000, threshold=1.5,
                          k_att=1.0, k_rep=5.0, influence_radius=10.0, momentum_beta=0.8, margin=500):

    all_points = np.vstack((obstacles, start, goal))  # Combine all points so our grid includes everything we need
    min_x = np.min(all_points[:, 0]) - margin
    max_x = np.max(all_points[:, 0]) + margin
    min_y = np.min(all_points[:, 1]) - margin
    max_y = np.max(all_points[:, 1]) + margin

    x_range = np.arange(min_x, max_x, 5.0)
    y_range = np.arange(min_y, max_y, 5.0)
    X, Y = np.meshgrid(x_range, y_range)

    U_total = compute_total_potential(X, Y, goal, obstacles, k_att, k_rep, influence_radius)

    path = [start.copy()]
    pos = start.copy()
    prev_step = np.zeros(2)

    for _ in range(max_iters):
        ix = np.argmin(np.abs(X[0, :] - pos[0]))
        iy = np.argmin(np.abs(Y[:, 0] - pos[1]))

        if ix <= 0 or ix >= X.shape[1] - 1 or iy <= 0 or iy >= Y.shape[0] - 1:
            break

        dU_dx = (U_total[iy, ix + 1] - U_total[iy, ix - 1]) / (X[0, 1] - X[0, 0])
        dU_dy = (U_total[iy + 1, ix] - U_total[iy - 1, ix]) / (Y[1, 0] - Y[0, 0])

        grad = np.array([dU_dx, dU_dy])
        grad_norm = np.linalg.norm(grad) + 1e-6
        step = momentum_beta * prev_step - alpha * grad / grad_norm
        # Local minimum escape: if stuck and far from goal
        if grad_norm < 1e-3 and np.linalg.norm(pos - goal) > threshold:
            step = np.random.uniform(-1, 1, 2) * 5.0
        pos += step
        prev_step = step

        path.append(pos.copy())

        if np.linalg.norm(pos - goal) < threshold:
            break

    return np.array(path), X, Y, U_total

if __name__ == "__main__":
    # Run and plot result
    start = np.array([0.0, 0.0])
    goal = np.array([45.0, 40.0])
    obstacles = np.array([
        [10.0, 10.0],
        [15.0, 20.0],
        [25.0, 25.0],
        [30.0, 15.0]
    ])

    path, X, Y, U_total = create_path_using_apf(
        start, goal, obstacles,
        alpha=0.5, max_iters=3000, threshold=1,
        k_att=1.0, k_rep=15, influence_radius=7, momentum_beta=0.5, margin=10
    )
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(X, Y, U_total, levels=150, cmap='plasma',)
    plt.colorbar(contour, label='Total Potential Value')

    plt.plot(path[:, 0], path[:, 1], 'b.-', label='APF Path')
    plt.plot(start[0], start[1], 'go', markersize=10, label='Start')
    plt.plot(goal[0], goal[1], 'r*', markersize=15, label='Goal')
    for obs in obstacles:
        circle = plt.Circle(obs, 0.5, color='gray', alpha=0.5)
        plt.gca().add_patch(circle)

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Artificial Potential Field Path Planning – Integrated Visualization")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()
