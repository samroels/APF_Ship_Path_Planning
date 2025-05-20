import numpy as np
import matplotlib.pyplot as plt


def attractive_potential(X, Y, goal, k_att):
    d = np.sqrt((X - goal[0]) ** 2 + (Y - goal[1]) ** 2)
    return 0.5 * k_att * d


def repulsive_potential(X, Y, obstacles, k_rep, influence_radius, epsilon=1e-3):
    U_rep = np.zeros_like(X)
    for obs in obstacles:
        dx = X - obs[0]
        dy = Y - obs[1]
        d = np.sqrt(dx ** 2 + dy ** 2)
        d = np.maximum(d, epsilon)

        mask = d < influence_radius # mask selects only the points within the influence radius
        U_rep_local = np.zeros_like(d)
        U_rep_local[mask] = 0.5 * k_rep * ((influence_radius - d[mask]) ** 2)
        U_rep += U_rep_local

    return U_rep



def total_potential(X, Y, goal, obstacles, k_att, k_rep, influence_radius):
    U_att = attractive_potential(X, Y, goal, k_att)
    U_rep = repulsive_potential(X, Y, obstacles, k_rep, influence_radius)
    return U_att + U_rep

# APF PARAMETERS
# alpha: Step size in the gradient direction, controls how fast the agent moves
# momentum_beta: Adds momentum so the agent can "push through" flat or high potential areas
# max_iters: Max number of iterations for path planning loop
# threshold: Distance from actual goal where it's considered reached
# k_att: Strength of attraction towards goal
# k_rep: Strength of repulsion from obstacles
# influence_radius: Range around obstacles where repulsive force applies
# margin: Extra space around the environment to include in potential field grid


def create_path_using_apf(start, goal, obstacles, alpha=0.3, max_iters=3000, threshold=1.5,
                          k_att=1.0, k_rep=5.0, influence_radius=10.0, momentum_beta=0.8, margin=500):

    # Set up grid bounds
    all_points = np.vstack((obstacles, start, goal))  # Combine all points so our grid includes everything we need
    min_x = np.min(all_points[:, 0]) - margin
    max_x = np.max(all_points[:, 0]) + margin
    min_y = np.min(all_points[:, 1]) - margin
    max_y = np.max(all_points[:, 1]) + margin

    # Create grid
    x_range = np.arange(min_x, max_x, 5.0)
    y_range = np.arange(min_y, max_y, 5.0)
    X, Y = np.meshgrid(x_range, y_range)

    U_total = total_potential(X, Y, goal, obstacles, k_att, k_rep, influence_radius)

    path = [start.copy()]
    pos = start.copy()
    prev_step = np.zeros(2)

    # Path planning loop limited to max_iterations
    for _ in range(max_iters):
        # Current grid cell index for the agent's position
        ix = np.argmin(np.abs(X[0, :] - pos[0]))
        iy = np.argmin(np.abs(Y[:, 0] - pos[1]))

        # Compute gradient with central difference approximation
        dU_dx = (U_total[iy, ix + 1] - U_total[iy, ix - 1]) / (X[0, 1] - X[0, 0])
        dU_dy = (U_total[iy + 1, ix] - U_total[iy - 1, ix]) / (Y[1, 0] - Y[0, 0])

        # Normalize gradient to get step direction
        grad = np.array([dU_dx, dU_dy])
        grad_norm = np.linalg.norm(grad) + 1e-6
        # added momentum to smoothen path and escape local minima
        # use (- alpha...) because we want to move towards negative gradient
        step = momentum_beta * prev_step - alpha * grad / grad_norm

        # Update position and add to path
        pos += step
        prev_step = step
        path.append(pos.copy())

        # Break if goal is reached
        if np.linalg.norm(pos - goal) < threshold:
            break
    # Return path and potential field grid
    return np.array(path), X, Y, U_total
