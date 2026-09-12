import numpy as np

from geometry import (
    eigendecomposition,
    make_anisotropic_spd,
    metric_cosine,
    steepest_descent_direction,
    symmetric_inverse_sqrt,
    symmetric_sqrt,
)


def make_problem(condition_number, angle_rad, minimizer=(1.0, 1.0)):
    a = make_anisotropic_spd(condition_number, angle_rad)
    x_star = np.asarray(minimizer, dtype=float)
    b = a @ x_star
    eigenvalues, eigenvectors = eigendecomposition(a)
    return {
        "a": a,
        "b": b,
        "x_star": x_star,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "angle_rad": angle_rad,
        "condition_number": eigenvalues[0] / eigenvalues[-1],
    }


def objective(x, a, b):
    x = np.atleast_2d(x)
    values = 0.5 * np.einsum("ij,jk,ik->i", x, a, x) - x @ b
    return values[0] if values.shape[0] == 1 else values


def gradient(x, a, b):
    return a @ x - b


def newton_direction(x, a, b):
    return -np.linalg.solve(a, gradient(x, a, b))


def exact_line_search_size(x, direction, a, b):
    g = gradient(x, a, b)
    curvature = direction @ a @ direction
    return -(g @ direction) / curvature


def descent_trajectory(x0, a, b, step_size, n_iterations, metric=None):
    metric = np.eye(a.shape[0]) if metric is None else metric
    iterates = np.empty((n_iterations + 1, a.shape[0]))
    iterates[0] = np.asarray(x0, dtype=float)
    for t in range(n_iterations):
        direction = steepest_descent_direction(gradient(iterates[t], a, b), metric)
        iterates[t + 1] = iterates[t] + step_size * direction
    return iterates


def preconditioned_trajectory_via_change_of_variables(x0, a, b, step_size, n_iterations, metric):
    root = symmetric_sqrt(metric)
    inverse_root = symmetric_inverse_sqrt(metric)
    transformed_a = inverse_root @ a @ inverse_root
    transformed_b = inverse_root @ b
    y_iterates = descent_trajectory(root @ np.asarray(x0, dtype=float), transformed_a,
                                    transformed_b, step_size, n_iterations)
    return y_iterates @ inverse_root.T, y_iterates, transformed_a


def newton_trajectory(x0, a, b, n_iterations):
    iterates = np.empty((n_iterations + 1, a.shape[0]))
    iterates[0] = np.asarray(x0, dtype=float)
    for t in range(n_iterations):
        iterates[t + 1] = iterates[t] + newton_direction(iterates[t], a, b)
    return iterates


def stable_step_size(eigenvalues, fraction):
    return fraction * 2.0 / eigenvalues[0]


def metric_optimal_step_size(a, metric):
    inverse_root = symmetric_inverse_sqrt(metric)
    values, _ = eigendecomposition(inverse_root @ a @ inverse_root)
    return 2.0 / (values[0] + values[-1])


def objective_history(iterates, a, b):
    return objective(iterates, a, b)


def step_quality(iterates, a, b):
    steps = np.diff(iterates, axis=0)
    n_steps = steps.shape[0]
    quality = {key: np.empty(n_steps) for key in (
        "cosine_euclidean_negative_gradient",
        "cosine_euclidean_newton",
        "cosine_hessian_newton",
        "decrease_fraction_of_best_single_step",
        "line_search_efficiency",
    )}
    for t in range(n_steps):
        x, s = iterates[t], steps[t]
        g = gradient(x, a, b)
        newton = newton_direction(x, a, b)
        quality["cosine_euclidean_negative_gradient"][t] = metric_cosine(s, -g, np.eye(a.shape[0]))
        quality["cosine_euclidean_newton"][t] = metric_cosine(s, newton, np.eye(a.shape[0]))
        quality["cosine_hessian_newton"][t] = metric_cosine(s, newton, a)
        best_decrease = 0.5 * newton @ a @ newton
        quality["decrease_fraction_of_best_single_step"][t] = (
            objective(x, a, b) - objective(x + s, a, b)) / best_decrease
        optimal_size = exact_line_search_size(x, s, a, b)
        best_along_line = objective(x, a, b) - objective(x + optimal_size * s, a, b)
        quality["line_search_efficiency"][t] = (
            objective(x, a, b) - objective(x + s, a, b)) / best_along_line
    return quality


def transformed_quadratic(a, b, transform):
    inverse = np.linalg.inv(transform)
    return inverse.T @ a @ inverse, inverse.T @ b
