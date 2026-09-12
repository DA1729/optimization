import numpy as np


def rotation_matrix(angle_rad):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s], [s, c]])


def make_anisotropic_spd(condition_number, angle_rad, smallest_eigenvalue=1.0):
    eigenvalues = np.array([smallest_eigenvalue * condition_number, smallest_eigenvalue])
    rotation = rotation_matrix(angle_rad)
    return rotation @ np.diag(eigenvalues) @ rotation.T


def eigendecomposition(a):
    values, vectors = np.linalg.eigh(a)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    for k in range(vectors.shape[1]):
        if vectors[np.argmax(np.abs(vectors[:, k])), k] < 0:
            vectors[:, k] *= -1
    if np.linalg.det(vectors) < 0:
        vectors[:, -1] *= -1
    return values, vectors


def to_eigenbasis(x, eigenvectors):
    return x @ eigenvectors


def from_eigenbasis(z, eigenvectors):
    return z @ eigenvectors.T


def symmetric_sqrt(a):
    values, vectors = eigendecomposition(a)
    return vectors @ np.diag(np.sqrt(values)) @ vectors.T


def symmetric_inverse_sqrt(a):
    values, vectors = eigendecomposition(a)
    return vectors @ np.diag(1.0 / np.sqrt(values)) @ vectors.T


def whiten(x, a):
    return x @ symmetric_sqrt(a).T


def metric_inner(u, v, metric):
    return u @ metric @ v


def metric_norm(v, metric):
    return np.sqrt(max(metric_inner(v, v, metric), 0.0))


def metric_cosine(u, v, metric):
    denominator = metric_norm(u, metric) * metric_norm(v, metric)
    if denominator == 0.0:
        return np.nan
    return metric_inner(u, v, metric) / denominator


def steepest_descent_direction(gradient_vector, metric):
    return -np.linalg.solve(metric, gradient_vector)


def unit_steepest_descent_direction(gradient_vector, metric):
    direction = steepest_descent_direction(gradient_vector, metric)
    return direction / metric_norm(direction, metric)


def sample_unit_sphere(metric, n_samples=720):
    angles = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)
    directions = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    norms = np.sqrt(np.einsum("ij,jk,ik->i", directions, metric, directions))
    return angles, directions / norms[:, None]


def brute_force_steepest_direction(gradient_vector, metric, n_samples=720):
    angles, directions = sample_unit_sphere(metric, n_samples)
    decrease_rates = -directions @ gradient_vector
    best = int(np.argmax(decrease_rates))
    return directions[best], angles, decrease_rates


def make_metric(name, a):
    if name == "identity":
        return np.eye(a.shape[0])
    if name == "jacobi":
        return np.diag(np.diag(a))
    if name == "hessian":
        return a.copy()
    raise ValueError(f"unknown metric: {name}")


def condition_number(a):
    values, _ = eigendecomposition(a)
    return values[0] / values[-1]
