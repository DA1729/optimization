import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geometry import (
    brute_force_steepest_direction,
    condition_number,
    make_metric,
    metric_cosine,
    symmetric_sqrt,
    unit_steepest_descent_direction,
)
from optimization import (
    descent_trajectory,
    gradient,
    make_problem,
    newton_direction,
    newton_trajectory,
    objective,
    preconditioned_trajectory_via_change_of_variables,
    stable_step_size,
    step_quality,
)

STEP_FRACTION = 0.9
N_ITERATIONS = 40
X0 = np.array([-2.0, 3.0])


@pytest.fixture(params=[(25.0, 30.0), (4.0, 0.0), (120.0, 63.0)])
def setup(request):
    condition, angle_deg = request.param
    problem = make_problem(condition, np.radians(angle_deg))
    step_size = stable_step_size(problem["eigenvalues"], STEP_FRACTION)
    iterates = descent_trajectory(X0, problem["a"], problem["b"], step_size, N_ITERATIONS)
    return problem, step_size, iterates


def test_eigendecomposition_reconstructs_a(setup):
    problem, _, _ = setup
    q, values = problem["eigenvectors"], problem["eigenvalues"]
    assert np.allclose(q @ np.diag(values) @ q.T, problem["a"])
    assert np.allclose(q.T @ q, np.eye(2))


def test_trajectory_is_identical_in_the_eigenbasis(setup):
    problem, _, iterates = setup
    q = problem["eigenvectors"]
    z_iterates = iterates @ q
    assert np.allclose(iterates, z_iterates @ q.T, atol=1e-12)
    assert np.allclose(z_iterates, iterates @ q, atol=1e-12)


def test_eigenbasis_iterates_satisfy_the_decoupled_recursion(setup):
    problem, step_size, iterates = setup
    q, values = problem["eigenvectors"], problem["eigenvalues"]
    errors = (iterates - problem["x_star"]) @ q
    assert np.allclose(errors[1:], errors[:-1] * (1.0 - step_size * values), atol=1e-9)


def test_objective_values_agree_across_bases(setup):
    problem, _, iterates = setup
    a, b, q = problem["a"], problem["b"], problem["eigenvectors"]
    root = symmetric_sqrt(a)
    values_x = objective(iterates, a, b)
    values_z = objective(iterates @ q, np.diag(problem["eigenvalues"]), q.T @ b)
    values_y = objective(iterates @ root.T, np.eye(2), np.linalg.solve(root, b))
    assert np.allclose(values_x, values_z, atol=1e-9)
    assert np.allclose(values_x, values_y, atol=1e-9)


def test_orthogonal_change_preserves_lengths_and_angles(setup):
    problem, _, iterates = setup
    a, b, q = problem["a"], problem["b"], problem["eigenvectors"]
    steps_x = np.diff(iterates, axis=0)
    steps_z = np.diff(iterates @ q, axis=0)
    assert np.allclose(np.linalg.norm(steps_x, axis=1), np.linalg.norm(steps_z, axis=1))
    for t in range(0, len(steps_x), 5):
        newton = newton_direction(iterates[t], a, b)
        assert np.isclose(metric_cosine(steps_x[t], newton, np.eye(2)),
                          metric_cosine(steps_z[t], q.T @ newton, np.eye(2)))


def test_condition_number_is_basis_independent(setup):
    problem, _, _ = setup
    assert np.isclose(condition_number(problem["a"]),
                      condition_number(np.diag(problem["eigenvalues"])))


def test_gradient_descent_step_is_euclidean_steepest(setup):
    problem, _, iterates = setup
    quality = step_quality(iterates, problem["a"], problem["b"])
    assert np.allclose(quality["cosine_euclidean_negative_gradient"], 1.0, atol=1e-9)


def test_same_step_scores_worse_under_the_hessian_metric(setup):
    problem, _, iterates = setup
    quality = step_quality(iterates, problem["a"], problem["b"])
    if problem["condition_number"] > 1.5:
        assert quality["cosine_hessian_newton"].mean() < 0.999


def test_decrease_fraction_matches_the_a_metric_identity(setup):
    problem, _, iterates = setup
    a, b = problem["a"], problem["b"]
    quality = step_quality(iterates, a, b)
    steps = np.diff(iterates, axis=0)
    for t in range(len(steps)):
        newton = newton_direction(iterates[t], a, b)
        residual = steps[t] - newton
        predicted = 1.0 - (residual @ a @ residual) / (newton @ a @ newton)
        assert np.isclose(quality["decrease_fraction_of_best_single_step"][t], predicted, atol=1e-8)


def test_objective_decreases_monotonically(setup):
    problem, _, iterates = setup
    values = objective(iterates, problem["a"], problem["b"])
    assert np.all(np.diff(values) < 0.0)


def test_metric_cosine_differs_from_euclidean_cosine(setup):
    problem, _, iterates = setup
    a, b = problem["a"], problem["b"]
    if problem["condition_number"] <= 1.5:
        pytest.skip("isotropic problem")
    step = np.diff(iterates, axis=0)[0]
    newton = newton_direction(iterates[0], a, b)
    assert not np.isclose(metric_cosine(step, newton, np.eye(2)), metric_cosine(step, newton, a))


def test_metric_steepest_direction_beats_sampled_directions(setup):
    problem, _, iterates = setup
    a, b = problem["a"], problem["b"]
    for name in ("identity", "jacobi", "hessian"):
        metric = make_metric(name, a)
        g = gradient(iterates[3], a, b)
        direction = unit_steepest_descent_direction(g, metric)
        _, _, rates = brute_force_steepest_direction(g, metric, n_samples=4000)
        assert -direction @ g >= rates.max() - 1e-6


def test_newton_is_steepest_under_the_hessian_metric(setup):
    problem, _, iterates = setup
    a, b = problem["a"], problem["b"]
    newton = newton_direction(iterates[0], a, b)
    direction = unit_steepest_descent_direction(gradient(iterates[0], a, b), a)
    assert np.isclose(metric_cosine(newton, direction, a), 1.0)


def test_newton_converges_in_one_step(setup):
    problem, _, _ = setup
    iterates = newton_trajectory(X0, problem["a"], problem["b"], 3)
    assert np.allclose(iterates[1], problem["x_star"], atol=1e-10)


def test_preconditioning_equals_gradient_descent_after_change_of_variables(setup):
    problem, step_size, _ = setup
    a, b = problem["a"], problem["b"]
    for name in ("identity", "jacobi", "hessian"):
        metric = make_metric(name, a)
        direct = descent_trajectory(X0, a, b, step_size, N_ITERATIONS, metric=metric)
        mapped, _, transformed_a = preconditioned_trajectory_via_change_of_variables(
            X0, a, b, step_size, N_ITERATIONS, metric)
        assert np.allclose(direct, mapped, atol=1e-8)
        assert condition_number(transformed_a) <= condition_number(a) + 1e-9


def test_preconditioning_is_not_a_change_of_basis(setup):
    problem, step_size, iterates = setup
    if problem["condition_number"] <= 1.5:
        pytest.skip("isotropic problem")
    preconditioned = descent_trajectory(X0, problem["a"], problem["b"], step_size, N_ITERATIONS,
                                        metric=make_metric("hessian", problem["a"]))
    assert not np.allclose(objective(iterates, problem["a"], problem["b"]),
                           objective(preconditioned, problem["a"], problem["b"]))
