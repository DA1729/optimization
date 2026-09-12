import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

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
    metric_optimal_step_size,
    stable_step_size,
    step_quality,
)
from visualization import (
    plot_metric_steepest_descent,
    plot_objective_history,
    plot_preconditioning_comparison,
    plot_step_quality,
    plot_trajectory_bases,
)

TOLERANCE = 1e-10


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description="How a change of basis, a change of metric and preconditioning "
                    "change the reading of one optimization trajectory.")
    parser.add_argument("--condition-number", type=float, default=25.0)
    parser.add_argument("--angle-deg", type=float, default=30.0,
                        help="rotation between the standard basis and the eigenbasis of A")
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--x0", type=float, nargs=2, default=(-2.0, 3.0))
    parser.add_argument("--minimizer", type=float, nargs=2, default=(1.0, 1.0))
    parser.add_argument("--lr-fraction", type=float, default=0.9,
                        help="step size as a fraction of the stability limit 2/lambda_max")
    parser.add_argument("--lr", type=float, default=None, help="explicit step size, overrides --lr-fraction")
    parser.add_argument("--metric", choices=("identity", "jacobi", "hessian"), default="hessian",
                        help="metric M used for the steepest-descent and preconditioning experiment")
    parser.add_argument("--arrows", type=int, default=5, help="number of update vectors drawn")
    parser.add_argument("--outdir", type=Path, default=Path(__file__).resolve().parent / "figures")
    parser.add_argument("--no-figures", action="store_true")
    return parser.parse_args(argv)


def report(name, passed, detail):
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name:52s} {detail}")
    return passed


def check_coordinate_change_invariances(problem, iterates):
    a, b = problem["a"], problem["b"]
    q = problem["eigenvectors"]
    z_iterates = iterates @ q
    y_iterates = iterates @ symmetric_sqrt(a).T

    print("\ninvariances under the orthogonal change of basis z = Q^T x")
    ok = True
    ok &= report("x_t == Q z_t at every iteration", np.allclose(iterates, z_iterates @ q.T, atol=TOLERANCE),
                 f"max |x_t - Q z_t| = {np.abs(iterates - z_iterates @ q.T).max():.3e}")

    values_x = objective(iterates, a, b)
    values_z = objective(z_iterates, np.diag(problem["eigenvalues"]), q.T @ b)
    values_y = objective(y_iterates, np.eye(2), np.linalg.solve(symmetric_sqrt(a), b))
    ok &= report("objective values agree in all three bases",
                 np.allclose(values_x, values_z, atol=1e-9) and np.allclose(values_x, values_y, atol=1e-9),
                 f"max discrepancy = {max(np.abs(values_x - values_z).max(), np.abs(values_x - values_y).max()):.3e}")

    steps_x, steps_z = np.diff(iterates, axis=0), np.diff(z_iterates, axis=0)
    ok &= report("Euclidean step lengths unchanged",
                 np.allclose(np.linalg.norm(steps_x, axis=1), np.linalg.norm(steps_z, axis=1), atol=TOLERANCE),
                 f"max |‖s_t^x‖ - ‖s_t^z‖| = "
                 f"{np.abs(np.linalg.norm(steps_x, axis=1) - np.linalg.norm(steps_z, axis=1)).max():.3e}")

    cos_x = np.array([metric_cosine(steps_x[t], newton_direction(iterates[t], a, b), np.eye(2))
                      for t in range(len(steps_x))])
    cos_z = np.array([metric_cosine(steps_z[t], -np.linalg.solve(np.diag(problem["eigenvalues"]),
                                                                 gradient(z_iterates[t], np.diag(problem["eigenvalues"]), q.T @ b)),
                                    np.eye(2)) for t in range(len(steps_z))])
    ok &= report("Euclidean angles to the Newton direction unchanged",
                 np.allclose(cos_x, cos_z, atol=1e-9), f"max difference = {np.abs(cos_x - cos_z).max():.3e}")

    ok &= report("eigenvalues and condition number unchanged",
                 np.isclose(condition_number(a), condition_number(np.diag(problem["eigenvalues"]))),
                 f"kappa = {condition_number(a):.4f}")

    decreases = np.diff(values_x)
    ok &= report("objective decreases monotonically", np.all(decreases < 0),
                 f"largest increase = {decreases.max():.3e}")
    return ok, z_iterates, values_x, values_z


def check_metric_dependent_quantities(problem, iterates, metric, metric_name):
    a, b = problem["a"], problem["b"]
    q = problem["eigenvectors"]
    steps = np.diff(iterates, axis=0)

    print(f"\nquantities that depend on the metric M = {metric_name} (a metric change is NOT a basis change)")
    ok = True
    euclidean = np.array([metric_cosine(steps[t], newton_direction(iterates[t], a, b), np.eye(2))
                          for t in range(len(steps))])
    under_metric = np.array([metric_cosine(steps[t], newton_direction(iterates[t], a, b), metric)
                             for t in range(len(steps))])
    ok &= report("alignment score changes when M changes",
                 metric_name == "identity" or not np.allclose(euclidean, under_metric, atol=1e-6),
                 f"mean cos_I = {euclidean.mean():.4f} vs mean cos_M = {under_metric.mean():.4f}")

    x = iterates[len(iterates) // 3]
    g = gradient(x, a, b)
    direction_metric = unit_steepest_descent_direction(g, metric)
    sampled, _, rates = brute_force_steepest_direction(g, metric)
    ok &= report("-M^-1 grad f is the steepest direction on the M-unit sphere",
                 np.isclose(-direction_metric @ g, rates.max(), rtol=1e-3),
                 f"analytic {-direction_metric @ g:.6f} vs sampled {rates.max():.6f}")

    direction_euclidean = unit_steepest_descent_direction(g, np.eye(2))
    cosine_between = float(direction_metric @ direction_euclidean
                           / np.linalg.norm(direction_metric) / np.linalg.norm(direction_euclidean))
    ok &= report("the M-steepest direction is not Euclidean-steepest",
                 metric_name == "identity" or cosine_between < 1 - 1e-6,
                 f"cos_I(d_M, d_I) = {cosine_between:.4f}")

    ok &= report("M-steepest direction is basis-covariant (computed in the eigenbasis)",
                 np.allclose(q.T @ direction_metric,
                             unit_steepest_descent_direction(q.T @ g, q.T @ metric @ q), atol=1e-9),
                 "d_M expressed in z-coordinates equals Q^T d_M")
    return ok


def check_preconditioning(problem, x0, step_size, n_iterations, metric, metric_name):
    a, b = problem["a"], problem["b"]
    print(f"\npreconditioning with M = {metric_name}: a genuinely different trajectory")
    ok = True
    direct = descent_trajectory(x0, a, b, step_size, n_iterations, metric=metric)
    mapped, transformed, transformed_a = preconditioned_trajectory_via_change_of_variables(
        x0, a, b, step_size, n_iterations, metric)
    ok &= report("M-steepest descent == Euclidean GD in y = M^(1/2)x",
                 np.allclose(direct, mapped, atol=1e-8),
                 f"max |x_t - M^(-1/2) y_t| = {np.abs(direct - mapped).max():.3e}")
    ok &= report("preconditioned problem is better conditioned",
                 condition_number(transformed_a) <= condition_number(a) + 1e-9,
                 f"kappa: {condition_number(a):.3f} -> {condition_number(transformed_a):.3f}")
    plain = descent_trajectory(x0, a, b, step_size, n_iterations)
    ok &= report("it is not a relabelling of plain gradient descent",
                 metric_name == "identity" or not np.allclose(plain, direct, atol=1e-6),
                 f"max |x_t^GD - x_t^M| = {np.abs(plain - direct).max():.3e}")
    return ok, direct, transformed


def check_algorithm_change(problem, x0):
    a, b, x_star = problem["a"], problem["b"], problem["x_star"]
    print("\nalgorithm change: Newton's method is a different process, not a different view")
    ok = True
    iterates = newton_trajectory(x0, a, b, 2)
    ok &= report("Newton reaches the minimizer in one step",
                 np.allclose(iterates[1], x_star, atol=1e-9),
                 f"|x_1 - x^*| = {np.linalg.norm(iterates[1] - x_star):.3e}")
    g = gradient(np.asarray(x0, dtype=float), a, b)
    ok &= report("Newton direction == steepest descent under M = A",
                 np.isclose(abs(metric_cosine(newton_direction(np.asarray(x0, dtype=float), a, b),
                                              unit_steepest_descent_direction(g, a), a)), 1.0, atol=1e-9),
                 "cos_A(d_newton, d_A) = 1")
    return ok


def main(argv=None):
    args = parse_arguments(argv)
    problem = make_problem(args.condition_number, np.radians(args.angle_deg), args.minimizer)
    a, b = problem["a"], problem["b"]
    step_size = args.lr if args.lr is not None else stable_step_size(problem["eigenvalues"], args.lr_fraction)
    x0 = np.asarray(args.x0, dtype=float)

    print("problem")
    print(f"  A =\n{np.array2string(a, precision=4, prefix='      ')}")
    print(f"  eigenvalues = {np.array2string(problem['eigenvalues'], precision=4)}, "
          f"kappa = {problem['condition_number']:.3f}, basis angle = {args.angle_deg:.1f} deg")
    print(f"  x0 = {x0}, x* = {problem['x_star']}, step size = {step_size:.6f} "
          f"(stability limit {2 / problem['eigenvalues'][0]:.6f})")

    iterates = descent_trajectory(x0, a, b, step_size, args.iterations)
    quality = step_quality(iterates, a, b)

    passed, z_iterates, values_x, values_z = check_coordinate_change_invariances(problem, iterates)
    metric = make_metric(args.metric, a)
    passed &= check_metric_dependent_quantities(problem, iterates, metric, args.metric)
    precondition_ok, _, _ = check_preconditioning(
        problem, x0, step_size, args.iterations, metric, args.metric)
    passed &= precondition_ok
    passed &= check_algorithm_change(problem, x0)

    print("\nlocal step quality of plain gradient descent (mean over iterations)")
    for key, values in quality.items():
        print(f"  {key:42s} {values.mean():+.4f}   (min {values.min():+.4f}, max {values.max():+.4f})")
    print("  the same steps are Euclidean-perfect and A-imperfect at the same time;")
    print(f"  meanwhile f(x_t) - f(x*) fell from {values_x[0] - objective(problem['x_star'], a, b):.4f} "
          f"to {values_x[-1] - objective(problem['x_star'], a, b):.3e}.")

    if not args.no_figures:
        args.outdir.mkdir(parents=True, exist_ok=True)
        arrow_indices = np.unique(np.linspace(0, min(args.iterations, 12) - 1,
                                              args.arrows).astype(int))
        runs = {}
        for name in ("identity", "jacobi", "hessian"):
            run_metric = make_metric(name, a)
            run_step = metric_optimal_step_size(a, run_metric)
            runs[name] = {"step_size": run_step,
                          "iterates": descent_trajectory(x0, a, b, run_step, args.iterations,
                                                         metric=run_metric)}
        plot_trajectory_bases(problem, iterates, arrow_indices, args.outdir / "fig_01_trajectory_bases.png")
        plot_objective_history(values_x, values_z, objective(problem["x_star"], a, b),
                               args.outdir / "fig_02_objective_history.png")
        plot_step_quality(quality, args.outdir / "fig_03_step_quality.png")
        plot_metric_steepest_descent(problem, iterates[len(iterates) // 3], metric, args.metric,
                                     args.outdir / "fig_04_metric_steepest_descent.png")
        plot_preconditioning_comparison(problem, runs, args.outdir / "fig_05_preconditioning.png")
        print(f"\nfigures written to {args.outdir}")

    print("\nall checks passed" if passed else "\nsome checks failed")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
