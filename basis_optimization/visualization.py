import matplotlib.pyplot as plt
import numpy as np

from geometry import brute_force_steepest_direction, sample_unit_sphere, unit_steepest_descent_direction
from optimization import gradient, objective

ARROW_COLOR = "#d1495b"
TRAJECTORY_COLOR = "#1b263b"


def _limits(points, margin=0.18, square=False):
    low, high = points.min(axis=0), points.max(axis=0)
    span = np.maximum(high - low, 1e-6)
    low, high = low - margin * span, high + margin * span
    if square:
        center, half = 0.5 * (low + high), 0.5 * (high - low).max()
        low, high = center - half, center + half
    return low, high


def _contour(ax, matrix, vector, points, n_levels=14, square=False):
    low, high = _limits(points, square=square)
    grid_u, grid_v = np.meshgrid(np.linspace(low[0], high[0], 400),
                                 np.linspace(low[1], high[1], 400))
    flat = np.stack([grid_u.ravel(), grid_v.ravel()], axis=1)
    values = objective(flat, matrix, vector).reshape(grid_u.shape)
    minimum = values.min()
    levels = minimum + np.geomspace(1e-3 * (values.max() - minimum) + 1e-12,
                                    values.max() - minimum + 1e-12, n_levels)
    ax.contour(grid_u, grid_v, values, levels=levels, colors="#8d99ae", linewidths=0.7, alpha=0.9)
    ax.set_xlim(low[0], high[0])
    ax.set_ylim(low[1], high[1])
    ax.set_aspect("equal")


def _trajectory(ax, points, arrow_indices):
    ax.plot(points[:, 0], points[:, 1], "-", color=TRAJECTORY_COLOR, lw=1.0, alpha=0.8, zorder=3)
    ax.scatter(points[:, 0], points[:, 1], s=12, c=np.arange(len(points)),
               cmap="viridis", zorder=4, edgecolors="none")
    steps = np.diff(points, axis=0)
    for t in arrow_indices:
        ax.quiver(points[t, 0], points[t, 1], steps[t, 0], steps[t, 1],
                  angles="xy", scale_units="xy", scale=1.0, color=ARROW_COLOR,
                  width=0.008, zorder=5)
        ax.annotate(f"$s_{{{t}}}$", points[t] + 0.5 * steps[t], color=ARROW_COLOR,
                    fontsize=8, xytext=(4, 4), textcoords="offset points", zorder=6)


def plot_trajectory_bases(problem, iterates, arrow_indices, path):
    a, b = problem["a"], problem["b"]
    eigenvectors, eigenvalues = problem["eigenvectors"], problem["eigenvalues"]
    from geometry import symmetric_sqrt

    root = symmetric_sqrt(a)
    panels = [
        (iterates, a, b, "original basis $x$", "$x_1$", "$x_2$",
         "coupled: contours tilted, steps zig-zag", True),
        (iterates @ eigenvectors, np.diag(eigenvalues), eigenvectors.T @ b,
         r"eigenbasis $z = Q^\top x$", "$z_1$ (eigenvector 1)", "$z_2$ (eigenvector 2)",
         "decoupled: two independent 1-D decays", False),
        (iterates @ root.T, np.eye(2), np.linalg.solve(root, b),
         "whitened basis $y = A^{1/2}x$", "$y_1$", "$y_2$",
         "isotropic contours, but angles are not preserved", False),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.6), constrained_layout=True)
    for ax, (points, matrix, vector, title, xlabel, ylabel, note, show_axes) in zip(axes, panels):
        _contour(ax, matrix, vector, points, square=True)
        _trajectory(ax, points, arrow_indices)
        minimum = np.linalg.solve(matrix, vector)
        ax.plot(*minimum, "*", color="#e09f3e", ms=14, mec="k", mew=0.4, zorder=7)
        ax.set_title(f"{title}\n{note}", fontsize=10)
        if show_axes:
            for k in range(2):
                axis = eigenvectors[:, k]
                ends = minimum + np.array([-1.0, 1.0])[:, None] * 6.0 * axis
                ax.plot(ends[:, 0], ends[:, 1], ls="--", lw=0.8, color="#457b9d", alpha=0.8,
                        label="principal axes of $A$" if k == 0 else None)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    axes[0].plot([], [], color=ARROW_COLOR, lw=2, label="same physical steps $s_t$")
    axes[0].legend(loc="upper left", fontsize=8, frameon=False)
    fig.suptitle("One trajectory, three coordinate systems (panels 1-2 differ by a rotation only)",
                 fontsize=12)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_objective_history(values_x, values_z, minimum_value, path):
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), constrained_layout=True)
    iterations = np.arange(len(values_x))
    axes[0].plot(iterations, values_x, "-o", ms=3, color=TRAJECTORY_COLOR, label="$f(x_t)$ (original basis)")
    axes[0].plot(iterations, values_z, "--", lw=2, color="#2a9d8f", label=r"$g(z_t)$ (eigenbasis)")
    axes[0].axhline(minimum_value, color="#e09f3e", ls=":", label="$f(x^\\star)$")
    axes[0].set_xlabel("iteration $t$")
    axes[0].set_ylabel("objective value")
    axes[0].set_title("global progress is monotone and basis-independent", fontsize=10)
    axes[0].legend(fontsize=8, frameon=False)

    axes[1].semilogy(iterations, values_x - minimum_value, "-o", ms=3, color=TRAJECTORY_COLOR)
    axes[1].set_xlabel("iteration $t$")
    axes[1].set_ylabel("$f(x_t) - f(x^\\star)$")
    axes[1].set_title("suboptimality (log scale)", fontsize=10)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_step_quality(quality, path):
    labels = {
        "cosine_euclidean_negative_gradient": r"$\cos_I(s_t,\,-\nabla f)$  (Euclidean criterion)",
        "cosine_euclidean_newton": r"$\cos_I(s_t,\,d_t^{\mathrm{newton}})$",
        "cosine_hessian_newton": r"$\cos_A(s_t,\,d_t^{\mathrm{newton}})$  ($A$-metric)",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
    iterations = np.arange(len(next(iter(quality.values()))))
    for key, label in labels.items():
        axes[0].plot(iterations, quality[key], "-o", ms=3, label=label)
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].set_xlabel("iteration $t$")
    axes[0].set_ylabel("alignment")
    axes[0].set_title("the same step scores differently under different criteria", fontsize=10)
    axes[0].legend(fontsize=8, frameon=False)

    axes[1].plot(iterations, quality["decrease_fraction_of_best_single_step"], "-o", ms=3,
                 label="achieved decrease / best possible single-step decrease")
    axes[1].plot(iterations, quality["line_search_efficiency"], "-s", ms=3,
                 label="achieved decrease / best decrease along the same ray")
    axes[1].set_xlabel("iteration $t$")
    axes[1].set_ylabel("fraction")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_title("'perfect step' depends on what you compare against", fontsize=10)
    axes[1].legend(fontsize=8, frameon=False)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_metric_steepest_descent(problem, x, metric, metric_name, path):
    a, b = problem["a"], problem["b"]
    g = gradient(x, a, b)
    identity = np.eye(2)
    euclidean_direction = unit_steepest_descent_direction(g, identity)
    metric_direction = unit_steepest_descent_direction(g, metric)
    _, angles, euclidean_rates = brute_force_steepest_direction(g, identity)
    _, _, metric_rates = brute_force_steepest_direction(g, metric)
    _, euclidean_circle = sample_unit_sphere(identity)
    _, metric_circle = sample_unit_sphere(metric)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0), constrained_layout=True)
    scale = 0.5 * np.linalg.norm(problem["x_star"] - x) / max(np.abs(metric_circle).max(), 1e-9)
    points = np.vstack([x + scale * metric_circle, x + scale * euclidean_circle])
    _contour(axes[0], a, b, points)
    axes[0].plot(x[0] + scale * euclidean_circle[:, 0], x[1] + scale * euclidean_circle[:, 1],
                 color="#1b263b", lw=1.2, label=r"$\|v\|_I = 1$")
    axes[0].plot(x[0] + scale * metric_circle[:, 0], x[1] + scale * metric_circle[:, 1],
                 color="#d1495b", lw=1.2, label=r"$\|v\|_M = 1$")
    axes[0].quiver(*x, *(scale * euclidean_direction), angles="xy", scale_units="xy", scale=1,
                   color="#1b263b", width=0.008)
    axes[0].quiver(*x, *(scale * metric_direction), angles="xy", scale_units="xy", scale=1,
                   color="#d1495b", width=0.008)
    axes[0].plot(*x, "ko", ms=5)
    axes[0].set_title(f"steepest descent at $x$: Euclidean vs $M$ = {metric_name}", fontsize=10)
    axes[0].set_xlabel("$x_1$")
    axes[0].set_ylabel("$x_2$")
    axes[0].legend(fontsize=8, frameon=False, loc="upper right")

    degrees = np.degrees(angles)
    axes[1].plot(degrees, euclidean_rates, color="#1b263b", label=r"$-\nabla f^\top v$, $\|v\|_I=1$")
    axes[1].plot(degrees, metric_rates, color="#d1495b", label=r"$-\nabla f^\top v$, $\|v\|_M=1$")
    for rates, color in ((euclidean_rates, "#1b263b"), (metric_rates, "#d1495b")):
        best = int(np.argmax(rates))
        axes[1].plot(degrees[best], rates[best], "o", color=color, ms=7, mec="k", mew=0.4)
        axes[1].axvline(degrees[best], color=color, ls=":", lw=0.9)
    axes[1].set_xlabel("direction angle (degrees)")
    axes[1].set_ylabel("first-order decrease per unit length")
    axes[1].set_title("the maximiser moves when the unit ball changes", fontsize=10)
    axes[1].legend(fontsize=8, frameon=False)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_preconditioning_comparison(problem, runs, path):
    a, b, x_star = problem["a"], problem["b"], problem["x_star"]
    minimum_value = objective(x_star, a, b)
    all_points = np.vstack([np.vstack([run["iterates"], x_star]) for run in runs.values()])
    floor = 1e-12 * (objective(next(iter(runs.values()))["iterates"][0], a, b) - minimum_value)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0), constrained_layout=True)
    _contour(axes[0], a, b, all_points)
    for (name, run), color in zip(runs.items(), ("#1b263b", "#2a9d8f", "#d1495b", "#e09f3e")):
        label = f"$M$ = {name}, $\\alpha$ = {run['step_size']:.4f}"
        axes[0].plot(run["iterates"][:, 0], run["iterates"][:, 1], "-o", ms=3, lw=1.0,
                     color=color, label=label)
        gaps = np.maximum(objective(run["iterates"], a, b) - minimum_value, floor)
        axes[1].semilogy(gaps, "-o", ms=3, color=color, label=label)
        if gaps[1] <= floor:
            axes[1].annotate("exact after one step\n(machine precision)", (1, floor),
                             xytext=(18, 12), textcoords="offset points", fontsize=8, color=color)
    axes[0].plot(*x_star, "*", color="#e09f3e", ms=14, mec="k", mew=0.4, zorder=7)
    axes[0].set_xlabel("$x_1$")
    axes[0].set_ylabel("$x_2$")
    axes[0].set_title("preconditioning changes the iterates (original basis)", fontsize=10)
    axes[0].legend(fontsize=8, frameon=False)
    axes[1].set_ylim(0.3 * floor, None)
    axes[1].set_xlabel("iteration $t$")
    axes[1].set_ylabel("$f(x_t) - f(x^\\star)$")
    axes[1].set_title(r"each metric with its own optimal step $2/(\lambda_{\min}+\lambda_{\max})$",
                      fontsize=10)
    axes[1].legend(fontsize=8, frameon=False)
    fig.savefig(path, dpi=160)
    plt.close(fig)
