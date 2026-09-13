# Basis, metric, and the reading of an optimization trajectory

A small numerical experiment about a simple claim:

> the same sequence of updates can look clumsy in one representation and natural in another,
> and the word "optimal" is not defined until a metric has been fixed.

Nothing here speeds up an optimizer. The point is the opposite: a change of basis changes
*nothing* about the trajectory, and yet it changes what the trajectory looks like. A change of
metric is a different operation entirely — it changes which direction even counts as "steepest".

## The problem

$$f(x) = \tfrac{1}{2} x^\top A x - b^\top x, \qquad A = Q \Lambda Q^\top \succ 0, \qquad x^\star = A^{-1}b$$

$A$ is built as a rotation of a diagonal matrix: $\Lambda = \operatorname{diag}(\kappa, 1)$ and $Q$ a
rotation by $\theta$ (default $\kappa = 25$, $\theta = 30^\circ$). The standard axes are therefore
deliberately misaligned with the principal axes of $A$, and the two coordinates of $x$ are coupled
through the off-diagonal entries of $A$.

Plain gradient descent, $x_{t+1} = x_t - \alpha (A x_t - b)$, is run from a nontrivial $x_0$ with
$\alpha$ just inside the stability limit $2/\lambda_{\max}$. The result zig-zags.

## Three views of one trajectory

The iterates $x_t$ are computed once. They are then *re-expressed*, never recomputed:

| view | map | what changes | what does not |
|---|---|---|---|
| original | $x_t$ | — | — |
| eigenbasis | $z_t = Q^\top x_t$ | the numbers on the axes | lengths, angles, objective values |
| whitened | $y_t = A^{1/2} x_t$ | lengths and angles too | the sequence of points, objective values |

In the eigenbasis the objective separates,

$$g(z) = \tfrac{1}{2} z^\top \Lambda z - (Q^\top b)^\top z = \sum_i \left( \tfrac{1}{2}\lambda_i z_i^2 - c_i z_i \right),$$

and the gradient descent recursion decouples into two independent scalar recursions

$$z_{t+1,i} - z^\star_i = (1 - \alpha \lambda_i)\,(z_{t,i} - z^\star_i).$$

That is the whole explanation of the zig-zag. With $\alpha$ near $2/\lambda_{\max}$ the stiff
coordinate has $1 - \alpha\lambda_{\max} < 0$ and flips sign every step, while the flat coordinate
decays slowly with $1 - \alpha\lambda_{\min} \lesssim 1$. The "oscillation" is not a pathology of the
algorithm; it is one under-damped scalar sequence seen through a rotated window. `tests/` asserts
the decoupled recursion holds to machine precision.

## Invariant vs. metric-dependent

Under the orthogonal change of basis $z = Q^\top x$ (checked numerically in `main.py`):

- the iterates themselves: $x_t = Q z_t$ exactly,
- objective values, gradient norms, Euclidean step lengths,
- Euclidean angles between any two vectors, so every Euclidean cosine score,
- the eigenvalues of $A$ and the condition number $\kappa$,
- the Newton direction as a vector: it transforms as $Q^\top d^{\text{newton}}$, i.e. it is the same
  arrow.

Depending on the chosen metric $M \succ 0$, and *not* obtainable by any change of basis:

- which unit direction gives the steepest first-order decrease,
  $\arg\min_{\|v\|_M = 1} \nabla f(x)^\top v \;\propto\; -M^{-1}\nabla f(x)$,
- the alignment score $\cos_M(u,v) = \frac{u^\top M v}{\|u\|_M \|v\|_M}$,
- therefore the verdict "this step was perfect".

The whitened view makes the difference concrete: $y = A^{1/2}x$ is a coordinate change that is not
orthogonal, so it does not preserve angles — and looking at the trajectory there is exactly the same
thing as measuring angles in the $A$-metric while staying in $x$.

## Local step quality

For a quadratic, every reasonable notion of "the best step from $x_t$" can be compared in closed
form. With $e_t = x_t - x^\star$, the Newton step is $s^{\text{newton}}_t = -e_t$ and

$$\frac{f(x_t) - f(x_t + s_t)}{f(x_t) - f(x^\star)} \;=\; 1 - \frac{\lVert s_t - s^{\text{newton}}_t \rVert_A^2}{\lVert s^{\text{newton}}_t \rVert_A^2}.$$

So "fraction of the available progress made by this step" is *literally* a distance measured in the
$A$-metric. A step is perfect in this sense iff it is the Newton step, and the metric under which
the criterion is well posed is $A$, not the Euclidean one.

The code reports, per iteration:

| score | meaning | value for plain GD |
|---|---|---|
| $\cos_I(s_t, -\nabla f)$ | Euclidean steepest-descent criterion | $1$ at every step, by construction |
| $\cos_I(s_t, d^{\text{newton}}_t)$ | Euclidean angle to the Newton direction | $0.38 \dots 0.999$ |
| $\cos_A(s_t, d^{\text{newton}}_t)$ | the same comparison in the $A$-metric | differs from the row above |
| decrease / best single-step decrease | the identity above | $\approx 0.14 \dots 0.33$ |
| decrease / best decrease along the same ray | exact line search on the chosen direction | hits $1.000$ at one iteration |

Every step of gradient descent is *exactly* optimal under the Euclidean criterion and *far* from
optimal under the $A$-criterion, simultaneously. Around iteration 16 of the default run, the step is
also exactly the best possible step along its own direction — a third sense in which it is "the
perfect step" — while still capturing only about a fifth of the available decrease. Meanwhile
$f(x_t)$ falls monotonically throughout: global progress is steady while local steps are
simultaneously perfect and poor, depending on which question is asked.

## Four operations that are easy to conflate

1. **Coordinate change.** $z = Q^\top x$ with $Q$ orthogonal. Same iterates, same objective values,
   same angles. Reveals structure; changes nothing.
2. **Metric change.** Replace $I$ by $M$ when measuring length and angle. No iterate moves — only
   the *scoring* of a step and the *definition* of steepest change.
3. **Preconditioning.** Actually follow $-M^{-1}\nabla f$. This is a different trajectory. The code
   verifies the identity: $M$-steepest descent in $x$ is exactly Euclidean gradient descent applied
   to $h(y) = f(M^{-1/2}y)$ with $y = M^{1/2}x$ — a metric change *acted upon* is a coordinate
   change plus the unchanged algorithm. With $M = A$ the transformed Hessian is $I$: the problem
   becomes isotropic, $\kappa: 25 \to 1$, and with the optimal step $\alpha = 1$ it is Newton's
   method, exact in one step. With $M = \operatorname{diag}(A)$ (Jacobi) it is a partial fix.
4. **Algorithm change.** Newton, momentum, conjugate gradients. Different process, not a different
   view of the same one.

(1) and (2) are re-descriptions. (3) and (4) change what actually happens. Conflating them produces
the false claim that "changing basis made the optimizer better".

## Running it

```bash
pip install -r requirements.txt
python main.py                       # default run, writes figures/ and prints all checks
python main.py --condition-number 100 --angle-deg 63 --lr-fraction 0.99
python main.py --metric jacobi --iterations 80 --x0 -3 4
python -m pytest tests -q
```

Parameters: `--condition-number`, `--angle-deg` (angle between the standard basis and the
eigenbasis), `--lr` / `--lr-fraction` (as a fraction of the stability limit $2/\lambda_{\max}$),
`--metric {identity,jacobi,hessian}`, `--iterations`, `--x0`, `--minimizer`, `--arrows`, `--outdir`,
`--no-figures`.

Figures written to `figures/`:

1. `fig_01_trajectory_bases.png` — the one trajectory in the original, eigen- and whitened bases,
   with the same physical steps drawn as arrows in all three.
2. `fig_02_objective_history.png` — $f(x_t)$ and $g(z_t)$ on top of each other, plus suboptimality
   on a log scale.
3. `fig_03_step_quality.png` — the alignment scores above, per iteration.
4. `fig_04_metric_steepest_descent.png` — the Euclidean and $M$-unit balls at one point, their
   steepest directions, and the first-order decrease as a function of direction under both norms:
   the maximiser moves when the unit ball changes.
5. `fig_05_preconditioning.png` — trajectories and convergence for $M \in \{I, \operatorname{diag}(A), A\}$.

## Numerical checks

`main.py` prints a pass/fail line for each, and `tests/test_invariance.py` runs them over several
$(\kappa, \theta)$ settings:

- $x_t = Q z_t$ at every iteration, and $A = Q\Lambda Q^\top$, $Q^\top Q = I$;
- objective values agree in all three bases;
- Euclidean step lengths and angles are unchanged by the orthogonal change of basis;
- the eigenbasis iterates satisfy the decoupled recursion exactly;
- $-M^{-1}\nabla f$ maximises the first-order decrease over a brute-force sample of the $M$-unit
  sphere, for each $M$;
- the $M$-steepest direction is *not* the Euclidean steepest direction, while still being
  basis-covariant;
- $M$-steepest descent equals Euclidean gradient descent after the change of variables
  $y = M^{1/2}x$, and $\kappa$ of the transformed problem does not increase;
- preconditioning is not a relabelling: the objective histories genuinely differ;
- Newton is the $A$-steepest direction and lands on $x^\star$ in one step;
- $f(x_t)$ decreases monotonically.

## On framing-dependent answers

The mathematical content above is narrow, and worth stating carefully.

A judgement like "that step was optimal" is not a property of the step alone. It is a property of
the pair (step, criterion), and the criterion silently includes a metric. Fix a different metric and
the same step gets a different verdict — not because anything about the step changed, but because
the question was under-specified. In this experiment the *same* update is simultaneously the exact
maximiser of decrease under one norm and roughly a fifth as good as available under another. Both
answers are correct; neither is complete without naming its norm.

Separately, and more weakly: a representation can hide structure. The zig-zag looks like erratic
behaviour in the original coordinates and is visibly two clean geometric decays in the eigenbasis.
Nothing improved — the arrows are the same arrows — but a description that made the process look
defective became one that makes it look inevitable.

Two things this does *not* show. It does not show that every disagreement is a matter of framing:
once a metric is fixed, the ranking of steps is determined, and the numbers are not negotiable. And
it does not show that all representations are equally good: some genuinely expose invariant
structure (eigenvalues, condition number) that others obscure. The honest summary is that *some*
apparently binary questions carry an implicit choice of coordinates or criterion, and that
identifying that choice is what makes the question answerable.

## Related concepts

- **Conditioning and steepest descent.** J. Nocedal and S. J. Wright, *Numerical Optimization*, 2nd
  ed., Springer, 2006, ch. 3.
- **Steepest descent under a general norm; Newton as steepest descent in the Hessian norm.**
  S. Boyd and L. Vandenberghe, *Convex Optimization*, Cambridge University Press, 2004, §9.4–9.5.
- **Preconditioning.** Y. Saad, *Iterative Methods for Sparse Linear Systems*, 2nd ed., SIAM, 2003.
  J. R. Shewchuk, *An Introduction to the Conjugate Gradient Method Without the Agonizing Pain*,
  Carnegie Mellon University, 1994.
- **Diagonal scaling is near-optimal.** A. van der Sluis, "Condition numbers and equilibration of
  matrices", *Numerische Mathematik* 14 (1969), 14–23.
- **Eigendecomposition, principal axes, matrix square roots.** G. H. Golub and C. F. Van Loan,
  *Matrix Computations*, 4th ed., Johns Hopkins University Press, 2013. L. N. Trefethen and D. Bau,
  *Numerical Linear Algebra*, SIAM, 1997.
- **Natural gradient.** S.-I. Amari, "Natural gradient works efficiently in learning",
  *Neural Computation* 10 (1998), 251–276. J. Martens, "New insights and perspectives on the natural
  gradient method", *Journal of Machine Learning Research* 21 (2020).
- **Information geometry.** S.-I. Amari and H. Nagaoka, *Methods of Information Geometry*, AMS/Oxford
  University Press, 2000. S.-I. Amari, *Information Geometry and Its Applications*, Springer, 2016.
- **Mirror descent (steepest descent under a non-quadratic geometry).** A. Nemirovski and D. Yudin,
  *Problem Complexity and Method Efficiency in Optimization*, Wiley, 1983. A. Beck and M. Teboulle,
  "Mirror descent and nonlinear projected subgradient methods for convex optimization",
  *Operations Research Letters* 31 (2003), 167–175.
- **Riemannian optimization and coordinate invariance.** P.-A. Absil, R. Mahony and R. Sepulchre,
  *Optimization Algorithms on Matrix Manifolds*, Princeton University Press, 2008. N. Boumal,
  *An Introduction to Optimization on Smooth Manifolds*, Cambridge University Press, 2023.
- **Whitening.** A. Hyvärinen, J. Karhunen and E. Oja, *Independent Component Analysis*, Wiley, 2001,
  ch. 6.
