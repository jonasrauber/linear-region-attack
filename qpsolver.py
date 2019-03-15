import logging
import time
import jax.numpy as np
import jax
from functools import partial


def step(x0, get_A, xr, normalizer, b_finite, vec, L, counter, xp, mustar, mu):
    # ---------------------------------------------------------------------

    # fixed param: x0, b_finite, vec
    # variable param: L, counter
    # taken and returned: xp, mustar, mu
    # returned: maxfeasible, dual_objective, primal_dual_gap

    # ---------------------------------------------------------------------

    logging.info('compiling step')

    Adot, ATdot, _, _, _ = get_A(xr, normalizer)

    # compute gradient of the dual objective
    Axp = Adot(xp)
    gradq = Axp - vec

    # compute step of accelerated projected gradient descent
    mustarold = mustar
    mustar = np.maximum(0, mu - gradq / L)
    mu = mustar + (counter / (counter + 3)) * (mustar - mustarold)
    mu = np.maximum(0, mu)

    # update ATmu, alpha, beta are set to their optimal values in the dual
    # NOTICE: this improves the dual value, but it is a HACK as we optimize
    # over mu (and jump around wrt to alpha, beta)
    ATmu = ATdot(mu)
    alpha = np.maximum(0, x0 - ATmu - 1)
    beta = np.maximum(0, ATmu - x0)
    xp = alpha + ATmu - beta

    # compute primal objective
    # x = x0 - xp is the primal variable (at the dual optimal), need not be feasible
    primal_objective = 0.5 * np.linalg.norm(xp)**2
    dual_objective = -0.5 * np.linalg.norm(xp)**2 + x0.T.dot(xp) - b_finite.T.dot(mu) - np.sum(alpha)
    primal_dual_gap = primal_objective - dual_objective

    feasible = vec - Axp
    maxfeasible = np.amax(feasible)

    return xp, mustar, mu, dual_objective, primal_dual_gap, maxfeasible


def cond_fun(maxiter, bound, feasStop, state):
    logging.info('compiling cond_fun')
    counter = state[1]
    dual_objective, primal_dual_gap, maxfeasible = state[7:10]

    cond1 = counter <= maxiter
    cond2 = dual_objective < bound
    cond3 = np.logical_or(
        np.absolute(primal_dual_gap) > 1e-6,
        np.logical_or(
            maxfeasible > feasStop,
            np.logical_and(counter < 200, maxfeasible >= 0)
        )
    )
    return np.logical_and(cond1, np.logical_and(cond2, cond3))


def state_update_fun(get_A, xr, normalizer, b_finite, vec, x0, state):
    logging.info('compiling state_update_fun')

    L, counter = state[0:2]
    xp, mustar, mu = state[2:5]
    xp, mustar, mu, dual_objective, primal_dual_gap, maxfeasible = step(
        x0, get_A, xr, normalizer, b_finite, vec,
        L, counter, xp, mustar, mu)

    # TODO: maybe use lax.cond once available https://github.com/google/jax/issues/325
    best_dual_objective, best_x = state[5:7]
    update_best = jax.lax.gt(dual_objective, best_dual_objective)
    best_x = update_best * (x0 - xp) + (1 - update_best) * best_x
    best_dual_objective = update_best * dual_objective + (1 - update_best) * best_dual_objective

    counter = counter + 1

    # TODO: update L if dual smaller than -100
    # if dual_objective < -100:
    #     logging.warning('divergence due to hack with Lipschitz constant')
    #     if L < LMAX:
    #         logging.warning('increasing L and restarting from scratch')
    #         L = min(10 * L, LMAX)
    #         mu = np.zeros_like(mu)
    #         mustar = mu
    #         counter = 1

    return (
        L, counter,
        xp, mustar, mu,
        best_dual_objective, best_x,
        dual_objective, primal_dual_gap, maxfeasible,
    )


# kwargs are treated as static, but bound changes, so don't use kwargs here
@partial(jax.jit, static_argnums=(0, 2))
def solve_jit(x0, Ax0, get_A, xr, normalizer, b, L, bound, maxiter, feasStop):
    logging.info('compiling solve_jit')

    # constants
    vec = Ax0 - b

    # initialization
    mu = np.zeros_like(b)
    mustar = mu
    xp = np.zeros_like(x0)
    best_dual_objective = 0.
    best_x = x0
    counter = 1

    b_finite = np.where(np.isposinf(b), np.array(np.finfo(b.dtype).max), b)

    init_state = (L, counter, xp, mustar, mu, best_dual_objective, best_x, 0., np.inf, np.inf)

    _cond_fun = partial(cond_fun, maxiter, bound, feasStop)
    _state_update_fun = partial(state_update_fun, get_A, xr, normalizer, b_finite, vec, x0)

    final_state = jax.lax.while_loop(_cond_fun, _state_update_fun, init_state)

    counter = final_state[1]
    best_dual_objective = final_state[5]
    best_x = final_state[6]

    return counter, best_dual_objective, best_x


def solve(x0, Ax0, get_A, xr, normalizer, b, L, *, bound=np.inf, maxiter=4000, feasStop=1e-8):
    """Solves the following quadratic programming (QP) problem:

    min_x 1/2 (x - x0)' * (x - x0)
    s.t. A * x ≤ b and 0 ≤ x ≤ 1
    """

    t0 = time.time()
    counter, best_dual_objective, best_x = solve_jit(x0, Ax0, get_A, xr, normalizer, b, L, bound, maxiter, feasStop)
    t0 = time.time() - t0

    logging.info(f'took {t0:.1f} secs for {counter} it -> {counter / t0:.1f} it/sec')
    return best_x, best_dual_objective, counter
