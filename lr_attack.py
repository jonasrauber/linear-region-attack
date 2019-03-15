import time
import logging
import random
import jax
import jax.numpy as np
import numpy as onp
import tqdm
from functools import partial
from functools import wraps

from qpsolver import solve
from utils import is_device_array, scatter, onehot


def accuracy(predict_class, images, labels, batch_size=100):
    total = len(images)
    correct = 0
    for i in tqdm.trange(0, len(images), batch_size):
        j = i + batch_size
        predicted_class = predict_class(images[i:j])
        correct += onp.sum(predicted_class == labels[i:j])
    return correct / total


def l2_distance(x0, x):
    assert x0.shape == x.shape
    return onp.linalg.norm(jax.device_get(x0) - jax.device_get(x))


def misclassification_polytope(a, c, ls):
    """creates misclassification constraints"""
    assert a.ndim == 2
    assert a.shape[0] == 1  # only batch size 1 is supported
    n_classes = a.shape[1]

    u = a[:, ls] - a[:, c]

    c = np.atleast_1d(np.asarray([c]).squeeze())
    ls = np.atleast_1d(np.asarray([ls]).squeeze())

    Av = lambda Vv: Vv[:, c] - Vv[:, ls]  # noqa: E731
    vA = lambda v: (scatter(c, np.sum(np.atleast_2d(v), axis=-1, keepdims=True), n_classes) +  # noqa: E731
                    scatter(ls, -np.atleast_2d(v), n_classes))

    return Av, vA, u


def relu_polytope(a, f):
    """creates polytope constraints"""
    sf = np.sign(f)
    nsf = -sf
    u = sf * a

    Av = lambda Vv: nsf * Vv  # noqa: E731
    vA = lambda v: nsf * v  # noqa: E731

    # Some of these constrains are always fulfilled and can be removed.
    non_trivial = sf != 0
    # create a vector with 0s for all non-trivial ones,
    # and inf for trivial ones
    trivial_inf = 1 / non_trivial.astype(np.float32) - 1
    # make the upper bound of trivial ones infinity to make them least violated
    u = u + trivial_inf
    return Av, vA, u


def get_other_classes(*, exclude, total, first=None):
    rest = [x for x in range(total) if x != exclude and x != first]
    random.shuffle(rest)
    first = [] if first is None else [first]
    return first + rest


def flatten(x):
    if isinstance(x, list):
        for y in x:
            yield from flatten(y)
    else:
        assert isinstance(x, tuple)
        for y in x:
            yield y


def flatten_dims(x):
    return np.reshape(x, (x.shape[0], -1))


def layer_size(x):
    _, n = x.shape
    return n


def flatten_predict(predict):
    @wraps(predict)
    def flat_predict(x):
        output, additional_outputs = predict(x)
        additional_outputs = list(flatten(additional_outputs))
        additional_outputs = list(map(flatten_dims, additional_outputs))
        additional_outputs = np.concatenate(additional_outputs, axis=-1)
        return output, additional_outputs

    return flat_predict


def return_classes_logits_layer_sizes(f, *args, **kwargs):
    logging.info(f'compiling return_classes_logits_layer_sizes')
    logits, additional_outputs = f(*args, **kwargs)
    additional_outputs = list(flatten(additional_outputs))
    additional_outputs = list(map(flatten_dims, additional_outputs))
    rows_per_layer = list(map(layer_size, additional_outputs))
    return np.argmax(logits, axis=-1), logits, rows_per_layer


def generic_get_A(predict, label, other_classes, xr, normalizer):
    # linearize net around xr
    fxr, vjp_fun = jax.vjp(predict, xr)
    jvp_fun = partial(jax.jvp, predict, (xr,))
    _, jxrp = jvp_fun((xr,))
    offset = tuple(fx - Jx for fx, Jx in zip(fxr, jxrp))

    Av_misc, vA_misc, u_misc = misclassification_polytope(offset[0], label, other_classes)
    Av_relu, vA_relu, u_relu = relu_polytope(offset[1], fxr[1])

    assert u_misc.ndim == u_relu.ndim == 2
    assert u_misc.shape[0] == u_relu.shape[0]   # batch dimension

    n_constraints = u_misc.shape[1] + u_relu.shape[1]

    if normalizer is not None:
        assert normalizer.shape == (n_constraints,)
        assert normalizer.dtype == np.float32

    _, N = u_misc.shape
    if normalizer is not None:
        u_misc = u_misc * normalizer[:N]
        u_relu = u_relu * normalizer[N:]

    def Adot(v):
        v = v.reshape((1,) + xr.shape[1:])
        _, Vv = jvp_fun((v,))
        Vv_misc, Vv_relu = Vv
        r_misc = Av_misc(Vv_misc)
        r_relu = Av_relu(Vv_relu)
        assert r_misc.ndim == r_relu.ndim == 2
        assert r_misc.shape[0] == r_relu.shape[0]  # batch dimension
        r = np.concatenate((r_misc, r_relu), axis=1)
        r = r.squeeze(axis=0)
        assert r.shape == (n_constraints,)
        if normalizer is not None:
            r = normalizer * r
        return r

    def ATdot(v):
        assert v.shape == (n_constraints,)
        v = v[onp.newaxis]
        if normalizer is not None:
            v = normalizer * v
        _, N = u_misc.shape
        assert v.ndim == 2
        v_misc, v_relu = v[:, :N], v[:, N:]
        v_misc = vA_misc(v_misc)
        v_relu = vA_relu(v_relu)
        v = (v_misc, v_relu)
        r, = vjp_fun(v)
        r = r.reshape((-1,))
        return r

    assert u_misc.shape[1] == len(other_classes)
    return Adot, ATdot, n_constraints, u_misc, u_relu


@partial(jax.jit, static_argnums=(0,))
def operator_norm_lower_bound(get_A, xr, normalizer):
    logging.info('compiling operator_norm_lower_bound')
    Adot, ATdot, _, _, _ = get_A(xr, normalizer)

    def body_fun(i, state):
        z, n = state
        u = ATdot(Adot(z))
        n = np.linalg.norm(u)
        return u / n, n

    # z = np.ones_like(xr.reshape(-1))
    z = xr.reshape(-1)  # a constant vector of e.g. ones fails if mean is subtracted
    _, n = jax.lax.fori_loop(0, 20, body_fun, (z, 0.))
    return n


@partial(jax.jit, static_argnums=(0,))
def init_region(get_A, xr, normalizer, v):
    logging.info('compiling init_region')
    Adot, _, _, u_misc, u_relu = get_A(xr, normalizer)
    Av = Adot(v)
    return Av, u_misc, u_relu


@partial(jax.jit, static_argnums=(0, 2, 3))
def calculate_normalizer(get_A, xr, n_constraints, rows_per_layer, *, k, normalizer=None, misc_factor=1.):
    logging.info('compiling calculate_normalizer')
    if normalizer is None:
        normalizer = np.ones((n_constraints,), dtype=np.float32)
    _, ATdot, _, u_misc, u_relu = get_A(xr, normalizer)
    _, n_misc = u_misc.shape
    misc_norms, layer_norms = estimate_layer_norms(ATdot, n_misc, rows_per_layer, k=k)
    assert misc_norms.shape == (n_misc,)
    normalizer = [misc_factor / misc_norms]
    assert len(rows_per_layer) == len(layer_norms)
    for n, norm in zip(rows_per_layer, layer_norms):
        normalizer.append(np.ones((n,)) / norm)
    normalizer = np.concatenate(normalizer)
    return normalizer, misc_norms, layer_norms


def estimate_layer_norms(ATdot, n_misc, rows_per_layer, *, k):
    """for each layer, samples k of the rows of A corresponding to that
    layer as well as all rows corresponding to the n logits and then
    estimates the norm of rows of A corresponding to each layer"""

    # TODO: consider using jax.random and thus drawing new samples every time;
    # right now we do the whole onehot vector creation statically
    indices = list(range(n_misc))
    offset = n_misc
    for layer_size in rows_per_layer:
        indices.extend(random.sample(range(offset, offset + layer_size), k))
        offset += layer_size

    logging.info(f'{len(indices)} randomly selected rows of A: {indices}')

    n_constraints = offset

    vs = onp.zeros((len(indices), n_constraints), dtype=onp.float32)
    for row, col in enumerate(indices):
        vs[row, col] = 1.

    ATdot = jax.vmap(ATdot)
    rows = ATdot(vs)
    assert rows.ndim == 2

    norms = np.linalg.norm(rows, axis=-1)

    # TODO: use median once supported by jax.numpy:
    # https://github.com/google/jax/issues/70

    layer_norms = []
    for i in range(n_misc, len(norms), k):
        assert i + k <= len(norms)
        m = np.mean(norms[i:i + k])
        layer_norms.append(m)

    return norms[:n_misc], layer_norms


def line_search(predict_class, x0, label, x, minimum=0., maximum=1., num=100, s=None):
    x = jax.device_get(x)

    assert not is_device_array(x0)
    assert not is_device_array(label)

    assert x0.shape == x.shape
    assert x0.shape[0] == 1  # batch dimension

    if s is None:
        s = onp.linspace(minimum, maximum, num=num + 1)[1:]

    p = x - x0
    ps = s.reshape((-1,) + (1,) * (p.ndim - 1)) * p
    xs = x0 + ps

    assert xs.shape[1:] == x0.shape[1:]

    classes = jax.device_get(predict_class(xs))
    assert classes.ndim == 1
    indices = onp.flatnonzero(classes != label)
    assert indices.ndim == 1
    try:
        best = indices[0]
    except IndexError:
        raise ValueError
    logging.info(f'best: {best} -> {s[best]}')
    return xs[best][onp.newaxis], classes[best], best


def get_region(k, x0, best_adv, *, gamma):
    x0 = jax.device_get(x0)
    best_adv = jax.device_get(best_adv)

    # TODO: maybe check region around original input
    # if k == 0:
    #     # try the region around the original input
    #     return x0

    delta = biased_direction(x0, best_adv, prob=0.8)

    u = onp.linalg.norm(best_adv - x0)
    r = onp.random.uniform()
    logging.debug(f'sampled r = {r}')
    x = best_adv + delta / onp.linalg.norm(delta) * u * r**gamma
    x = jax.device_put(x)
    return x


def biased_direction(x0, best_adv, *, prob):
    dx = x0 - best_adv
    dx = dx / onp.linalg.norm(dx.reshape(-1))

    delta = onp.random.normal(size=x0.shape)
    delta = delta - onp.dot(delta.reshape(-1), dx.reshape(-1)) * dx
    delta = delta / onp.linalg.norm(delta.reshape(-1))

    alpha = onp.random.uniform(0., onp.pi)

    if onp.random.uniform() > prob:
        # with probability 1 - prob, sample from the half space further away from x0
        alpha = -alpha

    return onp.sin(alpha) * dx + onp.cos(alpha) * delta


def run(n_classes, predict, params, images, labels, find_starting_point, args):
    t0 = time.time()

    random.seed(22)
    onp.random.seed(22)

    logging.info(f'number of samples: {len(images)}')
    logging.info(f'n_classes: {n_classes}')

    predict = partial(predict, params)

    predict_class_logits_layer_sizes = partial(return_classes_logits_layer_sizes, predict)
    predict_class_logits_layer_sizes = jax.jit(predict_class_logits_layer_sizes)

    predict = flatten_predict(predict)

    def predict_class(x):
        return predict_class_logits_layer_sizes(x)[0]

    if args.accuracy:
        logging.info(f'accuracy: {accuracy(predict_class, images, labels)}')

    x0_host = images[args.image][onp.newaxis]
    label_host = labels[args.image][onp.newaxis]
    logging.info(f'label: {label_host}')

    x0 = jax.device_put(x0_host)
    x0_flat = x0.reshape((-1,))

    l2 = partial(l2_distance, x0_host)

    x0_class, x0_logits, rows_per_layer = jax.device_get(predict_class_logits_layer_sizes(x0))
    logging.info(f'predicted class: {x0_class}, logits: {x0_logits}')

    logging.info(f'rows per layer: {rows_per_layer}')

    if x0_class != label_host:
        logging.warning(f'unperturbed input is misclassified by the model as {x0_class}')
        result = {
            'is_adv': True,
            'x0': x0_host,
            'label': label_host.item(),
            'adv': x0_host,
            'adv_class': x0_class,
            'l2': 0.,
            'duration': time.time() - t0,
        }
        return result

    label = jax.device_put(label_host)

    best_adv, best_adv_class = find_starting_point(args, x0_host, label_host, x0_logits, predict_class)
    best_adv_l2 = l2(best_adv)
    best_adv_l2_hist = [(time.time() - t0, best_adv_l2)]

    if not args.no_line_search:
        logging.info('running line search to determine better starting point')
        best_adv, best_adv_class, _ = line_search(predict_class, x0_host, label_host, best_adv)
        best_adv_l2 = l2(best_adv)

    best_adv_l2_hist.append((time.time() - t0, best_adv_l2))

    logging.info(f'starting point class: {best_adv_class}')

    best_adv_l2_hist_hist = [best_adv_l2_hist]

    other_classes = get_other_classes(exclude=label.squeeze(), total=n_classes, first=best_adv_class)
    if args.max_other_classes:
        other_classes = other_classes[:args.max_other_classes]
    logging.info(f'other classes: {other_classes}')

    n_constraints = len(other_classes) + sum(rows_per_layer)
    logging.info(f'n_constraints: {n_constraints}')

    total_solver_iterations = 0

    get_A = partial(generic_get_A, predict, label, other_classes)

    # ------------------------------------------------------------------------
    # Loop over region
    # ------------------------------------------------------------------------
    for region in range(args.regions):
        logging.info('-' * 70)
        logging.info(f'{region + 1}. REGION')
        logging.info('-' * 70)

        xr = get_region(region, x0, best_adv, gamma=args.gamma)

        if not args.no_normalization:
            normalizer, misc_norms, layer_norms = calculate_normalizer(
                get_A, xr, n_constraints, rows_per_layer, k=10, misc_factor=args.misc_factor)
            logging.info(f'misc norms: {misc_norms}')
            logging.info(f'layer norms: {layer_norms}')
        else:
            normalizer = None

        Ax0, u_misc, u_relu = init_region(get_A, xr, normalizer, x0)

        L = operator_norm_lower_bound(get_A, xr, normalizer)
        logging.info(f'L = {L}')

        best_adv_l2_hist = [(time.time() - t0, best_adv_l2)]

        # ------------------------------------------------------------------------
        # Loop over other classes
        # ------------------------------------------------------------------------
        for active in range(len(other_classes)):
            # update upper bounds
            mask = onehot(active, len(other_classes), dtype=np.float32)
            infs = 1 / mask - 1
            u_misc_active = u_misc + infs
            u = np.concatenate((u_misc_active, u_relu), axis=1)
            u = u.squeeze(axis=0)

            assert best_adv.shape[0] == x0.shape[0] == 1
            bound = 0.5 * best_adv_l2 ** 2
            logging.info(f'bound: {bound}')

            potential_adv, best_dual, counter = solve(
                x0_flat, Ax0, get_A, xr, normalizer, u, L,
                bound=bound, maxiter=args.iterations)

            total_solver_iterations += jax.device_get(counter).item()

            potential_adv = potential_adv.reshape(x0.shape)
            potential_adv_l2 = l2(potential_adv)
            closer = potential_adv_l2 < best_adv_l2
            logging.info(f'closer = {closer}')

            if closer:
                try:
                    ratio = best_adv_l2 / potential_adv_l2
                    if ratio > 1.1:
                        s = onp.linspace(0.9, 1.1, num=101, endpoint=True)
                    else:
                        s = onp.linspace(0.9, ratio, num=101, endpoint=False)

                    logging.info(f'running line search with factors between {s.min()} and {s.max()}')
                    best_adv, best_adv_class, index = line_search(
                        predict_class, x0_host, label_host, potential_adv, s=s)
                    new_l2 = l2(best_adv)
                    logging.info(f'-> new best adv with l2 = {new_l2} (before: {best_adv_l2})')
                    assert new_l2 < best_adv_l2
                    best_adv_l2 = new_l2
                except ValueError:
                    logging.info(f'-> result not adversarial (tried with line search)')
                else:  # the first line search succeeded
                    if index == 0:  # the range of our line search can be extended to even smaller values
                        logging.info(f'running another line search with factors between 0 and 0.9')
                        # this line search should not fail because 0.9 works for sure
                        best_adv, best_adv_class, _ = line_search(predict_class, x0_host, label_host, potential_adv,
                                                                  minimum=0., maximum=0.90, num=100)
                        new_l2 = l2(best_adv)
                        logging.info(f'-> new best adv with l2 = {new_l2} (before: {best_adv_l2})')
                        assert new_l2 <= best_adv_l2
                        best_adv_l2 = new_l2

            best_adv_l2_hist.append((time.time() - t0, best_adv_l2))
            logging.info('-' * 70)

        logging.info([l for _, l in best_adv_l2_hist])
        best_adv_l2_hist_hist.append(best_adv_l2_hist)

    logging.info([[round(l, 2) for _, l in h] for h in best_adv_l2_hist_hist])

    best_adv_l2 = l2(best_adv)
    logging.info(f'final adversarial has l2 = {best_adv_l2}')
    logging.info(f'total number of iterations in QP solver: {total_solver_iterations}')

    result = {
        'x0': x0_host,
        'label': label_host.item(),
        'adv': best_adv,
        'adv_class': best_adv_class,
        'l2': best_adv_l2,
        'duration': time.time() - t0,
        'history': best_adv_l2_hist_hist,
        'other_classes': onp.asarray(other_classes).tolist(),
        'total_solver_iterations': total_solver_iterations,
    }
    return result
