
from __future__ import print_function
import numpy as np
from scipy.special import digamma
from sklearn.neighbors import NearestNeighbors


def ksg_mi(X, Y, data, k=5, boolean=True, significance_level=0.01):
    """ Adapted from https://github.com/syanga/pycit/blob/master/pycit/estimators/ksg_mi.py
        KSG Mutual Information Estimator
        Based on: https://arxiv.org/abs/cond-mat/0305641
        x_data: data with shape (num_samples, x_dim) or (num_samples,)
        y_data: data with shape (num_samples, y_dim) or (num_samples,)
        k: number of nearest neighbors for estimation
           * k recommended to be on the order of ~ num_samples/10 for independence testing
    """
    x_data = np.array(data[X])
    y_data = np.array(data[Y])
    assert x_data.shape[0] == y_data.shape[0]
    num_samples = x_data.shape[0]

    lookup = NearestNeighbors(metric='chebyshev')
    lookup.fit(np.concatenate((x_data.reshape(-1, 1) if x_data.ndim == 1 else x_data,
                               y_data.reshape(-1, 1) if y_data.ndim == 1 else y_data), axis=1))

    # estimate entropies
    radius = lookup.kneighbors(n_neighbors=k, return_distance=True)[0]
    radius = np.nextafter(radius[:, -1], 0)

    lookup.fit(x_data.reshape(-1, 1) if x_data.ndim == 1 else x_data)
    n_x = np.array([i.size for i in lookup.radius_neighbors(radius=radius, return_distance=False)])

    lookup.fit(y_data.reshape(-1, 1) if y_data.ndim == 1 else y_data)
    n_y = np.array([i.size for i in lookup.radius_neighbors(radius=radius, return_distance=False)])

    p_val = digamma(num_samples) + digamma(k) - np.mean(digamma(n_x+1.) + digamma(n_y+1.))
    if p_val >= significance_level:
        return True
    else:
        return False


def ksg_cmi(X, Y, Z, data, boolean=True, significance_level=0.01, **kwargs):
    k = list(kwargs.values())[0]

    """ Adapted from https://github.com/syanga/pycit/blob/master/pycit/estimators/ksg_cmi.py
        KSG Conditional Mutual Information Estimator: I(X;Y|Z)
        See e.g. http://proceedings.mlr.press/v84/runge18a.html
        x_data: data with shape (num_samples, x_dim) or (num_samples,)
        y_data: data with shape (num_samples, y_dim) or (num_samples,)
        z_data: conditioning data with shape (num_samples, z_dim) or (num_samples,)
        k: number of nearest neighbors for estimation
           * k recommended to be on the order of ~ num_samples/10 for independence testing
    """

    if len(Z) == 0:
        return ksg_mi(X, Y, data, k=5)
    else:
        x_data = np.array(data[X])
        y_data = np.array(data[Y])
        z_data = np.array(data[list(Z)])
        xzy_data = np.concatenate((x_data.reshape(-1, 1) if x_data.ndim == 1 else x_data,
                                   y_data.reshape(-1, 1) if y_data.ndim == 1 else y_data,
                                   z_data.reshape(-1, 1) if z_data.ndim == 1 else z_data), axis=1)


        lookup = NearestNeighbors(metric='chebyshev')
        lookup.fit(xzy_data)

        radius = lookup.kneighbors(n_neighbors=k, return_distance=True)[0]
        radius = np.nextafter(radius[:, -1], 0)

        x_dim = x_data.shape[1] if x_data.ndim > 1 else 1
        z_dim = z_data.shape[1] if z_data.ndim > 1 else 1

        # compute entropies
        lookup.fit(xzy_data[:, :x_dim+z_dim])
        n_xz = np.array([i.size for i in lookup.radius_neighbors(radius=radius, return_distance=False)])

        lookup.fit(xzy_data[:, x_dim:])
        n_yz = np.array([i.size for i in lookup.radius_neighbors(radius=radius, return_distance=False)])

        lookup.fit(xzy_data[:, x_dim:x_dim+z_dim])
        n_z = np.array([i.size for i in lookup.radius_neighbors(radius=radius, return_distance=False)])
        p_val = digamma(k) + np.mean(digamma(n_z+1.) - digamma(n_xz+1.) - digamma(n_yz+1.))
        if p_val >= significance_level:
            return True
        else:
            return False



def mixed_mi(X, Y, data, k=5, boolean=True, significance_level=0.01):
    """adapted from https://github.com/syanga/pycit/blob/master/pycit/estimators/mixed_mi.py
        KSG Mutual Information Estimator for continuous/discrete mixtures.
        Based on: https://arxiv.org/abs/1709.06212
        data_matrix: all data
        x: data nodes
        y: data nodes
        s: conditioning nodes
        k: number of nearest neighbors for estimation
           * k recommended to be on the order of ~ num_samples/10 for independence testing
    """

    x_data = np.array(data[X])
    y_data = np.array(data[Y])

    assert x_data.shape[0] == y_data.shape[0]
    num_samples = x_data.shape[0]

    x_y = np.concatenate((x_data.reshape(-1, 1) if x_data.ndim == 1 else x_data,
                          y_data.reshape(-1, 1) if y_data.ndim == 1 else y_data), axis=1)

    lookup = NearestNeighbors(metric='chebyshev')
    lookup.fit(x_y)

    # compute k-NN distances
    radius = lookup.kneighbors(n_neighbors=k, return_distance=True)[0]
    radius = np.nextafter(radius[:, -1], 0)

    # modification for discrete-continuous
    k_list = k*np.ones(radius.shape, dtype='i')
    where_zero = np.array(radius == 0.0, dtype='?')
    if np.any(where_zero > 0):
        matches = lookup.radius_neighbors(x_y[where_zero], radius=0.0, return_distance=False)
        k_list[where_zero] = np.array([i.size for i in matches])

    # estimate entropies
    lookup.fit(x_data.reshape(-1, 1) if x_data.ndim == 1 else x_data)
    n_x = np.array([i.size for i in lookup.radius_neighbors(radius=radius, return_distance=False)])

    lookup.fit(y_data.reshape(-1, 1) if y_data.ndim == 1 else y_data)
    n_y = np.array([i.size for i in lookup.radius_neighbors(radius=radius, return_distance=False)])
    p_val = digamma(num_samples) + np.mean(digamma(k_list) - digamma(n_x+1.) - digamma(n_y+1.))
    if p_val >= significance_level:
        return True
    else:
        return False


def mixed_cmi(X, Y, Z, data, boolean=True, significance_level=0.01, **kwargs):
    k = list(kwargs.values())[0]

    """ adapted from https://github.com/syanga/pycit/blob/master/pycit/estimators/mixed_cmi.py
        KSG Conditional Mutual Information Estimator for continuous/discrete mixtures.
        See e.g. http://proceedings.mlr.press/v84/runge18a.html
        as well as: https://arxiv.org/abs/1709.06212
        data_matrix: all data
        x: data nodes
        y: data nodes
        s: conditioning nodes
        k: number of nearest neighbors for estimation
           * k recommended to be on the order of ~ num_samples/10 for independence testing
    """

    if len(Z) == 0:
        return mixed_mi(X, Y, data, k=k)
    else:
        x_data = np.array(data[X])
        y_data = np.array(data[Y])
        z_data = np.array(data[list(Z)])
        xzy_data = np.concatenate((x_data.reshape(-1, 1) if x_data.ndim == 1 else x_data,
                                   y_data.reshape(-1, 1) if y_data.ndim == 1 else y_data,
                                   z_data.reshape(-1, 1) if z_data.ndim == 1 else z_data), axis=1)

        lookup = NearestNeighbors(metric='chebyshev')
        lookup.fit(xzy_data)

        radius = lookup.kneighbors(n_neighbors=k, return_distance=True)[0]
        radius = np.nextafter(radius[:, -1], 0)

        # modification for discrete-continuous
        k_list = k*np.ones(radius.shape, dtype='i')
        where_zero = np.array(radius == 0.0, dtype='?')
        if np.any(where_zero > 0):
            matches = lookup.radius_neighbors(xzy_data[where_zero], radius=0.0, return_distance=False)
            k_list[where_zero] = np.array([i.size for i in matches])

        x_dim = x_data.shape[1] if x_data.ndim > 1 else 1
        z_dim = z_data.shape[1] if z_data.ndim > 1 else 1

        # compute entropies
        lookup.fit(xzy_data[:, :x_dim+z_dim])
        n_xz = np.array([i.size for i in lookup.radius_neighbors(radius=radius, return_distance=False)])

        lookup.fit(xzy_data[:, x_dim:])
        n_yz = np.array([i.size for i in lookup.radius_neighbors(radius=radius, return_distance=False)])

        lookup.fit(xzy_data[:, x_dim:x_dim+z_dim])
        n_z = np.array([i.size for i in lookup.radius_neighbors(radius=radius, return_distance=False)])
        p_val = np.mean(digamma(k_list) + digamma(n_z+1.) - digamma(n_xz+1.) - digamma(n_yz+1.))
        if p_val >= significance_level:
            return True
        else:
            return False