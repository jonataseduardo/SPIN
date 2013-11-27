# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
#from scipy.spatial.distance import pdist
#from scipy.spatial.distance import squareform
#from scipy.stats import scoreatpercentile


def matrix_plot(dist_matrix, y=None, labels=False):

    """
    Parameters
    __________
    dist_matrix: Distance matrix
    y: set of labels
    labels: boolean, if True show labels names


    Returns
    -------
    plot the distance matrix dist_matrix


    """
    fig = plt.figure(figsize=(5.1, 5))
    ax1 = fig.add_axes([0, .05, 1, 1])
    palette = plt.cm.jet
    #palette.set_under('b')
    #palette.set_over('r')
    ax1.matshow(dist_matrix,
                cmap=palette,
                #vmin=scoreatpercentile(X,2),
                #vmax=scoreatpercentile(X,98),
                aspect='auto')
    ax1.tick_params(length=0)

    if labels:
        ax1.xaxis.tick_top()
        ax1.set_xticklabels(y, rotation='vertical')
        ax1.set_yticklabels(y)
        ax1.set_xticks(np.arange(len(dist_matrix)) + 0.55, minor=False)
        ax1.set_yticks(np.arange(len(dist_matrix)) + 0.5, minor=False)
    else:
        ax1.set_xticks([])
        ax1.set_yticks([])

    if y is not None:
        ax2 = fig.add_axes([0, 0, 1, .04])

        cmap1 = plt.cm.get_cmap("RdYlBu", len(set(y)))
        #yl = np.array([i[0] for i in enumerate(y)])
        yl = np.array(y)
        set_y = list(set(y))
        for i in enumerate(set_y):
            yl[yl == i[1]] = i[0]
        yl = yl.astype(int)
        sl = list(set(yl))
        yl = yl.reshape(1, len(y))

        im2 = ax2.matshow(yl,
                          cmap=cmap1,
                          aspect='auto')

        ax2.set_xticks([])
        ax2.set_yticks([])

        ax2.tick_params(length=0)

        cax = fig.add_axes([1.04, 0.0, 0.05, len(set_y) * 0.05])
        cbar = fig.colorbar(im2, cax=cax, ticks=sl)
        #cbar.ax.set_yticks(sl, minor=False)
        cbar.ax.set_yticklabels(set_y)


def neighborhood_sort(dist_matrix, sigma, weight_matrix=None):
    """
    Parameters
    __________
    dist_matrix: np.array([n,n])
    sigma: float
    weight_matrix: None or np.array([n,n])


    Returns
    -------
    sorted_ind: np.array([n]) of integers
    sort_score: np.array([n]) of floats
    erg: float

    """
    mat_size = len(dist_matrix)
    if sigma < 2:
        if sigma < 0:
            print "Erro"
        sigma = 1

    if weight_matrix is None:
        weight_matrix = make_weight_matrix(sigma=sigma, n=len(dist_matrix))

    mismatch = np.dot(dist_matrix, weight_matrix)

    #encontra os indices e os valores minimos por linha
    idx_m = np.argmin(mismatch, axis=1)
    val_m = mismatch[np.arange(mat_size), idx_m]

    mx = max(val_m)
    #print 0.1*np.sign((mat_size/2. - idx_m + 1.))*val_m/mx
    sort_score = (idx_m + 1.
                  - 0.1 * np.sign((mat_size / 2. - idx_m + 1.)) * val_m / mx)
    sorted_ind = np.argsort(sort_score)
    #print np.argsort(idx_m)
    #print sorted_ind
    erg = np.trace(np.dot(dist_matrix[np.meshgrid(sorted_ind, sorted_ind)],
                          weight_matrix))
    return sorted_ind, sort_score, mismatch, erg


def make_weight_matrix(sigma, n):
    """
    Parameters
    __________
    sigma: float
    n: integer


    Returns
    -------
    weight_matrix: np.array([n.n])

    """
    [i, j] = np.meshgrid(range(n), range(n))
    w = np.exp(- (i - j) ** 2. / sigma / n)
    for i in range(10):
        w = w / np.reshape(np.sum(w, axis=1), (n, 1))
        w = w / np.sum(w, axis=0)
    w = 0.5 * (w + w.T)
    return w


def spin(dist_matrix, y, shuffle=False):
    """
    Parameters
    __________
    dist_matrix: np.array([n,n])
    y: np.array([n])
    shffle: boolean, if True distance_matrix will be shuffled


    Returns
    -------
    dist_matrix: np.array([n,n])
    y: np.array([n])
    """
    si = range(len(dist_matrix))
    if shuffle:
        np.random.shuffle(si)
    sigma = 2 ** 5.
    for j in range(6):
        w = make_weight_matrix(sigma, len(dist_matrix))
        out = []
        for i in range(20):
            y = y[si]
            dist_matrix = dist_matrix[np.meshgrid(si, si)]
            si, se, mm, e = neighborhood_sort(dist_matrix, sigma, w)
            if e in out:
                break
            else:
                out = out + [e]
            #print sigma, i, e
        sigma = sigma / 2.0
        #print e
    return dist_matrix, y
