#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.linalg import solve_continuous_lyapunov,eigh, qr, svd
from scipy.sparse.linalg import spsolve

def balancing(A,B,C,trunctol):
    """
    This function should compute a (numerically) minimal, balanced system. For
    this, it takes input matrices A,B,C describing an LTI system as well as a
    threshold/truncation tolerance to set up a (compressed) state space
    transformation (as discussed in the lecture). No further truncation beyond
    the numerical cut-off due to low-rank representation of the Gramians as
    expected - it will be done explicitly outside this routine. The code uses
    the "lrcf" function defined below which, given a symmetric positive semi-
    definite matrix M, computes a Cholesky-like factorization of the form M=LL^T
    where L is a (full-rank) but potentially rectangular matrix.

    :param A: system matrix, given as array of float64 or sparse csr_matrix
    :param B: input matrix, given as array of float64 or sparse csr_matrix
    :param C: output matrix, given as array of float64 or sparse csr_matrix
    :param trunctol: relative tolerance used for numerical cut-off/rounding,
    strictly between zero and one
    :return: Ab,Bb,Cb (numerically balanced LTI description) and Sigma
    (vector of singular values of balanced system)
    """
    if not 0 < trunctol < 1:
        raise ValueError("trunctol must lie strictly between zero and one")
    # Step 1: controllability and observability Gramians.
    P, Q = lyapunov_gramians(A, B, C)
    if not np.any(P) or not np.any(Q):
        # The transfer function is identically zero.
        return np.empty((0, 0)), np.empty((0, B.shape[1])), np.empty((C.shape[0], 0)), np.empty(0)
    # Step 2 is already supplied by the exercise template.
    R = lrcf(P, trunctol) # compute Cholesky-like factorization of P
    L = lrcf(Q, trunctol) # compute Cholesky-like factorization of Q
    # Step 3: Hankel singular values, ordered from largest to smallest.
    U, Sigma, Vt = svd(L.T @ R, full_matrices=False)
    cutoff = max(trunctol, np.finfo(float).eps * max(L.shape[1], R.shape[1]))
    keep = Sigma > cutoff * Sigma[0]
    Sigma = Sigma[keep]
    U, V = U[:, keep], Vt[keep, :].T

    # Step 4: x_bal = T @ x, x ~= T_tilde @ x_bal; T @ T_tilde = I.
    sqrt_sigma = np.sqrt(Sigma)
    T = (U.T @ L.T) / sqrt_sigma[:, None]
    T_tilde = (R @ V) / sqrt_sigma[None, :]

    # Step 5: balanced realization. Selecting the first r states is done
    # outside this routine, as required by the exercise template.
    Ab = T @ (A @ T_tilde)
    Bb = T @ B
    Cb = C @ T_tilde
    return np.asarray(Ab), np.asarray(Bb), np.asarray(Cb), Sigma


def lyapunov_gramians(A, B, C):
    """Compute the two infinite-horizon Gramians of a real, stable LTI system.

    P: controllability Gramian, A P + P A.T + B B.T = 0.
    Q: observability Gramian, A.T Q + Q A + C.T C = 0.

    SciPy solves a X + X a.T = rhs, hence the negative right-hand sides.
    The dense solver is suitable for our modest finite-difference grid.
    Sparse inputs are converted to dense arrays only for these solves.
    This helper allows us to inspect step 1 independently of balancing.
    """
    A = A.toarray() if sp.issparse(A) else np.asarray(A)
    B = B.toarray() if sp.issparse(B) else np.asarray(B)
    C = C.toarray() if sp.issparse(C) else np.asarray(C)
    P = solve_continuous_lyapunov(A, -(B @ B.T))
    Q = solve_continuous_lyapunov(A.T, -(C.T @ C))
    # Remove round-off asymmetry before symmetric factorizations.
    P = 0.5 * (P + P.T)
    Q = 0.5 * (Q + Q.T)
    return P, Q

def hinf_norm(A, B, C, n_sample_points):
    """
    This function computes a numerical approximation to the H-infinity norm of
    the dynamical system [A,B,C,D] and its associated transfer function
    G(s)=C (sI-A)^{-1}B + D. For this, it replaces the continuous maximum
    problem ||G||_{Hinf}=sup_{iR} |G(s)|_2 by a maximization over two
    (logarithmically!) spaced intervals 1i[-10^5,-10^{-5}],1i[10^{-5},10^5] along
    the imaginary axis each consisting of 'n_sampling_points' points.

    :param A: system matrix, given as array of float64 or sparse csr_matrix
    :param B: input matrix, given as array of float64 or sparse csr_matrix
    :param C: output matrix, given as array of float64 or sparse csr_matrix
    :param n_sample_points: number of discrete (logarithmically spaced) grid points
    for maximization
    :return: (approximate) H-inifinity norm of [A,B,C,D]
    """

    # add your code here
    #
    #
    #
    #
    return hinf

def implicit_euler(A, B, tf, u, x0, n_time_steps):
    """
    This functions computes an approximate solution to an LTI system of the form
                d/dt x(t) = Ax(t) + Bu(t), x(0)=x0
    via the implicit Euler scheme with 'n_time_steps' (+1 initial condition). The
    input signal u is assumed to be handed over as a callable function, e.g.,
    u = lambda t: np.sin(2*np.pi*t)*10

    :param A: system matrix, given as array of float64 or sparse csr_matrix
    :param B: input matrix, given as array of float64 or sparse csr_matrix
    :param tf: final time of simulation over [0,t_f]
    :param u: callable function describing an input signal on [0,t_f]
    :param x0: initial value of the LTI system
    :param n_time_steps: number of time steps (excluding initial step) on (0,t_f]
    :return: solution matrix x=[x(0),x(t_1),...,x(t_f)] at discrete time snaphots
    """

    # Backward Euler: (I - dt*A) x[k+1] = x[k] + dt*B*u(t[k+1]).
    # Factor once, since A and the time step do not change.
    from scipy.sparse.linalg import splu

    if not isinstance(n_time_steps, (int, np.integer)) or n_time_steps <= 0:
        raise ValueError("n_time_steps must be a positive integer")
    if tf <= 0:
        raise ValueError("tf must be positive")

    A = sp.csc_matrix(A)
    B = B.toarray() if sp.issparse(B) else np.asarray(B)
    if B.ndim == 1:
        B = B[:, None]
    x0 = np.asarray(x0).reshape(-1)
    dt = tf / n_time_steps
    dtype = np.result_type(A.dtype, B.dtype, x0.dtype, float)
    solver = splu((sp.eye(A.shape[0], format='csc') - dt * A).astype(dtype))
    x = np.empty((A.shape[0], n_time_steps + 1), dtype=dtype)
    x[:, 0] = x0
    for k in range(n_time_steps):
        next_input = np.asarray(u((k + 1) * dt)).reshape(-1)
        x[:, k + 1] = solver.solve(x[:, k] + dt * (B @ next_input))
    return x

def lrcf(X,trunctol):
    DX, UX = eigh(0.5 * (X + X.T))
    DX = np.diag(DX)
    Q, R = qr(UX, mode='economic')
    tmp = R @ DX @ R.T
    tmp = 0.5 * (tmp + tmp.T)
    d, Utmp = eigh(tmp)
    Dtmp = np.diag(d)
    y = np.argsort(np.abs(d))[::-1]
    Utmp = Utmp[:, y]
    Dtmp = np.diag(d[y])
    index = np.where(np.diag(Dtmp) / Dtmp[0, 0] > trunctol)[0][-1]
    S = Q @ Utmp[:, :index+1] @ np.sqrt(Dtmp[:index+1, :index+1])
    return S

def fdm_2d_poisson(d):
    h = 1 / (d + 1)
    e = np.ones(d)
    T = sp.diags([e, -2*e, e], [-1, 0, 1], shape=(d, d))  # Tridiagonal matrix for 1D Laplacian
    T = T / h**2
    I = sp.eye(d)
    A = sp.kron(I, T) + sp.kron(T, I)
    xd = np.linspace(0, 1, d+2)
    xd = xd[1:-1]
    yd = xd
    X, Y = np.meshgrid(xd, yd)
    B = np.sin(3/2*X*np.pi) * np.sin(4/3*Y*np.pi)
    B = B.flatten()
    B = B[:,None]
    C = h**2 * np.ones((1,d**2))

    return A,B,C,X,Y

def mpl_fontsize(
        fontsize: int = None,
        bigger_axis_labels: bool = True,
        ):

    if fontsize is not None:
        plt.rcParams.update({"font.size": fontsize})

        # make legend font size smaller
        plt.rcParams.update({
            "legend.fontsize": fontsize - 6,
            })

        # bigger axis labels if needed
        if bigger_axis_labels:
            plt.rcParams.update({
                "axes.labelsize": fontsize + 4,
                "axes.titlesize": fontsize + 4,
                })

    return

def mpl_settings(
        figsize: tuple = (5.5,4),
        backend: str = None,
        latex_font: str = 'computer modern',
        fontsize: int = None,
        bigger_axis_labels: bool = True,
        dpi: int = 500,
        ) -> None:
    """
    sets matplotlib settings for latex

    :return: None
    """

    plt.rcParams['figure.dpi'] = dpi
    # default for paper: (5.5,4)
    plt.rcParams['figure.figsize'] = figsize
    plt.rc('text', usetex=True)

    preamble = '\n'.join([
                        r'\usepackage{amsmath,amssymb}',
                        r'\newcommand{\projnodes}{s_{\Pi}}',
                        r'\newcommand{\quadnodes}{s_Q}',
                        r'\newcommand{\error}{\mathcal{E}}',
                        r'\newcommand{\errorenergy}{\error_{\text{energy}}}',
                        r'\newcommand{\errorstate}{\error_{\text{state}}}',
                    ])

    plt.rc('text.latex', preamble=preamble)

    plt.rcParams.update({
            "pgf.texsystem": "pdflatex",
            "pgf.rcfonts": False,      # don't setup fonts from rc parameters
            "pgf.preamble": preamble,
            "savefig.transparent": True,
            })

    mpl_fontsize(fontsize=fontsize, bigger_axis_labels=bigger_axis_labels)

    if latex_font == 'times':
        plt.rc('font',**{'family':'serif','serif':['Times']})
    elif latex_font == 'computer modern':
        plt.rc('font',**{'family':'serif'})

    plt.rc('axes.formatter', useoffset=False)
    # plt.rcParams['savefig.transparent'] = True

    if backend is not None:
        matplotlib.use(backend)
    if backend == 'macosx':
        plt.rcParams['figure.dpi'] = 140

    return
