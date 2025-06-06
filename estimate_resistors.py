import numpy as np
from scipy.optimize import least_squares
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# ----------  low-level helpers  ------------------------------------------------
def _build_laplacian(R):
    """Return Laplacian matrix L for an mxn cross-bar with resistances R (Ω)."""
    m, n = R.shape
    N = m + n
    L = np.zeros((N, N))
    for i in range(m):
        for j in range(n):
            g = 1.0 / R[i, j]          # conductance
            ri, cj = i, m + j
            L[ri, ri] += g
            L[cj, cj] += g
            L[ri, cj] -= g
            L[cj, ri] -= g
    return L


def _effective_resistances(R):
    """Return matrix of equivalent resistances between every row and column rail."""
    m, n = R.shape
    Lp = np.linalg.pinv(_build_laplacian(R))
    rows = np.arange(m)[:, None]                # shape (m,1)
    cols = m + np.arange(n)[None, :]            # shape (1,n)
    return Lp[rows, rows] + Lp[cols, cols] - 2 * Lp[rows, cols]


def _effective_resistances_sparse_vec(R):
    """
    R : (m, n) ndarray of individual resistances (Ω).
    Returns an (m, n) ndarray of equivalent row-to-column resistances (Ω).

    Implementation details
    ----------------------
    * last rail node (index N-1) is grounded
    * build Laplacian in CSC → no SparseEfficiencyWarning
    * one sparse LU, many RHS
    * no Python loops
    """
    m, n = R.shape
    N = m + n
    ground = N - 1                         # ground = last column rail

    # ---------- build Laplacian L in one shot (CSC) -------------------------
    g   = 1.0 / R.ravel(order='C')         # conductances, length m*n
    ri  = np.repeat(np.arange(m), n)
    cj  = m + np.tile(np.arange(n), m)
    data = np.concatenate([ g,  g, -g, -g ])
    rows = np.concatenate([ ri, cj, ri, cj ])
    cols = np.concatenate([ ri, cj, cj, ri ])
    L = sp.csc_matrix((data, (rows, cols)), shape=(N, N))

    # ---------- remove ground row & col, factor once ------------------------
    keep = np.arange(N-1)                  # nodes 0 … N-2 stay
    Lg   = L[keep][:, keep]
    lu   = spla.splu(Lg)                   # sparse LU / Cholesky

    # ---------- potentials for unit injections at every kept node ----------
    V = lu.solve(np.eye(N-1))              # shape (N-1, N-1)
    diag = np.diag(V)                      # V_pp  (length N-1)

    # ---------- assemble R_eq matrix ---------------------------------------
    Reff = np.empty((m, n))
    rows_idx   = np.arange(m)              # 0 … m-1  (row rails)
    cols_full  = m + np.arange(n)          # m … N-1  (column rails)

    # mask: which columns are *not* the ground column
    mask        = cols_full != ground
    cols_kept   = cols_full[mask]
    cols_red    = cols_kept                # kept indices are unchanged (ground is last)

    # R_eq for “normal” columns (vectorised broadcast)
    Reff[:, mask] = (
        diag[rows_idx][:, None]            # V_pp   term
        + diag[cols_red][None, :]          # V_qq   term
        - 2 * V[rows_idx[:, None], cols_red[None, :]]
    )

    # R_eq to the ground column: simply V_pp (see derivation in answer)
    Reff[:, ~mask] = diag[rows_idx][:, None]

    return Reff



# ----------  main solver  ------------------------------------------------------
def estimate_resistors(R_eq,
                       init="uniform",
                       tol=1e-9,
                       max_iter=100,
                       verbose=False):
    """
    Estimate individual resistor values in an mxn cross-bar.

    Parameters
    ----------
    R_eq : ndarray, shape (m, n)
        Measured equivalent resistances (Ω) between row and column rails.
    init : {"uniform", "diag", "random", ndarray}, optional
        Initial guess (Ω).  *uniform* uses the mean of R_eq,
        *diag* uses the diagonal of R_eq (min(m, n) entries),
        *random* jitters the uniform guess by ±10 %.
        You can also pass an (m, n) array for a custom guess.
    tol : float, optional
        Termination tolerance on the RMS residual.
    max_iter : int, optional
        Maximum iterations for the optimiser.
    verbose : bool, optional
        If True, prints optimiser progress.

    Returns
    -------
    R_est : ndarray, shape (m, n)
        Estimated individual resistances (Ω).
    """

    if (R_eq == 0).any():
        raise ValueError('Cannot handle zero resistances')

    m, n = R_eq.shape
    R_eq = np.asarray(R_eq, dtype=float)

    # --- initial guess ---------------------------------------------------------
    if isinstance(init, str):
        if init == "uniform":
            R0 = np.full((m, n), R_eq.mean())
        elif init == "diag":
            d = np.diag(R_eq) if m == n else np.diag(R_eq, k=0)
            R0 = np.full((m, n), d.mean())
        elif init == "random":
            base = R_eq.mean()
            R0 = base * (1 + 0.1 * np.random.uniform(-1, 1, size=(m, n)))
        else:
            raise ValueError("init must be 'uniform', 'diag', 'random', or an array")
    else:
        R0 = np.asarray(init, dtype=float)
        if R0.shape != (m, n):
            raise ValueError("custom init must have shape (m, n)")

    # we work in log-conductance space so variables stay positive
    x0 = np.log(1.0 / R0).ravel()          # x = log(g)

    # --- residual function -----------------------------------------------------
    def _residual(x):
        g = np.exp(x).reshape(m, n)        # conductances
        R_pred = _effective_resistances(1.0 / g)   # predicted equivalents
        return (R_pred - R_eq).ravel()

    # --- optimise --------------------------------------------------------------
    result = least_squares(_residual,
                           x0,
                           jac='2-point',
                           method='trf',
                           xtol=tol,
                           ftol=tol,
                           gtol=tol,
                           max_nfev=max_iter,
                           verbose=2 if verbose else 0)

    if not result.success and not verbose:
        # raise RuntimeError(f"Solver failed: {result.message}")
        print(f"Solver failed: {result.message}")
        return None

    R_est = 1.0 / np.exp(result.x.reshape(m, n))   # back to Ω
    return R_est

def estimate_resistors_fast(R_eq,
                            init="uniform",
                            tol=1e-9,
                            max_iter=300,
                            sparse=True,
                            verbose=False):
    """
    Faster version: vectorised residual, sparse Laplacian option.
    """
    m, n = R_eq.shape
    R_eq = np.asarray(R_eq, float)

    # ----- initial guess (unchanged from previous version) -----
    if isinstance(init, str):
        if init == "uniform":
            R0 = np.full((m, n), R_eq.mean())
        elif init == "diag":
            R0 = np.full((m, n), np.diag(R_eq).mean())
        elif init == "random":
            R0 = R_eq.mean() * (1 + 0.1 * np.random.uniform(-1, 1, (m, n)))
        else:
            raise ValueError("init must be 'uniform', 'diag', 'random', or an array")
    else:                       # custom array
        R0 = np.asarray(init, float)
    x0 = np.log(1.0 / R0).ravel()

    # pick fast residual
    _Reff = _effective_resistances_sparse_vec if sparse else _effective_resistances

    def _residual(x):
        g = np.exp(x).reshape(m, n)
        return (_Reff(1.0 / g) - R_eq).ravel()


    result = least_squares(_residual, x0,
                        jac='2-point', method='trf',
                        xtol=tol, ftol=tol, gtol=tol,
                        max_nfev=max_iter,
                        verbose=2 if verbose else 0)

    if not result.success and not verbose:
        raise RuntimeError(f"Solver failed: {result.message}")

    return 1.0 / np.exp(result.x.reshape(m, n))

