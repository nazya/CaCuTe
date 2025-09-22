import scipy
import numpy as np
import warnings

from sklearn.utils.extmath import row_norms

from .loss_oracle import Oracle


class LogSumExp(Oracle):
    """
    Logarithm of the sum of exponentials plus two optional quadratic terms:
        log(sum_{i=1}^n exp(<a_i, x>-b_i)) + 1/2*||Ax - b||^2 + l2/2*||x||^2
    See, for instance,
        https://arxiv.org/pdf/2002.00657.pdf
        https://arxiv.org/pdf/2002.09403.pdf
    for examples of using the objective to benchmark second-order methods.
    
    Due to the potential under- and overflow, log-sum-exp and softmax
    functions might be unstable. This implementation has not been tested
    for stability and an alternative choice of functions may lead to
    more precise results. See
        https://academic.oup.com/imajna/advance-article/doi/10.1093/imanum/draa038/5893596
    for a discussion of possible ways to increase stability.
    
    Sparse matrices are currently not supported as it is not clear if it is relevant to practice.
    
    Arguments:
        max_smoothing (float, optional): the smoothing constant of the log-sum-exp term
        least_squares_term (bool, optional): add term 0.5*||Ax-b||^2 to the objective (default: False)
    """
    
    def __init__(self, max_smoothing=1, least_squares_term=False, A=None, b=None, n=None, dim=None, store_mat_vec_prod=True, store_softmax=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_smoothing = max_smoothing
        self.least_squares_term = least_squares_term
        self.A = A
        self.b = np.asarray(b)
        if b is None:
            self.b = self.rng.normal(-1, 1, size=n)
        if A is None:
            self.A = self.rng.uniform(-1, 1, size=(n, dim))
            # for the first gradient computation, we have to set 
            # the values of self.store_mat_vec_prod and self.store_softmax
            self.store_mat_vec_prod = False
            self.store_softmax = False
            self.A -= self.gradient(np.zeros(dim))
            self.value(np.zeros(dim))
        self.store_mat_vec_prod = store_mat_vec_prod
        self.store_softmax = store_softmax
        
        self.n, self.dim = self.A.shape
        self.x_last_mv = 0.
        self.x_last_soft = 0.
        self._mat_vec_prod = np.zeros(self.n)
    
    def _value(self, x):
        Ax = self.mat_vec_product(x)
        regularization = 0
        if self.l2 != 0:
            regularization = self.l2/2 * self.norm(x)**2
        if self.least_squares_term:
            regularization += 1/2 * np.linalg.norm(Ax)**2
        return self.max_smoothing*scipy.special.logsumexp(((Ax-self.b)/self.max_smoothing)) + regularization
    
    def gradient(self, x):
        Ax = self.mat_vec_product(x)
        softmax = self.softmax(Ax=Ax)
        if self.least_squares_term:
            grad = (softmax + Ax) @ self.A
        else:
            grad = softmax @ self.A
            
        if self.l2 == 0:
            return grad
        return grad + self.l2 * x
    
    def hessian(self, x):
        Ax = self.mat_vec_product(x)
        softmax = self.softmax(x=x, Ax=Ax)
        hess1 = self.A.T * (softmax/self.max_smoothing) @ self.A
        grad = softmax @ self.A
        hess2 = -np.outer(grad, grad) / self.max_smoothing
        return hess1 + hess2 + self.l2 * np.eye(self.dim)

    def gHg(self, x):
        def _dense1d(v):
            # import scipy.sparse
            if scipy.sparse.issparse(v):
                return np.asarray(v.toarray()).ravel()
            return np.asarray(v).ravel()

        tau = float(self.max_smoothing)

        # Forward pass
        Ax = self.mat_vec_product(x)              # (n,)
        p  = self.softmax(x=x, Ax=Ax)             # (n,)

        # Gradient at x
        g = _dense1d(self.gradient(x))            # (d,)

        # s = A g
        s = _dense1d(self.mat_vec_product(g))     # (n,)

        # Core term: A^T[(Diag(p) - p p^T) s] / tau
        ps   = float(p @ s)                       # scalar
        inside = (p * s - ps * p) / tau           # (n,)

        # Add least-squares Hessian contribution: + A^T A g = A^T s
        if getattr(self, "least_squares_term", False):
            inside = inside + s                   # still (n,)

        # Map back with A^T (using row-vector @ A -> A^T * vector)
        Hg = _dense1d(inside @ self.A)            # (d,)

        # L2 regularizer Hessian: + λ I g
        if getattr(self, "l2", 0.0):
            Hg = Hg + self.l2 * g

        return g, Hg


    def GgHg(self, x):
        def _dense1d(x):
            """Return a 1-D dense numpy array regardless of sparse/dense input."""
            if scipy.sparse.issparse(x):
                return np.asarray(x.toarray()).ravel()
            return np.asarray(x).ravel()
        tau  = float(self.max_smoothing)
        Ax = self.mat_vec_product(x)
        p  = self.softmax(x=x, Ax=Ax)               # (n,)
        g  = _dense1d(self.gradient(x))             # (d,)
        s  = _dense1d(self.mat_vec_product(g))      # s = A g, (n,)

        val = (p @ (s*s) - (p @ s)**2) / tau          # logsumexp part
        if getattr(self, "least_squares_term", False):
            val += float(s @ s)                     # + ||A g||^2
        if getattr(self, "l2", 0.0):
            val += float(g @ g) * self.l2           # + λ||g||^2
        return g, float(val)

    def GeHe(self, x):
        """
        Returns (g, eHe) where g = gradient(x) and e = g / ||g||.
        eHe is computed for the Hessian of the log-sum-exp (+ optional LS and L2 terms).
        """
        import numpy as np
        import scipy.sparse

        def _dense1d(z):
            """Return a 1-D dense numpy array regardless of sparse/dense input."""
            if scipy.sparse.issparse(z):
                return np.asarray(z.toarray()).ravel()
            return np.asarray(z).ravel()

        tau = float(self.max_smoothing)
        Ax  = self.mat_vec_product(x)
        p   = self.softmax(x=x, Ax=Ax)          # (n,)

        g   = _dense1d(self.gradient(x))        # (d,)
        gn2 = float(np.dot(g, g))
        if not np.isfinite(gn2) or gn2 == 0.0:
            return g, float("nan")              # e undefined

        # Use e = g / ||g|| via scaling: s_e = A e = (A g) / ||g||
        s = _dense1d(self.mat_vec_product(g)) / np.sqrt(gn2)   # (n,)

        # logsumexp Hessian part along e: e^T A^T[(Diag(p) - p p^T)/tau] A e
        val = (p @ (s * s) - (p @ s) ** 2) / tau

        # Optional + (A^T A) along e for least squares term
        if getattr(self, "least_squares_term", False):
            val += float(s @ s)                 # ||A e||^2

        # Optional + λ I along e for L2 term (||e|| = 1)
        l2 = float(getattr(self, "l2", 0.0) or 0.0)
        if l2:
            val += l2

        return g, float(val)


    
    def stochastic_hessian(self, x, idx=None, batch_size=1, replace=False, normalization=None):
        pass
    
    def mat_vec_product(self, x):
        if self.store_mat_vec_prod and self.is_equal(x, self.x_last_mv):
            return self._mat_vec_prod
        
        Ax = self.A @ x
        if self.store_mat_vec_prod:
            self._mat_vec_prod = Ax
            self.x_last_mv = x.copy()
        return Ax
    
    def softmax(self, x=None, Ax=None):
        if x is None and Ax is None:
            raise ValueError("Either x or Ax must be provided to compute softmax.")
        if self.store_softmax and self.is_equal(x, self.x_last_soft):
            return self._softmax
        if Ax is None:
            Ax = self.mat_vec_product(x)
        
        softmax = scipy.special.softmax((Ax-self.b) / self.max_smoothing)
        if self.store_softmax and x is not None:
            self._softmax = softmax
            self.x_last_soft = x.copy()
        return softmax
    
    def hess_vec_prod(self, x, v, grad_dif=False, eps=None):
        pass
        
    @property
    def smoothness(self):
        if self._smoothness is not None:
            return self._smoothness
        matrix_coef = 1 + self.least_squares_term
        if self.dim > 20000 and self.n > 20000:
            warnings.warn("The matrix is too large to estimate the smoothness constant, so Frobeniius estimate is used instead.")
            self._smoothness = matrix_coef*np.linalg.norm(self.A, ord='fro')**2 + self.l2
        else:
            sing_val_max = scipy.sparse.linalg.svds(self.A, k=1, return_singular_vectors=False)[0]
            self._smoothness = matrix_coef*sing_val_max**2 + self.l2
        return self._smoothness
    
    @property
    def hessian_lipschitz(self):
        if self._hessian_lipschitz is None:
            row_norms = np.linalg.norm(self.A, axis=1)
            max_row_norm = np.max(row_norms)
            self._hessian_lipschitz = 2 * max_row_norm / self.max_smoothing * self.smoothness
        return self._hessian_lipschitz
    
    @staticmethod
    def norm(x):
        return np.linalg.norm(x)
    
    @staticmethod
    def inner_prod(x, y):
        return x @ y
    
    @staticmethod
    def outer_prod(x, y):
        return np.outer(x, y)
    
    @staticmethod
    def is_equal(x, y):
        return np.array_equal(x, y)
