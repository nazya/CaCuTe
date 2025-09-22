import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

from optmethods.optimizer import Optimizer


def ls_cubic_solver(x, g, H, M, it_max=100, epsilon=1e-8, loss=None):
    """
    Solve min_z <g, z-x> + 1/2<z-x, H(z-x)> + M/3 ||z-x||^3
    
    For explanation of Cauchy point, see "Gradient Descent 
        Efficiently Finds the Cubic-Regularized Non-Convex Newton Step"
        https://arxiv.org/pdf/1612.00547.pdf
    Other potential implementations can be found in paper
        "Adaptive cubic regularisation methods"
        https://people.maths.ox.ac.uk/cartis/papers/ARCpI.pdf
    """
    solver_it = 1
    try:
        newton_step = -np.linalg.solve(H, g)
    except:
        newton_step = -np.linalg.lstsq(H, g)[0]
    if M == 0:
        return x + newton_step, solver_it
    def cauchy_point(g, H, M):
        if la.norm(g) == 0 or M == 0:
            return 0 * g
        g_dir = g / la.norm(g)
        H_g_g = H @ g_dir @ g_dir
        R = -H_g_g / (2*M) + np.sqrt((H_g_g/M)**2/4 + la.norm(g)/M)
        return -R * g_dir
    
    def conv_criterion(s, r):
        """
        The convergence criterion is an increasing and concave function in r
        and it is equal to 0 only if r is the solution to the cubic problem
        """
        s_norm = la.norm(s)
        return 1/s_norm - 1/r
    
    # Solution s satisfies ||s|| >= Cauchy_radius
    r_min = la.norm(cauchy_point(g, H, M))
    
    if loss is not None:
        x_new = x + newton_step
        if loss.value(x) > loss.value(x_new):
            return x_new, solver_it
        
    r_max = la.norm(newton_step)
    if r_max - r_min < epsilon:
        return x + newton_step, solver_it
    id_matrix = np.eye(len(g))
    for _ in range(it_max):
        r_try = (r_min + r_max) / 2
        lam = r_try * M
        try:
            s_lam = -np.linalg.solve(H + lam*id_matrix, g)
        except:
            s_lam = -np.linalg.lstsq(H + lam*id_matrix, g)[0]
        solver_it += 1
        crit = conv_criterion(s_lam, r_try)
        if np.abs(crit) < epsilon:
            return x + s_lam, solver_it
        if crit < 0:
            r_min = r_try
        else:
            r_max = r_try
        if r_max - r_min < epsilon:
            break
    return x + s_lam, solver_it


class Cubic(Optimizer):
    """
    Newton method with cubic regularization for global convergence.
    The method was studied by Nesterov and Polyak in the following paper:
        "Cubic regularization of Newton method and its global performance"
        https://link.springer.com/article/10.1007/s10107-006-0706-8
    
    Arguments:
        reg_coef (float, optional): an estimate of the Hessian's Lipschitz constant
    """
    def __init__(self, reg_coef=None, solver_it_max=100, solver_eps=1e-8, cubic_solver=None, *args, **kwargs):
        # super(Cubic, self).__init__(*args, **kwargs)
        super().__init__(*args, **kwargs)
        self.reg_coef = reg_coef
        self.cubic_solver = cubic_solver
        self.solver_it = 0
        self.solver_it_max = solver_it_max
        self.solver_eps = solver_eps
        if reg_coef is None:
            self.reg_coef = self.loss.hessian_lipschitz
        if cubic_solver is None:
            self.cubic_solver = ls_cubic_solver
        self.H = self.reg_coef/2
        print(f"{self.reg_coef=}")
        
    def step(self):
        self.grad = self.loss.gradient(self.x)
        self.hess = self.loss.hessian(self.x)

        self.x, solver_it = self.cubic_solver(self.x, self.grad, self.hess, self.reg_coef/2, self.solver_it_max, self.solver_eps)
        
        # H = self.reg_coef/2
        # grad_norm = np.linalg.norm(self.grad)
        # self.identity_coef = (H * grad_norm)**0.5
        # # self.identity_coef = (grad_norm)**2
        # # self.x_old = copy.deepcopy(self.x)
        # # self.grad_old = copy.deepcopy(self.grad)
        # delta_x = -np.linalg.lstsq(self.hess + self.identity_coef*np.eye(self.loss.dim), self.grad)[0]
        # # delta_x = -np.linalg.solve(self.hess + self.identity_coef*np.eye(self.loss.dim), self.grad)
        # self.x += delta_x
        # solver_it = 0
        self.solver_it += solver_it
        
    def init_run(self, *args, **kwargs):
        super().init_run(*args, **kwargs)
        self.trace.solver_its = [0]
        
    def update_trace(self):
        super().update_trace()
        self.trace.solver_its.append(self.solver_it)



class AccCubic(Optimizer):
    """
    Newton method with cubic regularization for global convergence.
    The method was studied by Nesterov and Polyak in the following paper:
        "Cubic regularization of Newton method and its global performance"
        https://link.springer.com/article/10.1007/s10107-006-0706-8
    
    Arguments:
        reg_coef (float, optional): an estimate of the Hessian's Lipschitz constant
    """
    def __init__(self, reg_coef=None, solver_it_max=100, solver_eps=1e-8, cubic_solver=None, *args, **kwargs):
        # super(Cubic, self).__init__(*args, **kwargs)
        super().__init__(*args, **kwargs)
        self.reg_coef = reg_coef
        self.cubic_solver = cubic_solver
        self.solver_it = 0
        self.solver_it_max = solver_it_max
        self.solver_eps = solver_eps
        if reg_coef is None:
            self.reg_coef = self.loss.hessian_lipschitz
        if cubic_solver is None:
            self.cubic_solver = ls_cubic_solver
        
    def step(self):
        k = self.it
        if k > 1:
            self.v = self.x0 - self.s / np.sqrt( 12*(self.reg_coef/2) * np.linalg.norm(self.s))
        self.y = k/(k+3)*self.x + 3/(k+3)*self.v
        self.grad = self.loss.gradient(self.y)

        if k == 0:
            self.hess = self.loss.hessian(self.x)
            self.x, solver_it = self.cubic_solver(self.x, self.grad, self.hess, self.reg_coef, self.solver_it_max, self.solver_eps)
            self.solver_it += solver_it
            return
    
        self.hess = self.loss.hessian(self.y)
        self.x, solver_it = self.cubic_solver(self.y, self.grad, self.hess, self.reg_coef, self.solver_it_max, self.solver_eps)
        self.s += (k+1)*(k+2)/2 * self.loss.gradient(self.x)
        self.solver_it += solver_it
        
    def init_run(self, *args, **kwargs):
        super().init_run(*args, **kwargs)
        self.trace.solver_its = [0]
        #init
        self.x0 = self.x.copy()
        self.y = self.x.copy()
        self.v = self.x.copy()
        self.s = np.zeros_like(self.x)
        
    def update_trace(self):
        super().update_trace()
        self.trace.solver_its.append(self.solver_it)


