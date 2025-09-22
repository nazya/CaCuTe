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



class CaCuN(Optimizer):
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
        
    def step(self):
        self.grad = self.loss.gradient(self.x)
        
        # H_hat = (self.grad.dot(self.hess.dot(self.grad)))**2 / np.linalg.norm(self.grad)
        H_hat = (self.reg_coef/2) * 3/4
        condition = self.loss.value(self.x - self.grad/np.sqrt(H_hat*np.linalg.norm(self.grad))) <= \
            self.loss.value(self.x)  - (2/3)**(3/2) * np.linalg.norm(self.grad)**1.5 / np.sqrt(2*self.H)

        # print(f"{condition=}")

        if condition:
            self.x = self.x - self.grad/np.sqrt(H_hat*np.linalg.norm(self.grad))
            solver_it = 0
        else:      
            self.hess = self.loss.hessian(self.x)
            self.x, solver_it = self.cubic_solver(self.x, self.grad, self.hess, self.reg_coef/2, self.solver_it_max, self.solver_eps)

        self.solver_it += solver_it
        
    def init_run(self, *args, **kwargs):
        super().init_run(*args, **kwargs)
        self.trace.solver_its = [0]
        
    def update_trace(self):
        super().update_trace()
        self.trace.solver_its.append(self.solver_it)






class AccCaCuN(Optimizer):
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
            H_hat = (self.reg_coef/2)
            condition = self.loss.value(self.x - self.grad/np.sqrt(H_hat*np.linalg.norm(self.grad))) <= \
                self.loss.value(self.x)  - np.sqrt(2)/3 * np.linalg.norm(self.grad)**1.5 / np.sqrt(H_hat)
            if condition:
                self.x = self.x - self.grad/np.sqrt(H_hat*np.linalg.norm(self.grad))
                solver_it = 0
            else:      
                self.hess = self.loss.hessian(self.x)
                self.x, solver_it = self.cubic_solver(self.x, self.grad, self.hess, self.reg_coef, self.solver_it_max, self.solver_eps)
                self.solver_it += solver_it
            return
        

        H_hat = (self.reg_coef/2)*2
        condition = self.loss.value(self.y - self.grad/np.sqrt(H_hat*np.linalg.norm(self.grad))) <= \
            self.loss.value(self.y)  - 1/np.sqrt(3) * np.linalg.norm(self.grad)**1.5 / np.sqrt(H_hat)

        # print(f"{condition=}")

        if condition:
            self.x = self.y - self.grad/np.sqrt(H_hat*np.linalg.norm(self.grad))
            # solver_it = 0
            self.s += (k+1)*(k+2)/2 * self.grad
        else: 
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








# class SuperCubic(Optimizer):
#     """
#     Newton method with cubic regularization for global convergence.
#     The method was studied by Nesterov and Polyak in the following paper:
#         "Cubic regularization of Newton method and its global performance"
#         https://link.springer.com/article/10.1007/s10107-006-0706-8
    
#     Arguments:
#         reg_coef (float, optional): an estimate of the Hessian's Lipschitz constant
#     """
#     def __init__(self, reg_coef=None, solver_it_max=100, solver_eps=1e-8, cubic_solver=None, *args, **kwargs):
#         # super(Cubic, self).__init__(*args, **kwargs)
#         super().__init__(*args, **kwargs)
#         self.reg_coef = reg_coef
#         self.cubic_solver = cubic_solver
#         self.solver_it = 0
#         self.solver_it_max = solver_it_max
#         self.solver_eps = solver_eps
#         if reg_coef is None:
#             self.reg_coef = self.loss.hessian_lipschitz
        
#         self.H = self.reg_coef/2

#         self.fnew =1
#         self.fx =2

#         # self.reg = None
#         # self.norm = None
        
#     def step(self):
#         self.grad = self.loss.gradient(self.x)
        
#         H_hat = self.reg_coef/2
#         # fnew = self.loss.value(self.x - self.grad/np.sqrt(H_hat*np.linalg.norm(self.grad)))
#         # fx = self.loss.value(self.x)
#         # condition = fnew <= \
#         #     fx  - 1.9 * np.linalg.norm(self.grad)**1.5 / 3 / np.sqrt(H_hat)

#         # if condition:
#         #     if fnew >= fx:
#         #         return fx

#         #     self.x = self.x - self.grad/np.sqrt(H_hat*np.linalg.norm(self.grad))
#         #     solver_it = 0
#         # else:      
#         #     self.hess = self.loss.hessian(self.x)
#         #     H = self.H
#         #     grad_norm = np.linalg.norm(self.grad)
#         #     self.identity_coef = (H * grad_norm)**0.5
#         #     self.identity_coef = (grad_norm)**2
#         #     # self.x_old = copy.deepcopy(self.x)
#         #     # self.grad_old = copy.deepcopy(self.grad)
#         #     # delta_x = -np.linalg.solve(self.hess + self.identity_coef*np.eye(self.loss.dim), self.grad)
#         #     delta_x = -np.linalg.lstsq(self.hess + self.identity_coef*np.eye(self.loss.dim), self.grad)[0]
#         #     self.x += delta_x
#         #     solver_it = 0

#         grad_norm = np.linalg.norm(self.grad)
#         if self.fnew >= self.fx or grad_norm == 0.:
#             return self.fx

#         # self.identity_coef = (H * grad_norm)**p

#         self.fx = self.loss.value(self.x)
          
#         self.hess = self.loss.hessian(self.x)
#         H = self.H
#         self.identity_coef = (grad_norm)**2
#         # if self.identity_coef < 1e-180: # for covtype
#         if self.identity_coef < 1e-18:
#             # print(f"{(grad_norm)**2=}")
#             self.identity_coef = min((H * grad_norm)**0.5, grad_norm)
#         # self.identity_coef = (grad_norm)
#         # self.x_old = copy.deepcopy(self.x)
#         # self.grad_old = copy.deepcopy(self.grad)
#         delta_x = -np.linalg.solve(self.hess + self.identity_coef*np.eye(self.loss.dim), self.grad)
#         # delta_x = -np.linalg.lstsq(self.hess + self.identity_coef*np.eye(self.loss.dim), self.grad)[0]
#         self.x += delta_x

#         self.fnew = self.loss.value(self.x)

#         # if self.fnew >= self.fx:
#         #     return self.fx

        
#         solver_it = 0

#         self.solver_it += solver_it
        
#     def init_run(self, *args, **kwargs):
#         super().init_run(*args, **kwargs)
#         self.trace.solver_its = [0]
        
#     def update_trace(self):
#         super().update_trace()
#         self.trace.solver_its.append(self.solver_it)



from optmethods.line_search import RegNewtonLS
from optmethods.optimizer import Optimizer


def empirical_hess_lip(grad, grad_old, hess, x, x_old, loss):
    grad_error = grad - grad_old - hess@(x - x_old)
    r2 = loss.norm(x - x_old)**2
    if r2 > 0:
        return 2 * loss.norm(grad_error) / r2
    return np.finfo(float).eps


class SuperCubic(Optimizer):
    """
    Regularized Newton algorithm for second-order minimization.
    By default returns the Regularized Newton method from paper
        "Regularized Newton Method with Global O(1/k^2) Convergence"
        https://arxiv.org/abs/2112.02089
    
    Arguments:
        loss (optmethods.loss.Oracle): loss oracle
        identity_coef (float, optional): initial regularization coefficient (default: None)
        hess_lip (float, optional): estimate for the Hessian Lipschitz constant. 
            If not provided, it is estimated or a small value is used (default: None)
        adaptive (bool, optional): use decreasing regularization based on either empirical Hessian-Lipschitz constant
            or a line-search procedure
        line_search (optmethods.LineSearch, optional): a callable line search. If None, line search is intialized
            automatically as an instance of RegNewtonLS (default: None)
        use_line_search (bool, optional): use line search to estimate the Lipschitz constan of the Hessian.
            If adaptive is True, line search will be non-monotonic and regularization may decrease (default: False)
        backtracking (float, optional): backtracking constant for the line search if line_search is None and
            use_line_search is True (default: 0.5)
    """
    def __init__(self, loss, identity_coef=None, hess_lip=None, adaptive=True, line_search=None,
                 use_line_search=True, backtracking=0.5, *args, **kwargs):
        if hess_lip is None:
            hess_lip = loss.hessian_lipschitz
            if loss.hessian_lipschitz is None:
                hess_lip = 1e-5
                warnings.warn(f"No estimate of Hessian-Lipschitzness is given, so a small value {hess_lip} is used as a heuristic.")
        self.hess_lip = hess_lip
        
        self.H = hess_lip / 2
            
        if use_line_search and line_search is None:
            if adaptive:
                line_search = RegNewtonLS(decrease_reg=adaptive, backtracking=backtracking, H0=self.H)
            else:
                # use a more optimistic initial estimate since hess_lip is often too optimistic
                line_search = RegNewtonLS(decrease_reg=adaptive, backtracking=backtracking, H0=self.H / 100)
        super().__init__(loss=loss, line_search=line_search, *args, **kwargs)
        
        self.identity_coef = identity_coef
        self.adaptive = adaptive
        self.use_line_search = use_line_search
        
    def step(self):
        self.grad = self.loss.gradient(self.x)
        if self.adaptive and self.hess is not None and not self.use_line_search:
            self.hess_lip /= 2
            empirical_lip = empirical_hess_lip(self.grad, self.grad_old, self.hess, self.x, self.x_old, self.loss)
            self.hess_lip = max(self.hess_lip, empirical_lip)
        self.hess = self.loss.hessian(self.x)
        
        if self.use_line_search:
            self.fx = self.loss.value(self.x)
            self.x = self.line_search(self.x, self.grad, self.hess)
            self.fnew = self.loss.value(self.x)

            if self.fnew >= self.fx:
                return self.fx

        else:
            if self.adaptive:
                self.H = self.hess_lip / 2
            grad_norm = self.loss.norm(self.grad)
            self.identity_coef = (self.H * grad_norm)**0.5
            self.x_old = copy.deepcopy(self.x)
            self.grad_old = copy.deepcopy(self.grad)
            delta_x = -np.linalg.solve(self.hess + self.identity_coef*np.eye(self.loss.dim), self.grad)
            self.x += delta_x
        
    def init_run(self, *args, **kwargs):
        super().init_run(*args, **kwargs)
        self.x_old = None
        self.hess = None
        self.trace.lrs = []
        
    def update_trace(self, *args, **kwargs):
        super().update_trace(*args, **kwargs)
        if not self.use_line_search:
            self.trace.lrs.append(1 / self.identity_coef)


# class CubicFastB(Optimizer):
#     """
#     Newton method with cubic regularization for global convergence.
#     The method was studied by Nesterov and Polyak in the following paper:
#         "Cubic regularization of Newton method and its global performance"
#         https://link.springer.com/article/10.1007/s10107-006-0706-8
    
#     Arguments:
#         reg_coef (float, optional): an estimate of the Hessian's Lipschitz constant
#     """
#     def __init__(self, reg_coef=None, solver_it_max=100, solver_eps=1e-8, cubic_solver=None, *args, **kwargs):
#         # super(Cubic, self).__init__(*args, **kwargs)
#         super().__init__(*args, **kwargs)
#         self.reg_coef = reg_coef
#         self.cubic_solver = cubic_solver
#         self.solver_it = 0
#         self.solver_it_max = solver_it_max
#         self.solver_eps = solver_eps
#         if reg_coef is None:
#             self.reg_coef = self.loss.hessian_lipschitz
#         if cubic_solver is None:
#             self.cubic_solver = ls_cubic_solver
#         self.H = self.reg_coef/2
        
#     def step(self):
#         self.grad = self.loss.gradient(self.x)
#         self.hess = self.loss.hessian(self.x)



        
        
#         # linesearch 
#         H_hat = min(2**0.5 * (self.grad.dot(self.hess.dot(self.grad)))**2 / (np.linalg.norm(self.grad)**5), self.H)
#         condition = self.loss.value(self.x - self.grad/np.sqrt(H_hat*np.linalg.norm(self.grad))) <= \
#             self.loss.value(self.x) - np.linalg.norm(self.grad)**1.5 / 6 / np.sqrt(H_hat)

#         while not condition:
#             H_hat *= 2
#             if H_hat > self.H:
#                 H_hat = self.H
#                 condition = self.loss.value(self.x - self.grad/np.sqrt(H_hat*np.linalg.norm(self.grad))) <= \
#                     self.loss.value(self.x)  - np.linalg.norm(self.grad)**1.5 / 6 / np.sqrt(H_hat)
#                 break
#             condition = self.loss.value(self.x - self.grad/np.sqrt(H_hat*np.linalg.norm(self.grad))) <= \
#                 self.loss.value(self.x)  - np.linalg.norm(self.grad)**1.5 / 6 / np.sqrt(H_hat)


#         if condition:
#             self.x = self.x - self.grad/np.sqrt(H_hat*np.linalg.norm(self.grad))
#             solver_it = 0
#         else:      
#             print(f"{condition=}")
#             # self.hess = self.loss.hessian(self.x)
#             self.x, solver_it = self.cubic_solver(self.x, self.grad, self.hess, self.reg_coef/2, self.solver_it_max, self.solver_eps)
#             # H = self.H
#             # H = self.reg_coef/2
#             # grad_norm = np.linalg.norm(self.grad)
#             # self.identity_coef = (H * grad_norm)**0.5
#             # self.identity_coef = (grad_norm)**2
#             # delta_x = -np.linalg.solve(self.hess + self.identity_coef*np.eye(self.loss.dim), self.grad)
#             # delta_x = -np.linalg.lstsq(self.hess + self.identity_coef*np.eye(self.loss.dim), self.grad)[0]
#             # self.x += delta_x
#             # solver_it = 0

#         self.solver_it += solver_it
        
#     def init_run(self, *args, **kwargs):
#         super().init_run(*args, **kwargs)
#         self.trace.solver_its = [0]
        
#     def update_trace(self):
#         super().update_trace()
#         self.trace.solver_its.append(self.solver_it)



# class AccCubicFastB(Optimizer):
#     """
#     Newton method with cubic regularization for global convergence.
#     The method was studied by Nesterov and Polyak in the following paper:
#         "Cubic regularization of Newton method and its global performance"
#         https://link.springer.com/article/10.1007/s10107-006-0706-8
    
#     Arguments:
#         reg_coef (float, optional): an estimate of the Hessian's Lipschitz constant
#     """
#     def __init__(self, reg_coef=None, solver_it_max=100, solver_eps=1e-8, cubic_solver=None, *args, **kwargs):
#         # super(Cubic, self).__init__(*args, **kwargs)
#         super().__init__(*args, **kwargs)
#         self.reg_coef = reg_coef
#         self.cubic_solver = cubic_solver
#         self.solver_it = 0
#         self.solver_it_max = solver_it_max
#         self.solver_eps = solver_eps
#         if reg_coef is None:
#             self.reg_coef = self.loss.hessian_lipschitz
#         if cubic_solver is None:
#             self.cubic_solver = ls_cubic_solver 
        
#     def step(self):
#         k = self.it
#         if k > 1:
#             self.v = self.x0 - self.s / np.sqrt( 2*(self.reg_coef/2) * np.linalg.norm(self.s))
#         self.y = k/(k+3)*self.x + 3/(k+3)*self.v



#         self.grad = self.loss.gradient(self.y)

#         # condition = self.loss.value(self.y - self.grad/np.sqrt(H_hat*np.linalg.norm(self.grad))) <= \
#         #         self.loss.value(self.y)  - np.linalg.norm(self.grad)**1.5 / 6 / np.sqrt(H_hat)
#         # #check self.N = 0 with FO backtracking
#         # while True:



#         self.hess = self.loss.hessian(self.y)

#         if k == 0:
#             # H_hat = (self.grad.dot(self.hess.dot(self.grad)))**2 / (np.linalg.norm(self.grad)**5)
#             H_hat = min((self.grad.dot(self.hess.dot(self.grad)))**2 / (np.linalg.norm(self.grad)**5), (self.reg_coef/2)/4)
#             condition = self.loss.value(self.y - self.grad/np.sqrt(H_hat*np.linalg.norm(self.grad))) <= \
#                 self.loss.value(self.y)  - np.linalg.norm(self.grad)**1.5 / 6 / np.sqrt(H_hat)
#             while not condition:
#                 H_hat *= 2
#                 if H_hat >= (self.reg_coef/2)/ 4:
#                     H_hat = (self.reg_coef/2) / 4
#                     condition = self.loss.value(self.y - self.grad/np.sqrt(H_hat*np.linalg.norm(self.grad))) <= \
#                         self.loss.value(self.y)  - np.linalg.norm(self.grad)**1.5 / 6 / np.sqrt(H_hat)
#                     break
#                 condition = self.loss.value(self.y - self.grad/np.sqrt(H_hat*np.linalg.norm(self.grad))) <= \
#                     self.loss.value(self.y)  - np.linalg.norm(self.grad)**1.5 / 6 / np.sqrt(H_hat)
            
#             if condition:
#                 self.x = self.y - self.grad/np.sqrt(H_hat*np.linalg.norm(self.grad))
#                 solver_it = 0
#             else:      
#                 self.x, solver_it = self.cubic_solver(self.y, self.grad, self.hess, self.reg_coef, self.solver_it_max, self.solver_eps)
#                 self.solver_it += solver_it
#             return
        

#         H_hat = (self.grad.dot(self.hess.dot(self.grad)))**2 / (np.linalg.norm(self.grad)**5) * 2**0.5
#         # H_hat = min((self.grad.dot(self.hess.dot(self.grad)))**2 / (np.linalg.norm(self.grad)**5), (self.reg_coef/2)/12)
#         condition = self.loss.value(self.y - self.grad/np.sqrt(H_hat*np.linalg.norm(self.grad))) <= \
#             self.loss.value(self.y)  - np.linalg.norm(self.grad)**1.5 / 6 / np.sqrt(H_hat)
#         while not condition:
#             H_hat *= 2
#             if H_hat >= (self.reg_coef/2) / 12:
#                 H_hat = (self.reg_coef/2) / 12
#                 condition = self.loss.value(self.y - self.grad/np.sqrt(H_hat*np.linalg.norm(self.grad))) <= \
#                     self.loss.value(self.y)  - np.linalg.norm(self.grad)**1.5 / 6 / np.sqrt(H_hat)
#                 break
#             condition = self.loss.value(self.y - self.grad/np.sqrt(H_hat*np.linalg.norm(self.grad))) <= \
#                 self.loss.value(self.y)  - np.linalg.norm(self.grad)**1.5 / 6 / np.sqrt(H_hat)

            
#         if condition:
#             self.x = self.y - self.grad/np.sqrt(H_hat*np.linalg.norm(self.grad))
#             # solver_it = 0
#             self.s += (k+1)*(k+2)/2 * self.grad
#         else: 
#             print(f"{condition=}")
#             self.x, solver_it = self.cubic_solver(self.y, self.grad, self.hess, self.reg_coef, self.solver_it_max, self.solver_eps)
#             self.s += (k+1)*(k+2)/2 * self.loss.gradient(self.x)
#             self.solver_it += solver_it
        
#     def init_run(self, *args, **kwargs):
#         super().init_run(*args, **kwargs)
#         self.trace.solver_its = [0]
#         #init
#         self.x0 = self.x.copy()
#         self.y = self.x.copy()
#         self.v = self.x.copy()
#         self.s = np.zeros_like(self.x)
#         self.N = (self.reg_coef/2) / 10000
        
#     def update_trace(self):
#         super().update_trace()
#         self.trace.solver_its.append(self.solver_it)

# class CubicFastLS(Optimizer):
#     """
#     Newton method with cubic regularization for global convergence.
#     The method was studied by Nesterov and Polyak in the following paper:
#         "Cubic regularization of Newton method and its global performance"
#         https://link.springer.com/article/10.1007/s10107-006-0706-8
    
#     Arguments:
#         reg_coef (float, optional): an estimate of the Hessian's Lipschitz constant
#     """
#     def __init__(self, reg_coef=None, solver_it_max=100, solver_eps=1e-8, cubic_solver=None, *args, **kwargs):
#         # super(Cubic, self).__init__(*args, **kwargs)
#         super().__init__(*args, **kwargs)
#         self.reg_coef = reg_coef
#         self.cubic_solver = cubic_solver
#         self.solver_it = 0
#         self.solver_it_max = solver_it_max
#         self.solver_eps = solver_eps
#         if reg_coef is None:
#             self.reg_coef = self.loss.hessian_lipschitz
#         if cubic_solver is None:
#             self.cubic_solver = ls_cubic_solver
#         self.H = self.reg_coef/2

#     def step(self):
#         self.grad = self.loss.gradient(self.x)
#         self.hess = self.loss.hessian(self.x)
        
#         H_hat = (self.grad.dot(self.hess.dot(self.grad)))**2 / (np.linalg.norm(self.grad)**5)

#         H_hat = min((self.grad.dot(self.hess.dot(self.grad)))**2 / (np.linalg.norm(self.grad)**5), self.H)
#         # H_hat = (self.grad.dot(self.hess.dot(self.grad)))**2 / (np.linalg.norm(self.grad)**5)*1.2
#         condition = self.loss.value(self.x - self.grad/np.sqrt(H_hat*np.linalg.norm(self.grad))) <= \
#             self.loss.value(self.x)  - np.linalg.norm(self.grad)**1.5 / 6 / np.sqrt(H_hat)

#         # print(f"{H_hat=}")
#         while not condition:
#             H_hat *= 2
#             if H_hat > self.H:
#                 H_hat = self.H
#                 condition = self.loss.value(self.x - self.grad/np.sqrt(H_hat*np.linalg.norm(self.grad))) <= \
#                     self.loss.value(self.x)  - np.linalg.norm(self.grad)**1.5 / 6 / np.sqrt(H_hat)
#                 break
#             condition = self.loss.value(self.x - self.grad/np.sqrt(H_hat*np.linalg.norm(self.grad))) <= \
#                 self.loss.value(self.x)  - np.linalg.norm(self.grad)**1.5 / 6 / np.sqrt(H_hat)

#         # print(f"{condition=}")
#         if condition:
#             self.x = self.x - self.grad/np.sqrt(H_hat*np.linalg.norm(self.grad))
#             solver_it = 0
#         else:      
#             # H = (self.grad.dot(self.hess.dot(self.grad)))**2 / (np.linalg.norm(self.grad)**5)
#             # H = self.H
#             # self.x, solver_it = self.cubic_solver(self.x, self.grad, self.hess, H, self.solver_it_max, self.solver_eps)



#             H = self.H
#             grad_norm = np.linalg.norm(self.grad)
#             self.identity_coef = (H * grad_norm)**0.5
#             self.identity_coef = (grad_norm)**2
#             # self.x_old = copy.deepcopy(self.x)
#             # self.grad_old = copy.deepcopy(self.grad)
#             delta_x = -np.linalg.solve(self.hess + self.identity_coef*np.eye(self.loss.dim), self.grad)
#             self.x += delta_x
#             solver_it = 0

#         self.solver_it += solver_it
        
#     def init_run(self, *args, **kwargs):
#         super().init_run(*args, **kwargs)
#         self.trace.solver_its = [0]
        
#     def update_trace(self):
#         super().update_trace()
#         self.trace.solver_its.append(self.solver_it)



# class CubicFastLS(Optimizer):
#     """
#     Newton method with cubic regularization for global convergence.
#     The method was studied by Nesterov and Polyak in the following paper:
#         "Cubic regularization of Newton method and its global performance"
#         https://link.springer.com/article/10.1007/s10107-006-0706-8
    
#     Arguments:
#         reg_coef (float, optional): an estimate of the Hessian's Lipschitz constant
#     """
#     def __init__(self, reg_coef=None, solver_it_max=100, solver_eps=1e-8, cubic_solver=None, *args, **kwargs):
#         # super(Cubic, self).__init__(*args, **kwargs)
#         super().__init__(*args, **kwargs)
#         self.reg_coef = reg_coef
#         self.cubic_solver = cubic_solver
#         self.solver_it = 0
#         self.solver_it_max = solver_it_max
#         self.solver_eps = solver_eps
#         if reg_coef is None:
#             self.reg_coef = self.loss.hessian_lipschitz
#         if cubic_solver is None:
#             self.cubic_solver = ls_cubic_solver
#         self.H = self.reg_coef/2
        
#     def step(self):
#         self.grad = self.loss.gradient(self.x)
#         fx = self.loss.value(self.x)


#         # print(f"{self.it, H_hat/np.linalg.norm(self.grad)=}")
#         # self.hess = self.loss.hessian(self.x)


#         e = self.grad / np.linalg.norm(self.grad)
#         f = lambda H_hat:  -(\
#             fx \
#             - 1*np.linalg.norm(self.grad)**1.5 / 6 / np.sqrt(H_hat)  \
#             # + 0.5 * self.grad.dot(self.hess.dot(self.grad)) / H_hat * np.linalg.norm(self.grad) \
#             - self.loss.value(self.x - self.grad/np.sqrt(H_hat*np.linalg.norm(self.grad))) \
#         )


#         left = 0
#         right = 2*self.H
#         H_hat = right
#         condition = self.loss.value(self.x - self.grad/np.sqrt(H_hat*np.linalg.norm(self.grad))) <= \
#                 self.loss.value(self.x)  - np.linalg.norm(self.grad)**1.5 / 6 / np.sqrt(H_hat)

#         if condition:

#             # print(f"{right=}")

#             # 2) Golden-section search on [left, right]
#             rho = (np.sqrt(5.0) - 1.0) / 2.0  # ~0.618
#             c = right - rho * (right - left)
#             d = left  + rho * (right - left)
#             fc, fd = f(c), f(d)
        

#             atol, rtol = 1e-12, 1e-1

#             while (right - left) > atol:

#                 if fc > fd:                   # minimum in [c, right]
#                     left = c
#                     c, fc = d, fd
#                     d = left + rho * (right - left)
#                     fd = f(d)
#                 else:                        # minimum in [left, d]
#                     right = d
#                     d, fd = c, fc
#                     c = right - rho * (right - left)
#                     fc = f(c)
#                 # i+=1
#                 if abs(fc - fd) < atol/100:
#                     break

#             # H_hat = (right+left)/2
#             H_hat = right
#             # print(f"{H_hat=}")

#         # # print(f"{H_hat=}")
#         # if H_hat > self.reg_coef/2:
#         #     print(f"{H_hat/(self.reg_coef/2)=}")

        
#         # if self.it % 100 == 0:
#         #     # self.hess = self.loss.hessian(self.x)
#         #     # xi = (self.grad.dot(self.hess.dot(self.grad)))**2 / (np.linalg.norm(self.grad)**5)
#         #     # x = np.linspace(min(1 / 128 * xi,   xi), max(1 / 32 * xi, xi), 50)
        
#         #     # x = np.linspace(self.reg_coef/2*0.4, self.reg_coef/2, 50)
#         #     # x = np.logspace(np.log10(1e-10), np.log10(1e-6), 50)
#         #     x = np.logspace(np.log10(0.9 * H_hat ), np.log10(1.1 * H_hat ), 50)
#         #     # x = np.logspace(-7, np.log10(2*max(xi, right)), 50)
#         #     x = [i for i in x]
#         #     y = [f(h) for h in x]
#         #     # y = [self.loss.value(self.x - self.grad/np.sqrt(h*np.linalg.norm(self.grad))) - (self.loss.value(self.x)  - np.linalg.norm(self.grad)**1.5 / 6 / np.sqrt(h)) for h in x]
#         #     print(f"{y[-1]=}")
#         #     ymin = y[0]
#         #     for i in y:
#         #         ymin = min(i, ymin)
#         #     y = [i - ymin + 1e-8 for i in y]
#         #     # plt.axhline(y=-ymin + 1e-8, color='red')

#         #     # y = [i + 1e-15 for i in y]
#         #     # plt.axvline(x=xi, color='blue')
#         #     plt.axvline(x=H_hat, color='red')
#         #     # plt.axvline(x=left, color='green')
#         #     # plt.axvline(x=right, color='green')
#         #     plt.semilogy(x, y)
#         #     plt.grid(True)
#         #     plt.show()
        
#         # self.hess = self.loss.hessian(self.x)
#         # linesearch 
#         # H_hat = (self.grad.dot(self.hess.dot(self.grad)))**2 / (np.linalg.norm(self.grad)**5)*2
#         # condition = self.loss.value(self.x - self.grad/np.sqrt(H_hat*np.linalg.norm(self.grad))) <= \
#         #     self.loss.value(self.x)  - np.linalg.norm(self.grad)**1.5 / 6 / np.sqrt(H_hat)
#         # while not condition:
#         #     H_hat *= 2
#         #     if H_hat > (self.reg_coef/2):
#         #         H_hat = (self.reg_coef/2)
#         #         condition = self.loss.value(self.x - self.grad/np.sqrt(H_hat*np.linalg.norm(self.grad))) <= \
#         #             self.loss.value(self.x)  - np.linalg.norm(self.grad)**1.5 / 6 / np.sqrt(H_hat)
#         #         break
#         #     condition = self.loss.value(self.x - self.grad/np.sqrt(H_hat*np.linalg.norm(self.grad))) <= \
#         #         self.loss.value(self.x)  - np.linalg.norm(self.grad)**1.5 / 6 / np.sqrt(H_hat)
            
#         # H_hat = 1.2*xi
#         # print(f"{self.reg_coef=}")
#         if condition:
#             self.x = self.x - self.grad/np.sqrt(H_hat*np.linalg.norm(self.grad))
#             solver_it = 0
#         else:      
#             print(f"{condition=}")

#             self.hess = self.loss.hessian(self.x)
#             # self.x, solver_it = self.cubic_solver(self.x, self.grad, self.hess, self.reg_coef/2, self.solver_it_max, self.solver_eps)
#             H = self.H
#             grad_norm = np.linalg.norm(self.grad)
#             self.identity_coef = (H * grad_norm)**0.5
#             self.identity_coef = (grad_norm)**2
#             self.identity_coef = 0
#             # self.x_old = copy.deepcopy(self.x)
#             # self.grad_old = copy.deepcopy(self.grad)
#             delta_x = -np.linalg.solve(self.hess + self.identity_coef*np.eye(self.loss.dim), self.grad)
#             self.x += delta_x
#             solver_it = 0

#         self.solver_it += solver_it
        
#     def init_run(self, *args, **kwargs):
#         super().init_run(*args, **kwargs)
#         self.trace.solver_its = [0]
#         self.prevH = self.reg_coef/2
        
#     def update_trace(self):
#         super().update_trace()
#         self.trace.solver_its.append(self.solver_it)
