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
    newton_step = -np.linalg.solve(H, g)
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
        s_lam = -np.linalg.solve(H + lam*id_matrix, g)
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




class CaCuAdGD(Optimizer):
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
        self.H = self.reg_coef/2
        self.H_hat = 1e-8

    def _store_h_hat_use(self, M, loss_val=None):
        if self.H_hat >= M:
            self.trace.h_hat_used_its.append(self.it + 1)
            if loss_val is not None:
                self.trace.h_hat_used_loss_vals.append(loss_val)

    # def step(self):
    #     # self.grad = self.loss.gradient(self.x)
    #     # self.hess = self.loss.hessian(self.x)
    #     fx = self.loss.value(self.x)
    #     # p = self.grad.dot(self.hess.dot(self.grad))
    #     self.grad, p = self.loss.GgHg(self.x)

    #     gnorm = np.linalg.norm(self.grad)
    #     M = p**2 / (gnorm**5) * 1.16
    #     # M = p**2 / (gnorm**5) * 1.16
        
    #     self.H_hat = max( self.H_hat/ 4, 1e-128)
    #     check = False
    #     while not check:
    #         self.H_hat *= 2
    #         fnew = self.loss.value(self.x - self.grad/np.sqrt(self.H_hat*gnorm))
    #         check = fnew  <= \
    #             fx \
    #                 + 1/2 * p / self.H_hat / gnorm \
    #                     - 2/3 * gnorm**1.5 / np.sqrt(self.H_hat)

    #         # check = check and (fnew <= fx)
            

    #     H_hat = max(M, self.H_hat)        
    #     self.x = self.x - self.grad/np.sqrt(H_hat*gnorm)



    # def step(self):
    #     fx = self.loss.value(self.x)
        
        
    #     # g, Hg = self.loss.gHg(self.x)
    #     g, p = self.loss.GgHg(self.x)
    #     self.grad = g

    #     gnorm = np.linalg.norm(g)
    #     if gnorm == 0:
    #         return
        
    #     pe = p / gnorm / gnorm
    #     M = pe**2 / gnorm * (1.07)**2

    #     alpha = 3/4/1.07


    #     check = self.H_hat/16 > M
    #     # check = max(self.H_hat/16, M)
    #     if check:
    #         self.H_hat = self.H_hat/16
    #         check = self.loss.value(self.x - g/np.sqrt(self.H_hat*gnorm)) <= fx - 2/3 * (1-alpha)* gnorm**1.5 / np.sqrt(self.H_hat) 
    #         if check:
    #             print(f"{self.H_hat=}")
    #         else:
    #             # if M > 0.:
    #             #     self.H_hat = min(self.H_hat * 16, M)
    #             # else:
    #             #     self.H_hat = self.H_hat * 16
    #             self.H_hat = self.H_hat * 16
            
        
    #     if check:
    #         self.x = self.x - self.grad/np.sqrt(max(M, self.H_hat)*gnorm)
    #         return
           

    #     while True:
    #         check = self.loss.value(self.x - self.grad/np.sqrt(max(M, self.H_hat)*gnorm)) <= fx - 2/3 * (1-alpha)* gnorm**1.5 / np.sqrt(max(M, self.H_hat)) 
    #         if check:
    #             self.x = self.x - self.grad/np.sqrt(max(M, self.H_hat)*gnorm) 
    #             # print(f"{self.H_hat=}")
    #             # if max(M, self.H_hat) == M:
    #             #     self.H_hat*=2
    #             return
    #         self.H_hat *= 2


    def step(self):
        fx = self.loss.value(self.x)
        
        
        # g, Hg = self.loss.gHg(self.x)
        g, p = self.loss.GgHg(self.x)
        self.grad = g

        gnorm = np.linalg.norm(g)
        if gnorm == 0:
            return
        
        pe = p / gnorm / gnorm
        M = pe**2 / gnorm * (1.07)**2

        alpha = 3/4/1.07
        # alpha = 0.7

        cacu = 16
        check = self.H_hat/cacu > M
        # check = self.H_hat/2 > M
        # check = max(self.H_hat/16, M)
        if check:
            self.H_hat = self.H_hat/cacu
            x_new = self.x - self.grad/np.sqrt(self.H_hat*gnorm)
            fxnew = self.loss.value(x_new)
            check = fxnew  <= fx + 1/2 * p / self.H_hat / gnorm - 2/3 * gnorm**1.5 / np.sqrt(self.H_hat)
            if check:
                # print(f"{self.H_hat=}")
                pass
            else:
                self.H_hat = self.H_hat * cacu 
            
        
        if check:
            self._store_h_hat_use(M, loss_val=fxnew)
            self.x = x_new
            return
           

        while True:
            H_used = max(M, self.H_hat)
            x_new = self.x - self.grad/np.sqrt(H_used*gnorm)
            fxnew = self.loss.value(x_new)
            check = fxnew <= fx - 2/3 * (1-alpha)* gnorm**1.5 / np.sqrt(H_used)
            if check:
                self._store_h_hat_use(M, loss_val=fxnew)
                self.x = x_new
                return
            self.H_hat *= 2

        
    def init_run(self, *args, **kwargs):
        super().init_run(*args, **kwargs)
        self.trace.h_hat_used_its = []
        self.trace.h_hat_used_loss_vals = []
        self.trace.solver_its = [0]
        
    def update_trace(self):
        super().update_trace()
        self.trace.solver_its.append(self.solver_it)



class CaCuAdGDLS(Optimizer):
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
        self.H = self.reg_coef/2
        self.H_hat = self.reg_coef/2
        
    def step(self):
        self.grad = self.loss.gradient(self.x)
        fx = self.loss.value(self.x)

        f = lambda H_hat:  -(\
            fx \
            - 1/6 * np.linalg.norm(self.grad)**1.5 / np.sqrt(H_hat)  \
            - self.loss.value(self.x - self.grad/np.sqrt(H_hat*np.linalg.norm(self.grad))) \
        )

        condition = False
        self.H_hat /= 4
        left = f(self.H_hat)
        while not condition:
            self.H_hat *= 4
            right = f(self.H_hat)
            condition = left < right
            left = right

        left = 0
        right = self.H_hat
        

        rho = (np.sqrt(5.0) - 1.0) / 2.0  # ~0.618
        c = right - rho * (right - left)
        d = left  + rho * (right - left)
        fc, fd = f(c), f(d)
        atol, rtol = 1e-13, 1e-1
        while (right - left) > atol:

            if fc > fd:                   # minimum in [c, right]
                left = c
                c, fc = d, fd
                d = left + rho * (right - left)
                fd = f(d)
            else:                        # minimum in [left, d]
                right = d
                d, fd = c, fc
                c = right - rho * (right - left)
                fc = f(c)
            if abs(fc - fd) < atol:
                break


        # H_hat = (right+left)/2
        H_hat = right
        self.x = self.x - self.grad/np.sqrt(H_hat*np.linalg.norm(self.grad))
        
    def init_run(self, *args, **kwargs):
        super().init_run(*args, **kwargs)
        self.trace.solver_its = [0]
        self.prevH = self.reg_coef/2
        
    def update_trace(self):
        super().update_trace()
        self.trace.solver_its.append(self.solver_it)


class AccCaCuAdGD(Optimizer):
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
        self.H = self.reg_coef/2
        self.H_hat = 1
        self.L_hat = 1
        self.A = 1
        # self.H_hat = self.reg_coef/2
        # def step(self):
        #     k = self.it

        #     coeff = 2
        #     H = self.H_hat
        #     self.H_hat /= coeff*coeff

        #     # assume
        #     condition = False
        #     while True:
        #         self.H_hat *= coeff
                
        #         if k > 1:
        #             self.v = self.x0 - self.s / np.sqrt( 23.04 * self.H_hat  * np.linalg.norm(self.s))
        #         self.y = k/(k+3)*self.x + 3/(k+3)*self.v
                
        #         fy = self.loss.value(self.y)
        #         self.grad = self.loss.gradient(self.y)
        #         gnorm = np.linalg.norm(self.grad)

        #         self.hess = self.loss.hessian(self.y)
        #         p = self.grad.dot(self.hess.dot(self.grad))
                
        #         condition = self.loss.value(self.y - self.grad/np.sqrt(self.H_hat*gnorm)) <= \
        #                 fy \
        #                     + 1/2 * p / self.H_hat / gnorm \
        #                         - 2/3 * gnorm**1.5 / np.sqrt(self.H_hat)
        #                     # - 1/6 * gnorm**1.5 / np.sqrt(self.H_hat)
        #         if condition is np.True_:
        #             break

        #     H_hat = self.H_hat
        #     condition = self.H_hat >= 4*p**2 / (gnorm**5)
        #     # print(f"{condition=}")
        #     # if condition is np.True_:
        #     #     print(f"{self.loss.value(self.y - self.grad/np.sqrt(self.H_hat*gnorm)) <= fy - 5/12 * gnorm**1.5 / np.sqrt(self.H_hat)=}")
        #     # print(f"{k, condition=}")
        #     if condition is np.False_:
        #         self.H_hat = H
        #         self.L_hat /= coeff*coeff
        #         self.L_hat = np.sqrt(H_hat * gnorm) / 2
        #         while condition is np.False_:
        #             self.L_hat *= coeff
                    
        #             if k > 1:
        #                 # self.v = self.x0 - self.s / np.sqrt( 9.6**2 * H_hat * self.L_hat  * np.linalg.norm(self.s))
        #                 self.v = self.x0 - self.s / np.sqrt(9.6**2 * self.L_hat**2  * np.linalg.norm(self.s))
        #             self.y = k/(k+3)*self.x + 3/(k+3)*self.v
                    
        #             fy = self.loss.value(self.y)
        #             self.grad = self.loss.gradient(self.y)
        #             gnorm = np.linalg.norm(self.grad)

        #             self.hess = self.loss.hessian(self.y)
        #             p = self.grad.dot(self.hess.dot(self.grad))

        #             # if gnorm**2 < gnorm**1.5:
        #             #     print(k)
        #             #     return
                    
        #             condition = self.L_hat > p / (gnorm**2)
        #             # print(f"{k, self.L_hat, p / (gnorm**2)=}")
        #             # print(f"{condition=}")
        #             if condition is np.True_:
        #                 condition = self.loss.value(self.y - self.grad/self.L_hat) <= \
        #                         fy \
        #                             - 5/24 * gnorm**2 / self.L_hat
        #                             # + 1/2 * p / self.H_hat / gnorm \
        #                             #     - 2/3 * gnorm**1.5 / np.sqrt(self.H_hat)
                    
        #         H_hat = self.L_hat**2/gnorm

        #     if k == 0:
        #         self.x = self.x - self.grad/np.sqrt(H_hat*np.linalg.norm(self.grad))
        #     else:
        #         self.x = self.y - self.grad/np.sqrt(H_hat*np.linalg.norm(self.grad))
        #         self.s += (k+1)*(k+2)/2 * self.grad


    def step(self):
        k = self.it

        coeff = 2
        H = self.H_hat
        self.H_hat /= coeff*coeff

        # # assume
        # condition = False
        # while True:
        #     self.H_hat *= coeff
            
        #     if k > 1:
        #         self.v = self.x0 - self.s / np.sqrt( 23.04 * self.H_hat  * np.linalg.norm(self.s))
            
        #     a = (k+1)*(k+2)/2
        #     alpha = a / (a + self.A)
        #     self.y = (1-alpha)*self.x + alpha*self.v
            
        #     fy = self.loss.value(self.y)
        #     self.grad = self.loss.gradient(self.y)
        #     gnorm = np.linalg.norm(self.grad)

        #     self.hess = self.loss.hessian(self.y)
        #     p = self.grad.dot(self.hess.dot(self.grad))
            
        #     condition = self.loss.value(self.y - self.grad/np.sqrt(self.H_hat*gnorm)) <= \
        #             fy \
        #                 + 1/2 * p / self.H_hat / gnorm \
        #                     - 2/3 * gnorm**1.5 / np.sqrt(self.H_hat)
        #                 # - 1/6 * gnorm**1.5 / np.sqrt(self.H_hat)
        #     if condition is np.True_:
        #         break

        # H_hat = self.H_hat
        # condition = self.H_hat >= 2*p**2 / (gnorm**5) #* np.sqrt(k+1)
        # # if condition is np.False_:
        # # if condition is np.True_:
        # #     print(f"{k, condition=}")
        # #     print(f"{self.loss.value(self.y - self.grad/np.sqrt(self.H_hat*gnorm)) <= fy - 5/12 * gnorm**1.5 / np.sqrt(self.H_hat)=}")
        # # print(f"{k, condition=}")
        # if condition is np.False_:
        #     gnorm_ = gnorm
        #     # self.H_hat = H
        #     # self.L_hat /= coeff*coeff
        #     # self.L_hat = np.sqrt(H_hat * gnorm) / 2
        #     coeff=1.1
        #     while condition is np.False_:
        #         self.H_hat *= coeff
            
        #         if k > 1:
        #             self.v = self.x0 - self.s / np.sqrt( 23.04 * self.H_hat  * np.linalg.norm(self.s))
        #         a = (k+1)*(k+2)/2
        #         alpha = a / (a + self.A)
        #         self.y = (1-alpha)*self.x + alpha*self.v
                
        #         fy = self.loss.value(self.y)
                
        #         self.grad = self.loss.gradient(self.y)
        #         gnorm = np.linalg.norm(self.grad)

        #         self.hess = self.loss.hessian(self.y)
                
        #         e = self.grad / gnorm
        #         p = e.dot(self.hess.dot(e))

                


        #         self.L_hat = np.sqrt(self.H_hat * gnorm)
        #         condition = self.L_hat > 2*p
        #         print(f"{self.L_hat/p=}")
        #         # print(f"{k, self.L_hat, e.dot(self.hess.dot(e)) =}")
        #         # print(f"{condition=}")
                
                
        #         # if True:
        #         # if condition is np.True_:
        #         #     condition = self.loss.value(self.y - self.grad/self.L_hat) <= \
        #         #             fy \
        #         #                 - 5/24 * gnorm**2 / self.L_hat
        #         #                     # - 5/12 * gnorm**1.5 / np.sqrt(self.H_hat)
        #         #                 # + 1/2 * p / self.H_hat / gnorm \
                
                
        #         # condition = 12 / (k+1) / gnorm < (4.8)**2 * self.L_hat**2
        #         # if condition is np.False_:
        #         #     continue

        #         # condition = N < (9.6)**2 / gnorm / k * self.L_hat**2 
        #         gnorm_ = gnorm 
        #         # if condition is np.False_:
        #         #     continue

                
                
                
                
        #     H_hat = self.L_hat**2/gnorm
        #     if k == 0:
        #         self.x = self.x - self.grad/np.sqrt(H_hat*np.linalg.norm(self.grad))
        #     else:
        #         self.x = self.y - self.grad/self.L_hat
        #         self.s += (k+1)*(k+2)/2 * self.grad
        #     self.A = (a + self.A)

        # else:
        #     if k == 0:
        #         self.x = self.x - self.grad/np.sqrt(H_hat*np.linalg.norm(self.grad))
        #     else:
        #         self.x = self.y - self.grad/np.sqrt(H_hat*np.linalg.norm(self.grad))
        #         self.s += (k+1)*(k+2)/2 * self.grad
        #     self.A = (a + self.A)

    
        
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


# class AccCaCuNA(Optimizer):
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
#         self.H_hat = 1e-8
#         self.N = self.H_hat


#     def step(self):
#         k = self.it

#         self.H_hat /= 4
#         check = False
#         while not check:
#             if k > 1:
#                 self.v = self.x0 - self.s / np.sqrt( 144*self.N * np.linalg.norm(self.s))
#             self.y = k/(k+3)*self.x + 3/(k+3)*self.v
#             self.grad = self.loss.gradient(self.y)
#             self.hess = self.loss.hessian(self.y)
#             fy = self.loss.value(self.y)
#             p = self.grad.dot(self.hess.dot(self.grad))

#             self.H_hat *= 2
#             # self.N = max(self.N, self.H_hat)

#             check = self.loss.value(self.y - self.grad/np.sqrt(self.H_hat*np.linalg.norm(self.grad))) <= \
#                 fy \
#                     + 1/2 * p / self.H_hat / np.linalg.norm(self.grad) \
#                         - 2/3 * np.linalg.norm(self.grad)**1.5 / np.sqrt(self.H_hat)

#         H_hat = max(p**2 / (np.linalg.norm(self.grad)**5), self.H_hat)
#         self.N = max(self.N, self.H_hat)

#         # H_hat = max(p**2 / (np.linalg.norm(self.grad)**5) * 2**0.5, self.H_hat)
#         if k == 0:
#             self.x = self.x - self.grad/np.sqrt(H_hat*np.linalg.norm(self.grad))
#         else:
#             self.x = self.y - self.grad/np.sqrt(H_hat*np.linalg.norm(self.grad))
#             self.s += (k+1)*(k+2)/2 * self.grad
        
#     def init_run(self, *args, **kwargs):
#         super().init_run(*args, **kwargs)
#         self.trace.solver_its = [0]
#         #init
#         self.x0 = self.x.copy()
#         self.y = self.x.copy()
#         self.v = self.x.copy()
#         self.s = np.zeros_like(self.x)
        
#     def update_trace(self):
#         super().update_trace()
#         self.trace.solver_its.append(self.solver_it)
