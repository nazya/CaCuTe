import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

import copy

from optmethods.optimizer import Optimizer
# from optmethods.second_order import RegNewton


from optmethods.line_search import LineSearch


class RegNewtonLS(LineSearch):
    """
    This line search estimates the Hessian Lipschitz constant for the Global Regularized Newton.
    See the following paper for the details and convergence proof:
        "Regularized Newton Method with Global O(1/k^2) Convergence"
        https://arxiv.org/abs/2112.02089
    For consistency with other line searches, 'lr' parameter is used to denote the inverse of regularization.
    Arguments:
        decrease_reg (boolean, optional): multiply the previous regularization parameter by 1/backtracking (default: True)
        backtracking (float, optional): constant by which the current regularization is divided (default: 0.5)
    """
    
    def __init__(self, decrease_reg=True, backtracking=0.5, H0=None, *args, **kwargs):
        super(RegNewtonLS, self).__init__(*args, **kwargs)
        self.decrease_reg = decrease_reg
        self.backtracking = backtracking
        self.H0 = H0
        self.H = self.H0
        self.attempts = 0
        
    def condition(self, x_new, x, grad, identity_coef):
        if self.f_prev is None:
            self.f_prev = self.loss.value(x)
        self.f_new = self.loss.value(x_new)
        r = self.loss.norm(x_new - x)

        condition_f = self.f_new <= self.f_prev - 2/3 * identity_coef * r**2
        
        grad_new = self.loss.gradient(x_new)
        condition_grad = self.loss.norm(grad_new) <= 2 * identity_coef * r
        
        
        self.attempts = self.attempts + 1 if not condition_f or not condition_grad else 0
        return condition_f and condition_grad
        
    def __call__(self, x, f_prev, grad, hess):
        self.f_prev = f_prev
    
        if self.decrease_reg:
            self.H *= self.backtracking
        grad_norm = self.loss.norm(grad)
        identity_coef = np.sqrt(self.H * grad_norm)
        
        x_new = x - np.linalg.solve(hess + identity_coef*np.eye(self.loss.dim), grad)
        condition_met = self.condition(x_new, x, grad, identity_coef)

        # try:
        #     x_new = x - np.linalg.solve(hess + identity_coef*np.eye(self.loss.dim), grad)
        #     condition_met = self.condition(x_new, x, grad, identity_coef)
        # except np.linalg.LinAlgError:
        #     condition_met = False
        #     self.H *= 16
        
        
        # self.it += self.it_per_call
        it_extra = 0
        it_max = min(self.it_max, self.optimizer.ls_it_max - self.it)
        while not condition_met and it_extra < it_max:
            if self.backtracking / self.H == 0:
                break
            self.H /= self.backtracking
            identity_coef = np.sqrt(self.H * grad_norm)
            x_new = x - np.linalg.solve(hess + identity_coef*np.eye(self.loss.dim), grad)
            condition_met = self.condition(x_new, x, grad, identity_coef)
                        
            condition_met = self.condition(x_new, x, grad, identity_coef)
            it_extra += 1
            
        # self.f_prev = self.f_new
        # self.it += it_extra
        self.lr = 1 / identity_coef
        return x_new, self.f_new

    def reset(self, *args, **kwargs):
        super(RegNewtonLS, self).reset(*args, **kwargs)
        self.f_prev = None


# fair line search
class CaCuAdaN(Optimizer):
    def __init__(self, loss, identity_coef=None, hess_lip=None, backtracking=0.5, *args, **kwargs):
        if hess_lip is None:
            hess_lip = loss.hessian_lipschitz
            if loss.hessian_lipschitz is None:
                hess_lip = 1e-5
                warnings.warn(f"No estimate of Hessian-Lipschitzness is given, so a small value {hess_lip} is used as a heuristic.")
        self.hess_lip = hess_lip
        
        self.H = hess_lip / 2
            
        line_search = RegNewtonLS(decrease_reg=True, backtracking=backtracking, H0=self.H)
        super().__init__(loss=loss, line_search=line_search, *args, **kwargs)
        
        self.identity_coef = identity_coef
        self.flag = False
        self.f_prev = None
        self.H_min = self.H
        self.H_hat = self.H

      
    # def step(self):
    #     if True:
    #         self.f_prev = fx = self.loss.value(self.x)
    #         self.grad, p = self.loss.GgHg(self.x)
    #         gnorm = np.linalg.norm(self.grad)
    #         if gnorm == 0:
    #             return
    #         pe = p / gnorm / gnorm
    #         alpha = 3/4/1.07
    #         M = pe**2 / gnorm * 9/16/(alpha)**2
            
    #         self.H_hat /= 2
    #         while True:
    #             f_new = self.loss.value(self.x - self.grad/np.sqrt(self.H_hat*gnorm))
    #             check = f_new <= fx# - 2/3*gnorm**1.5 / np.sqrt(self.H_hat)
    #             # check = f_new  <= fx + 1/2 * p / self.H_hat / gnorm - 2/3 * gnorm**1.5 / np.sqrt(self.H_hat)

    #             if check or self.H_hat >= self.H:
    #                 print(f"{self.H_hat=}")
    #                 break
                
    #             self.H_hat = min(2*self.H_hat, self.H)
                
    #         self.hess = self.loss.hessian(self.x)
            
    #         grad_norm = self.loss.norm(self.grad)
    #         self.identity_coef = (self.H_hat * grad_norm)**0.5
    #         # self.identity_coef = grad_norm
    #         self.x_old = copy.deepcopy(self.x)
    #         self.grad_old = copy.deepcopy(self.grad)
    #         delta_x = -np.linalg.solve(self.hess + self.identity_coef*np.eye(self.loss.dim), self.grad)
    #         self.x += delta_x
    #     else:
    #         self.grad = self.loss.gradient(self.x)
    #         self.hess = self.loss.hessian(self.x)
    #         self.x, self.f_prev = self.line_search(self.x, self.f_prev, self.grad, self.hess)
    def step(self):
        if self.flag is False:
            self.f_prev = fx = self.loss.value(self.x)
            self.grad, p = self.loss.GgHg(self.x)
            gnorm = np.linalg.norm(self.grad)
            
            self.H_hat /= 2
            # self.H_hat = min(self.H, self.H_hat)
            f_new = None
            while True:
                f_new = self.loss.value(self.x - self.grad/np.sqrt(self.H_hat*gnorm))
                # check = f_new <= fx - 2/3/64*gnorm**1.5 / np.sqrt(self.H_hat)
                check = f_new  <= fx + 1/2 * p / self.H_hat / gnorm - 2/3 * gnorm**1.5 / np.sqrt(self.H_hat)

                if check:
                    break
                # self.H_hat *= 2
                self.H_hat = min(2*self.H_hat, self.H)
                
                # if not check:
                #     # self.H_hat *= 2
                #     # self.H_hat = min(self.H_hat, self.H)
                #     self.H_hat = min(2*self.H_hat, self.H)
                #     break
                
                # if M < self.H_hat/2:
                #     self.H_hat = self.H_hat / 2
                #     break
                # else:
                #     self.H_hat = min(self.H_hat, self.H)
                #     break

            if  f_new <= fx - 2/3*gnorm**1.5 / np.sqrt(self.H_hat)/64:# and self.H_hat <= self.H:
                self.x = self.x - self.grad/np.sqrt(self.H_hat*gnorm) 
                self.f_prev = f_new
                self.H_min = min(self.H_min, self.H_hat)
                # print(f"{self.it=}")
            else:
                self.flag = True
                self.line_search.H = self.H_min
                self.hess = self.loss.hessian(self.x)
                self.x, self.f_prev = self.line_search(self.x, self.f_prev, self.grad, self.hess)

        else:
            self.grad = self.loss.gradient(self.x)
            self.hess = self.loss.hessian(self.x)
            self.x, self.f_prev = self.line_search(self.x, self.f_prev, self.grad, self.hess)

        # print(f"{self.f_prev=}")


    def init_run(self, *args, **kwargs):
        super().init_run(*args, **kwargs)
        self.x_old = None
        self.hess = None
        self.trace.lrs = []
        
    def update_trace(self, *args, **kwargs):
        super().update_trace(*args, **kwargs)
        # if not self.use_line_search:
        #     self.trace.lrs.append(1 / self.identity_coef)


def empirical_hess_lip(grad, grad_old, hess, x, x_old, loss):
    grad_error = grad - grad_old - hess@(x - x_old)
    r2 = loss.norm(x - x_old)**2
    if r2 > 0:
        return 2 * loss.norm(grad_error) / r2
    return np.finfo(float).eps


class CaCuGN(Optimizer):
    def __init__(self, loss, identity_coef=None, hess_lip=None, adaptive=False, line_search=None,
                 use_line_search=False, backtracking=0.5, *args, **kwargs):
        if hess_lip is None:
            hess_lip = loss.hessian_lipschitz
            if loss.hessian_lipschitz is None:
                hess_lip = 1e-5
                warnings.warn(f"No estimate of Hessian-Lipschitzness is given, so a small value {hess_lip} is used as a heuristic.")
        self.hess_lip = hess_lip
        
        self.H = hess_lip / 2

        super().__init__(loss=loss, line_search=line_search, *args, **kwargs)
        
        self.identity_coef = identity_coef
        
    def step(self):
        self.grad = self.loss.gradient(self.x)
        self.hess = self.loss.hessian(self.x)
        
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
        # if not self.use_line_search:
        #     self.trace.lrs.append(1 / self.identity_coef)


# estimate H with first order
class AdaNWU(Optimizer):
    def __init__(self, loss, identity_coef=None, hess_lip=None, backtracking=0.5, *args, **kwargs):
        if hess_lip is None:
            hess_lip = loss.hessian_lipschitz
            if loss.hessian_lipschitz is None:
                hess_lip = 1e-5
                warnings.warn(f"No estimate of Hessian-Lipschitzness is given, so a small value {hess_lip} is used as a heuristic.")
        self.hess_lip = hess_lip
        
        self.H = hess_lip / 2
            
        line_search = RegNewtonLS(decrease_reg=True, backtracking=backtracking, H0=self.H)
        
        super().__init__(loss=loss, line_search=line_search, *args, **kwargs)
        
        self.identity_coef = identity_coef
        # self.flag = False

      
    def step(self):
        
        if self.it == 0:
            self.f_prev = fx = self.loss.value(self.x)
            self.grad, p = self.loss.GgHg(self.x)
            gnorm = np.linalg.norm(self.grad)

            if gnorm == 0:
                return
            pe = p / gnorm / gnorm
            alpha = 1-1/64
            M = pe**2 / gnorm * 9/16/(alpha)**2
            
            self.H_hat = self.H
            self.H_hat = min(self.H, self.H_hat)
            f_new = None
            while True:
                f_new = self.loss.value(self.x - self.grad/np.sqrt(self.H_hat*gnorm))
                # f_new = self.loss.value(self.x - self.grad/np.sqrt(max(M, self.H_hat)*gnorm))
                check = f_new <= fx - 2/3/64*gnorm**1.5 / np.sqrt(self.H_hat)

                # check = f_new  <= fx + 1/2 * p / self.H_hat / gnorm - 2/3 * gnorm**1.5 / np.sqrt(self.H_hat)

                if not check:
                    # self.H_hat *= 2
                    # self.H_hat = min(self.H_hat, self.H)
                    self.H_hat = min(2*self.H_hat, self.H)
                    break
                
                if M < self.H_hat/2:
                    self.H_hat = self.H_hat / 2
                else:
                    self.H_hat = min(self.H_hat, self.H)
                    break
            self.line_search.H = self.H_hat
        else:
            self.grad = self.loss.gradient(self.x)
        self.hess = self.loss.hessian(self.x)
        self.x, self.f_prev = self.line_search(self.x, self.f_prev, self.grad, self.hess)

    def init_run(self, *args, **kwargs):
        super().init_run(*args, **kwargs)
        self.x_old = None
        self.hess = None
        self.trace.lrs = []
        
    def update_trace(self, *args, **kwargs):
        super().update_trace(*args, **kwargs)
        # if not self.use_line_search:
        #     self.trace.lrs.append(1 / self.identity_coef)


# estimate H with first order
class AdaN(Optimizer):
    def __init__(self, loss, identity_coef=None, hess_lip=None, backtracking=0.5, *args, **kwargs):
        if hess_lip is None:
            hess_lip = loss.hessian_lipschitz
            if loss.hessian_lipschitz is None:
                hess_lip = 1e-5
                warnings.warn(f"No estimate of Hessian-Lipschitzness is given, so a small value {hess_lip} is used as a heuristic.")
        self.hess_lip = hess_lip
        
        self.H = hess_lip / 2
            
        line_search = RegNewtonLS(decrease_reg=True, backtracking=backtracking, H0=self.H)
        
        super().__init__(loss=loss, line_search=line_search, *args, **kwargs)
        
        self.identity_coef = identity_coef
        self.f_prev = None

    def step(self):
        self.grad = self.loss.gradient(self.x)
        # self.grad, p = self.loss.GgHg(self.x)
        self.hess = self.loss.hessian(self.x)
        self.x, self.f_prev = self.line_search(self.x, self.f_prev, self.grad, self.hess)
        
    def init_run(self, *args, **kwargs):
        super().init_run(*args, **kwargs)
        self.x_old = None
        self.hess = None
        self.trace.lrs = []
        
    def update_trace(self, *args, **kwargs):
        super().update_trace(*args, **kwargs)
        # if not self.use_line_search:
        #     self.trace.lrs.append(1 / self.identity_coef)


# FO only backtracking
class CaCuAdaNP(Optimizer):
    def __init__(self, loss, identity_coef=None, hess_lip=None, backtracking=0.5, *args, **kwargs):
        if hess_lip is None:
            hess_lip = loss.hessian_lipschitz
            if loss.hessian_lipschitz is None:
                hess_lip = 1e-5
                warnings.warn(f"No estimate of Hessian-Lipschitzness is given, so a small value {hess_lip} is used as a heuristic.")
        self.hess_lip = hess_lip
        super().__init__(loss=loss, line_search=None, *args, **kwargs)
        
        self.H = hess_lip / 2
        self.H_hat = self.H
        self.H_max = self.H_hat
        self.H_hat = 1e-5
        self.f_prev = None
        self.grad_old = 1e5
        
        self.identity_coef = identity_coef
        
    def step(self):
        fx = self.loss.value(self.x)
        # self.grad, Hg = self.loss.gHg(self.x)
        self.grad, p = self.loss.GgHg(self.x)
        # self.grad, pe = self.loss.GeHe(self.x)
        gnorm = np.linalg.norm(self.grad)
        
        pe = p / gnorm / gnorm

        self.hess = self.loss.hessian(self.x)

        cacu = 4
        if self.H_hat/cacu > 0:
            self.H_hat = self.H_hat/cacu

        while True:
            f_new = self.loss.value(self.x - self.grad/np.sqrt(self.H_hat*gnorm))
            check = f_new  <= fx + 1/2 * p / self.H_hat / gnorm - 2/3 * gnorm**1.5 / np.sqrt(self.H_hat)

            # check *= self.loss.norm(self.loss.gradient(self.x - self.grad/np.sqrt(self.H_hat*gnorm))) < 2*gnorm
            # check *= f_new  <= fx + 64 * gnorm**1.5 / np.sqrt(self.H_hat)

            
            if check:
                # print(f"{self.H_hat=}")
                break
            self.H_hat = self.H_hat * 2
        self.f_prev = fx

        # self.H_max = max(self.H_max, self.H_hat)

        self.H = self.H_hat
        gnorm = self.loss.norm(self.grad)
        self.identity_coef = (self.H * gnorm)**0.5
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
        # if not self.use_line_search:
        #     self.trace.lrs.append(1 / self.identity_coef)
