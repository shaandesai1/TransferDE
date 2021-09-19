import torch
import numpy as np


class F_Gen:
    def __init__(self, force_bias = False):
        self.force_bias = force_bias
        self.sin = torch.sin
        self.cos = torch.cos
        self.exp = lambda t: self.exp(-t)
        self.sin_cos = lambda t: self.sin(t)*self.cos(t)
        self.sin2 = lambda t: self.sin(t)*self.sin(t)
        self.cos2 = lambda t: self.cos(t)*self.cos(t)

    def realize(self):
    	pass

    def realize_recursive(self, high = 6):
    	with torch.no_grad():
	        num_repeats = torch.randint(1, 5, (1,))
	        f_lst = []
	        for i in range(num_repeats):
	            f_lst.append(self.realize())

	        if self.force_bias:
	            #bias = float(np.sign(torch.rand(1)-0.5))*bias
	            bias = (torch.rand(1) -0.5) * 2 * self.force_bias

	            b_lst = [bias]
	            b_lst = b_lst +[0]*(num_repeats -1)
	        return lambda t: torch.tensor([f(t).detach().numpy() for i, f in enumerate(f_lst)], dtype = torch.float32).sum(axis = 0)

class Wave_Gen(F_Gen):
    def __init__(self, phase_shift = 2*np.pi, amplitude_range = 5, angular_freq_range = 3, force_bias = False):
        
        super().__init__(force_bias)

        self.phi_shift = phase_shift
        self.a_range = amplitude_range
        self.w_range = angular_freq_range
    
    
    def realize(self):
        
        f = np.random.choice([self.sin, self.cos, self.sin_cos])

        alpha = torch.rand(1) * self.a_range
        omega = torch.rand(1) * self.w_range
        phi = torch.rand(1) * self.phi_shift

        # if self.force_bias:
        #     k = torch.rand(1) * self.force_bias
        # else:
        #     k = 0
        
        def force(t, f, alpha, omega, phi):
            return alpha * f(omega *t + phi) #+ k


        return lambda t: force(t, f, alpha, omega, phi)
    
    