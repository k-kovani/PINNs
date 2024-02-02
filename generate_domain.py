import numpy as np
from pyDOE import lhs
import random
import matplotlib.pyplot as plt

# ================================
# Generate the training patterns:
# ================================

class Generate_Custom_Domain:

    def __init__(self, num_domain, num_bc, num_ic, sampling_methods, Xmin, Xmax):
        self.num_domain = num_domain            # number of points for the pde computational domain
        self.num_bc = num_bc                    # number of boundary condition points
        self.num_ic = num_ic                    # number of initial condition points
        self.sampling_method = sampling_method  # 'linspace', 'random', 'meshgrid', 'lhs'
        self.xmin, self.tmin = Xmin             # min. values of input parameters
        self.xmax, self.tmax = Xmax             # max. values of input parameters


    def generate_samples(self, num):
      
      if self.sampling_method == 'linspace':
         x = np.linspace(self.xmin, self.xmax, num).reshape((num, 1))
         t = np.linspace(self.tmin, self.tmax, num).reshape((num, 1))
         X_samples = np.concatenate([x, t], axis=-1)
  
        
      elif self.sampling_method == 'random':
         x = np.random.uniform(low=self.xmin, high=self.xmax, size=(num, 1))
         t = np.random.uniform(low=self.tmin, high=self.tmax, size=(num, 1))
         X_samples = np.concatenate([x, t], axis=-1)
     
      elif self.sampling_method == 'meshgrid':
        xx = np.linspace(self.xmin, self.xmax, num)
        tt = np.linspace(self.tmin, self.tmax, num)
        X_samples = np.array(np.meshgrid(xx, tt)).reshape(2, -1).T

      elif self.sampling_method == 'lhs':
        samples = lhs(2, num)
        x = samples[:, 0:1] * (self.xmax - self.xmin) + self.xmin
        t = samples[:, 1:2] * (self.tmax - self.tmin) + self.tmin
        X_samples = np.concatenate([x, t], axis=-1)
        
      return X_samples


    def generate_domain(self):
      X_domain = self.generate_samples(self.num_domain)
      Y_domain = np.zeros((self.num_samples, 1))
      return X_domain, Y_domain

  
    def boundary_value(self, X, a, b):
      y_bc = a * X[:, 0:1] + b * X[:, 1:2] 
      return y_bc

  
    def generate_bc(self):

      X_bc1 = self.generate_samples(self.num_bc)
      X_bc1[:, 0:1] = self.xmin

      X_bc2 = self.generate_samples(self.num_bc)
      X_bc2[:, 0:1] = self.xmax

      Y_bc1 = self.boundary_value(X_bc1, 0.0, 0.0)
      Y_bc2 = self.boundary_value(X_bc2, 2.0, 0.0)

      X_bc = np.vstack([X_bc1, X_bc2])
      Y_bc = np.vstack([Y_bc1, Y_bc2])
      return X_bc, Y_bc


    def ic_value(self, X):
        u_ic = np.sin(np.pi * X[:, 0:1])
        return u_ic


    def generate_ic(self):

        X_ic = self.generate_samples(self.num_ic)
        X_ic[:, 1:2] = 0
      
        Y_ic = self.ic_value(X_ic)
        return X_ic, Y_ic



# Visualize the training patterns:
def show_training_patterns(X_domain, X_bc, X_ic)
    plt.scatter(X_d[:, 0:1], X_d[:, 1:2], color="b", label="Computational Domain points")
    plt.scatter(X_bc[:, 0:1], X_bc[:, 1:2], color="r", label="Boundary Condition points")
    plt.scatter(X_ic[:, 0:1], X_ic[:, 1:2], color="g", label="Initial Condition points")
    plt.title("Training Patterns")
    plt.xlabel("x")
    plt.ylabel("t", rotation=0)
    plt.legend()
    plt.grid(alpha=0.5)
    plt.show()
    return None

# Visualize the initial condition:
def show_initial_condition(X_ic, Y_ic)
    plt.scatter(X_ic[:, 0:1], Y_ic, color="g")
    plt.title("Initial Condition")
    plt.xlabel("x")
    plt.ylabel("u", rotation=0)
    plt.legend()
    plt.grid(alpha=0.5)
    plt.show()
    return None
