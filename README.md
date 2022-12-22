# Ridge regression with Laplace Kernel

An implementation of training a kernel machine using the laplace kernel.  See main.py for an example of using the laplace kernel on toy data.  The LaplaceKernel class (defined in kernel.py) follows the sklearn API.  The reg parameter in the fit function controls the scale of the regularization term in ridge regression (higher values will give a more regularized solution, 0 will lead to interpolation of training data provided there are no duplicate entries).  

