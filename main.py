import math
import matplotlib.pyplot as plt
from methods import *
plt.style.use('seaborn-poster')
# %matplotlib inline

# Define parameters
f = lambda x, y: (x + y) ** 2  # y'
y = lambda x: math.tan(x) - x  # analytical solution
h = 0.1  # Step size
x_grid = np.arange(0, 1 + h, h)  # Numerical grid
s0 = 0  # Initial Condition


# Calculations
exact_values = exact(y, x_grid)
euler_values = euler(f, x_grid, h, s0)
euler_cauchy_values = euler_cauchy(f, x_grid, h, s0)
runge_kutta_values = runge_kutta4(f, x_grid, h, s0)
# for Adams method we take initial values from RK run since it is the most precise
adams_values = adams(f, x_grid, h, runge_kutta_values[0:4])

plt.figure(figsize=(12, 8))
plt.plot(x_grid, euler_values, 'co--', label='Euler method')
plt.plot(x_grid, euler_cauchy_values, 'bo--', label='Euler-Cauchy method')
plt.plot(x_grid, runge_kutta_values, 'ro--', label='Runge-Kutta of 4th order')
plt.plot(x_grid, adams_values, "ko--", label="Adams method")
plt.plot(x_grid, exact_values, 'g', label='Exact solution')
plt.title('Approximate and Exact Solution for Simple ODE')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid()
plt.legend(loc='lower right')
plt.show()


