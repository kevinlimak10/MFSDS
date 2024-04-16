import numpy as np
import matplotlib.pyplot as plt

# Define the target image (plot)
x = np.linspace(-10, 10, 1000)
y_target = np.sin(x) + np.cos(2*x)

# Define initial parameters
num_iterations = 10
population_size = 50

# Define the fitness function
def fitness_function(y_actual, y_target):
    return np.mean(np.abs(y_actual - y_target))

# Generate initial population of formulas (random)
def generate_formula():
    a = np.random.uniform(-1, 1)
    b = np.random.uniform(-1, 1)
    c = np.random.uniform(-1, 1)
    return lambda x: a*np.sin(b*x) + c*np.cos(2*x)

# Optimization loop
best_formula = None
best_fitness = float('inf')

for _ in range(num_iterations):
    population = [generate_formula() for _ in range(population_size)]
    for formula in population:
        y_actual = formula(x)
        fitness = fitness_function(y_actual, y_target)
        if fitness < best_fitness:
            best_fitness = fitness
            best_formula = formula

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(x, y_target, label='Target')
plt.plot(x, best_formula(x), label='Generated')
plt.legend()
plt.title('Generated Formula vs Target')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
