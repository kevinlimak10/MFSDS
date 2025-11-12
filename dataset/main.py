import numpy as np
import matplotlib.pyplot as plt
import uuid
import os

def plot_cartesian_function(func, x_range, key,func_str):
    """
    Plot a given function on a Cartesian graph and save it as an image.

    Parameters:
    - func: The function to be plotted.
    - x_range: A tuple (start, end) defining the range of x values for the plot.
    - filename: The filename to save the plot as an image.
    """
    x = np.linspace(x_range[0], x_range[1], 1000)
    y = func(x)

    plt.figure(figsize=(8, 6))
    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Plot of ' + func_str)
    plt.grid(True)
    plt.savefig(key)
    #plt.show()

    # Write to the text file (ensure path relative to this script)
    with open(os.path.join(os.path.dirname(__file__), 'function.txt'), 'a') as file:
        file.write(f"{key}={func_str}\n")

# Generate multiple function variations until function.txt has 100 lines

def make_function(amplitude, frequency, phase):
    def f(x):
        return amplitude * np.exp(1j * (2 * np.pi * frequency * x + phase))
    return f

# Paths
base_dir = os.path.dirname(__file__)
func_file = os.path.join(base_dir, 'function.txt')

# Count existing lines
try:
    with open(func_file, 'r') as fh:
        existing_lines = sum(1 for _ in fh)
except FileNotFoundError:
    existing_lines = 0

target_lines = 100
needed = max(0, target_lines - existing_lines)

# Define the range of x values
x_range = (-2*np.pi, 2*np.pi)

for _ in range(needed):
    amplitude = float(np.round(np.random.uniform(0.1, 1.0), 3))
    frequency = float(np.round(np.random.uniform(0.1, 0.9), 3))
    # choose phase term from a few variants similar to tan(pi)
    phase_choice = np.random.choice(['tan(pi)', 'sin(pi/4)', 'cos(pi/3)', 'tan(pi/2 - 0.1)', '0'])
    # Compute numeric phase value for function
    if phase_choice == 'tan(pi)':
        phase_val = np.tan(np.pi)
    elif phase_choice == 'sin(pi/4)':
        phase_val = np.sin(np.pi/4)
    elif phase_choice == 'cos(pi/3)':
        phase_val = np.cos(np.pi/3)
    elif phase_choice == 'tan(pi/2 - 0.1)':
        phase_val = np.tan(np.pi/2 - 0.1)
    else:
        phase_val = 0.0

    func = make_function(amplitude, frequency, phase_val)
    # Build display string similar to ".3*eˆ(1j*2*pi*.6*x+tan(pi))"
    func_str = f"{amplitude}*e^(1j*2*pi*{frequency}*x+{phase_choice})"
    # File name for output image
    key = os.path.join(base_dir, f"{uuid.uuid4()}.png")
    plot_cartesian_function(func, x_range, key, func_str)
