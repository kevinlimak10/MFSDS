import numpy as np
import matplotlib.pyplot as plt
import uuid

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

    # Write to the text file
    with open('function.txt', 'a') as file:
        file.write(f"{key}={func_str}\n")

# Example usage:
# Define the function you want to plot
def my_function(x):
    frequency = .4  # Adjust frequency as needed
    amplitude = 1  # Adjust amplitude as needed
    phase = np.tan(np.pi)
    complex_sine_wave = amplitude * np.exp(1j * (2 * np.pi * frequency * x + phase))
    return complex_sine_wave

# Define the range of x values and the filename
x_range = (-2*np.pi, 2*np.pi)
key = str(uuid.uuid4())

# Call the function to plot and save the image
plot_cartesian_function(my_function, x_range, key, '.3*eˆ(1j*2*pi*.6*x+tan(pi)')
