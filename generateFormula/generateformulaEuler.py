import numpy as np
import matplotlib.pyplot as plt
import uuid

def plot_cartesian_function(func, x_range, key, func_str):
    """
    Plot a given function on a Cartesian graph and save it as an image.

    Parameters:
    - func: The function to be plotted.
    - x_range: A tuple (start, end) defining the range of x values for the plot.
    - key: The unique identifier for the plot image filename.
    - func_str: The string representation of the function.
    """
    x = np.linspace(x_range[0], x_range[1], 1000)
    y = func(x)

    plt.figure(figsize=(10, 6))

    # Plot real part
    plt.plot(x, np.real(y), label=f'Re({func_str})', color='blue')
    plt.xlabel('x')
    plt.ylabel(f'Re({func_str})')
    plt.legend()
    plt.grid(True)

    plt.savefig(f'{key}.png')

    with open('function.txt', 'a', encoding='utf-8') as file:
        file.write(f"{key} = {func_str}\n")


# Define the function you want to plot
def my_function(x):
    amplitude = 17  # Adjust amplitude as needed
    frequency = 11  # Adjust frequency as needed

    phase = np.pi ** 3  # Adjust phase as needed
    func_str = f'{amplitude} * exp(1j * ({frequency} * x + π^3))'
    return amplitude * np.exp(1j * (frequency * x + phase)), func_str

# Define the range of x values and the filename
x_range = (0, 2 * np.pi)
key = str(uuid.uuid4())

# Get the function values and its string representation
y, func_str = my_function(np.linspace(x_range[0], x_range[1], 1000))

# Call the function to plot and save the image
plot_cartesian_function(lambda x: my_function(x)[0], x_range, key, func_str)
