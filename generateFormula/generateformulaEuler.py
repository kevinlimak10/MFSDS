import numpy as np
import matplotlib.pyplot as plt

# Define Euler's formula function with frequency, phase, and amplitude
def euler_formula(x, amplitude=1, frequency=1, phase=0):
    return amplitude * (np.cos(frequency * x + phase) + 1j * np.sin(frequency * x + phase))

# Define a range of x values
x_values = np.linspace(0, 2 * np.pi, 1000)

# Parameters
amplitude = 2
frequency = 1
phase = np.pi / 4

# Compute euler formula values with the given parameters
euler_values = euler_formula(x_values, amplitude, frequency, phase)

# Extract real and imaginary parts
real_parts = np.real(euler_values)
imaginary_parts = np.imag(euler_values)

# Plotting
plt.figure(figsize=(10, 6))

# Plot real part
plt.subplot(2, 1, 1)
plt.plot(x_values, real_parts, label='Re($e^{i(fx + \phi)}$) = A cos(fx + \u03C6)', color='blue')
plt.title('Euler Formula with Amplitude, Frequency, and Phase')
plt.xlabel('x')
plt.ylabel('Re($e^{i(fx + \phi)}$)')
plt.legend()
plt.grid(True)

# Plot imaginary part
plt.subplot(2, 1, 2)
plt.plot(x_values, imaginary_parts, label='Im($e^{i(fx + \phi)}$) = A sin(fx + \u03C6)', color='red')
plt.xlabel('x')
plt.ylabel('Im($e^{i(fx + \phi)}$)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
