import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

# Updated sensor positions for a narrower foot outline
sensor_positions = np.array([
    # Toe area
    [0.84, 0.2], [1.06, 0.2],  # Row 1 (2 sensors)

    # Ball of the foot
    [0.64, 0.4], [0.95, 0.4], [1.16, 0.4],  # Row 2 (3 sensors)
    [0.54, 0.6], [0.95, 0.6], [1.26, 0.6],  # Row 3 (3 sensors)

    # Midfoot
    [0.46, 0.8], [0.95, 0.8], [1.34, 0.8],  # Row 4 (3 sensors)
    [0.54, 1.0], [0.95, 1.0], [1.30, 1.0],  # Row 5 (3 sensors)

    # Heel area
    [0.74, 1.2], [1.16, 1.2]  # Row 6 (2 sensors)
])

# Example sensor values (replace with actual data)
sensor_values = np.array([
    1.0, 2.5,      # Row 1
    2.0, 3.0, 1.8, # Row 2
    1.5, 2.2, 3.1, # Row 3
    1.7, 3.4, 2.2, # Row 4
    2.0, 1.8, 2.5, # Row 5
    1.5, 2.2       # Row 6
])

# Normalize sensor values
sensor_values = (sensor_values - np.min(sensor_values)) / (np.max(sensor_values) - np.min(sensor_values))

# Create a grid for interpolation (size adjusted for the shape of the foot)
grid_x, grid_y = np.mgrid[0:2:100j, 0:1.4:100j]  # Adjusted for normalized sensor positions

# Interpolate sensor data to create a cohesive heatmap
grid_z = griddata(sensor_positions, sensor_values, (grid_x, grid_y), method='cubic')

# Set up figure and axis
fig, ax = plt.subplots()
ax.set_aspect('equal')

# Plot the heatmap
plt.imshow(grid_z.T, extent=(0, 2, 0, 1.4), origin='lower', cmap='hot', alpha=0.8)

# Plot sensor positions as dots
plt.scatter(sensor_positions[:, 0], sensor_positions[:, 1], color='blue', s=50, edgecolor='black', label='Sensors')

# Set plot limits and hide axis
plt.xlim(0, 2)
plt.ylim(0, 1.4)
ax.axis('off')

# Add title and legend
plt.title("Plantar Pressure Distribution Heatmap - Size 8 US Foot (Further Adjusted Width)")
plt.legend()

# Show the plot
plt.show()