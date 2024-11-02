import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from matplotlib.animation import FuncAnimation

# Updated sensor positions for a narrower and elongated outline
narrow_factor = 0.7  # Factor to reduce the width of the feet

# Original positions for left foot sensors
left_sensor_positions = np.array([
    # Toe area
    [0.84 * narrow_factor, 0.2], [1.06 * narrow_factor, 0.2],  # Row 1 (2 sensors)

    # Ball of the foot
    [0.64 * narrow_factor, 0.4], [0.95 * narrow_factor, 0.4], [1.16 * narrow_factor, 0.4],  # Row 2 (3 sensors)
    [0.54 * narrow_factor, 0.6], [0.95 * narrow_factor, 0.6], [1.26 * narrow_factor, 0.6],  # Row 3 (3 sensors)

    # Midfoot
    [0.46 * narrow_factor, 0.8], [0.95 * narrow_factor, 0.8], [1.34 * narrow_factor, 0.8],  # Row 4 (3 sensors)
    [0.54 * narrow_factor, 1.0], [0.95 * narrow_factor, 1.0], [1.30 * narrow_factor, 1.0],  # Row 5 (3 sensors)

    # Heel area
    [0.74 * narrow_factor, 1.2], [1.16 * narrow_factor, 1.2]  # Row 6 (2 sensors)
])

# Right foot positions, mirroring the left foot positions
right_sensor_positions = np.array([
    # Mirrored positions for right foot (symmetric copy)
    [2.4 - pos[0], pos[1]] for pos in left_sensor_positions
])

# Create a figure and axis for the plot
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_aspect('equal')

# Set plot limits and hide axis
plt.xlim(0, 4.8)  # Updated to fit both feet
plt.ylim(0, 1.4)
ax.axis('off')

# Title and legend
plt.title("Real-time Plantar Pressure Distribution Heatmap - Size 8 US Feet (Narrowed and Symmetric)")

# Initialize the heatmap
heatmap_left = ax.imshow(np.zeros((100, 100)), extent=(0, 2.4, 0, 1.4), origin='lower', cmap='hot', alpha=0.8)
heatmap_right = ax.imshow(np.zeros((100, 100)), extent=(2.4, 4.8, 0, 1.4), origin='lower', cmap='hot', alpha=0.8)

# Plot sensor positions as dots for both feet
ax.scatter(left_sensor_positions[:, 0], left_sensor_positions[:, 1], color='blue', s=50, edgecolor='black', label='Left Foot Sensors')
ax.scatter(right_sensor_positions[:, 0] + 2.4, right_sensor_positions[:, 1], color='green', s=50, edgecolor='black', label='Right Foot Sensors')
plt.legend()

def read_sensor_data():
    # Simulated data reading - replace this with your actual sensor reading logic
    return np.random.rand(len(left_sensor_positions))

def update(frame):
    # Read the latest sensor values
    sensor_values = read_sensor_data()

    # Normalize sensor values
    normalized_values = (sensor_values - np.min(sensor_values)) / (np.max(sensor_values) - np.min(sensor_values))

    # Create grids for interpolation
    grid_x, grid_y = np.mgrid[0:2.4:100j, 0:1.4:100j]  # Adjusted for the shape of both feet
    grid_z_left = griddata(left_sensor_positions, normalized_values, (grid_x, grid_y), method='cubic')
    grid_z_right = griddata(right_sensor_positions, normalized_values, (grid_x, grid_y), method='cubic')

    # Update the heatmap data
    heatmap_left.set_array(grid_z_left.T)
    heatmap_right.set_array(grid_z_right.T)

# Use FuncAnimation to update the plot in real-time
ani = FuncAnimation(fig, update, frames=np.arange(0, 100), interval=200)  # Update every 200 ms

plt.show()
