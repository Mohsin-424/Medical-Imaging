import matplotlib.pyplot as plt

# Symmetry axis for mirroring
symmetry_axis = 200

# Left foot sensor positions and their mappings
left_sensor_positions = [
     (183, 41, 'fsrReading7'), (220, 42, 'fsrReading6'), (220, 42, 'fsrReading6'),
     (164, 85, 'fsrReading15'), (196, 84, 'fsrReading8'), (225, 84, 'fsrReading5'),
     (160, 134, 'fsrReading14'), (192, 132, 'fsrReading11'), (224, 130, 'fsrReading4'),
     (165, 185, 'fsrReading13'), (193, 184, 'fsrReading10'), (220, 182, 'fsrReading0'),
     (171, 238, 'fsrReading12'), (191, 237, 'fsrReading9'), (215, 235, 'fsrReading1'),
    (176, 301, 'fsrReading2'), (208, 299, 'fsrReading3'), (208, 299, 'fsrReading3')
]

# Mirror left foot sensor positions to right foot
right_sensor_positions = [
    (2 * symmetry_axis - x, y, label) for x, y, label in left_sensor_positions
]

# Foot outline points for left foot
left_foot_outline = [
    (176, 301), (208, 299), (215, 235), (220, 182),
    (224, 130), (225, 84), (220, 42), (183, 41),
    (164, 85), (160, 134), (165, 185), (171, 238), (176, 301)
]

# Mirror left foot outline to right foot
right_foot_outline = [
    (2 * symmetry_axis - x, y) for x, y in left_foot_outline
]

# Create two separate plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))

# Plot for Left Foot
for x, y, label in left_sensor_positions:
    ax1.scatter(x, y, color='blue', s=100, label=label if label not in ax1.get_legend_handles_labels()[1] else "")
    ax1.text(x + 5, y, label, fontsize=8, color='darkblue')
outline_x, outline_y = zip(*left_foot_outline)
ax1.plot(outline_x, outline_y, linestyle='--', color='gray', alpha=0.7, label='Left Foot Outline')
ax1.invert_yaxis()  # Invert y-axis for correct orientation
ax1.set_title("Left Foot Sensor Mapping")
ax1.set_xlabel("X Position")
ax1.set_ylabel("Y Position")
ax1.grid(False)
ax1.axis('equal')

# Plot for Right Foot
for x, y, label in right_sensor_positions:
    ax2.scatter(x, y, color='red', s=100, label=label if label not in ax2.get_legend_handles_labels()[1] else "")
    ax2.text(x + 5, y, label, fontsize=8, color='darkred')
outline_x, outline_y = zip(*right_foot_outline)
ax2.plot(outline_x, outline_y, linestyle='--', color='lightgray', alpha=0.7, label='Right Foot Outline')
ax2.invert_yaxis()  # Invert y-axis for correct orientation
ax2.set_title("Right Foot Sensor Mapping")
ax2.set_xlabel("X Position")
ax2.set_ylabel("Y Position")
ax2.grid(False)
ax2.axis('equal')

# Show the plot
plt.tight_layout()
plt.show()
