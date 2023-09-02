import numpy as np
import matplotlib.pyplot as plt

def plot_left_right_grid(x_left_list, y_left_list, x_right_list, y_right_list):
    """
    Plot a grid of left and right plots with separate x and y data for each plot.

    Parameters:
    - x_left_list: List of x-axis data for left plots.
    - y_left_list: List of data for left plots.
    - x_right_list: List of x-axis data for right plots.
    - y_right_list: List of data for right plots.
    """
    nrows = max(len(x_left_list),len(x_right_list))
    # Create subplots with a 3x3 grid layout
    #sharex mismo eje x
    fig, axes = plt.subplots(nrows=nrows, ncols=3, figsize=(12, 8), sharex=True)

    # Plot the left plots in positions [0,0], [1,0], [2,0]
    for i in range(len(x_left_list)):
        axes[i, 0].scatter(x_left_list[i], y_left_list[i], label=f'Left Plot {i + 1}',s=5)
        axes[i, 0].set_title(f'Left Plot {i + 1}')
        axes[i, 0].set_xlabel(f'X-axis (left {i + 1})')
        axes[i, 0].set_ylabel(f'Y-axis (left {i + 1})')
        axes[i, 0].grid(True)
        axes[i, 0].legend()

    # Plot the right plots in positions [2,0], [2,1], [2,2]
    for i in range(len(x_right_list)):
        axes[i, 2].scatter(x_right_list[i], y_right_list[i], label=f'Right Plot {i + 1}',s=3)
        axes[i, 2].set_title(f'Right Plot {i + 1}')
        axes[i, 2].set_xlabel(f'X-axis (right {i + 1})')
        axes[i, 2].set_ylabel(f'Y-axis (right {i + 1})')
        axes[i, 2].grid(True)
        axes[i, 2].legend()

    # Adjust subplot spacing
    plt.tight_layout()

    # Show the plots
    plt.show()

if __name__ == "__main__":
    # Sample data
    x_left_list = [np.linspace(0, 10, 100)] * 5  # List of X-axis data for left plots
    y_left_list = [np.sin(x_left_list[0])] * 5  # List of data for left plots
    x_right_list = [np.linspace(0, 8, 100), np.linspace(0, 6, 100), np.linspace(0, 4, 100)]  # List of X-axis data for right plots
    y_right_list = [np.cos(x_right_list[0]), np.cos(x_right_list[1]), np.cos(x_right_list[2])]  # List of data for right plots

    # Call the function to plot the grid
    plot_left_right_grid(x_left_list, y_left_list, x_right_list, y_right_list)
