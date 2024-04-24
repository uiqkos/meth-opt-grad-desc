import numpy as np
import plotly.graph_objects as go
import sympy
from functools import wraps

from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def tupled(f):
    @wraps(f)
    def wrapper(args):
        return f(*args)

    return wrapper


def derivative(f, vars):
    diffs = [tupled(sympy.lambdify(vars, f.diff(var), 'numpy')) for var in vars]
    return diffs


def coefs_from_dict(coefs: dict[str, float]) -> list[float]:
    return [
        coefs.get("x^2", 0),
        coefs.get("x", 0),
        coefs.get("y^2", 0),
        coefs.get("y", 0),
        coefs.get("xy", 0),
        coefs.get("1", 0)
    ]


def gen_func(coefs: np.ndarray | list[float] | dict[str, float]):
    if isinstance(coefs, dict):
        coefs = coefs_from_dict(coefs)

    def func(t):
        return (
            coefs[0] * t[0] ** 2 +
            coefs[1] * t[0] +
            coefs[2] * t[1] ** 2 +
            coefs[3] * t[1] +
            coefs[4] * t[0] * t[1] +
            coefs[5])

    return func


def R2_derivatives(coefs: np.ndarray | list[float] | dict[str, float]):
    if isinstance(coefs, dict):
        coefs = coefs_from_dict(coefs)

    return [
        gen_func(np.array([0, coefs[0] * 2, 0, coefs[4], 0, coefs[1]])),
        gen_func(np.array([0, coefs[4], 0, coefs[2] * 2, 0, coefs[3]])),
    ]


def plot_func(f, path=(), limit=10, label=''):
    if len(path) == 0:
        xmin = -limit
        xmax = limit
        ymin = -limit
        ymax = limit
    else:
        xmin = np.array(path)[:, 0].min() * 1.5
        xmax = np.array(path[:, 0]).max() * 1.5
        ymin = np.array(path)[:, 1].min() * 1.5
        ymax = np.array(path[:, 1]).max() * 1.5

        xmin = max(abs(xmin), abs(xmax))
        xmax = -xmin

        ymin = max(abs(ymin), abs(ymax))
        ymax = -ymin

    x = np.linspace(xmin, xmax, 1000)
    y = np.linspace(ymin, ymax, 1000)

    X, Y = np.meshgrid(x, y)

    Z = f([X, Y])

    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y)])

    # Extract x, y, and z coordinates from the path
    path_x = []
    path_y = []
    path_z = []

    for point in path:
        # if -limit < f(point) < limit:
        path_x.append(point[0])
        path_y.append(point[1])
        path_z.append(f(point))

    if len(path) > 0:
        zmin = min(path_z)
        zmax = max(path_z)
        #
        fig.update_scenes(zaxis=dict(range=[zmin, zmax]))

    fig.update_scenes(zaxis=dict(range=[-limit, +limit]))
    # Add the path as a scatter plot on the surface plot
    fig.add_trace(go.Scatter3d(x=path_x, y=path_y, z=path_z,
                               mode='markers+lines', marker=dict(size=5),
                               name='Path'))

    fig.update_layout(title=label, autosize=False,
                      width=500, height=500,
                      margin=dict(l=65, r=50, b=65, t=90))

    # color bar limit
    fig.update_coloraxes(colorbar=dict(title='Z value', tickvals=[-limit, 0, limit], ticktext=[-limit, 0, limit]))

    # latex function label
    fig.update_layout(scene=dict(xaxis_title='x',
                                 yaxis_title='y',
                                 zaxis_title='z'),
                      scene_aspectmode='cube')

    return fig


def call_counter(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        wrapper.calls += 1
        return f(*args, **kwargs)

    wrapper.calls = 0

    return wrapper


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_func_with_path_matplotlib(f, path, limit=10):
    # Define the range for x and y values
    x = np.linspace(-limit, limit, 100)
    y = np.linspace(-limit, limit, 100)

    # Create a meshgrid for x and y values
    X, Y = np.meshgrid(x, y)

    # Apply the function to every point in the meshgrid
    Z = f([X, Y])

    # Create a 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')

    # Extract x, y, and z coordinates from the path
    path_x = [point[0] for point in path]
    path_y = [point[1] for point in path]
    path_z = [f(point) for point in path]

    # Plot the path
    ax.scatter(path_x, path_y, np.array(path_z), color='r', s=10)  # Red dots
    ax.plot(path_x, path_y, path_z, color='r')  # Connect dots with red line

    # Set labels and title
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_title('3D Function Plot with Path')

    plt.show()


def plot_2d_with_color(f, path=(), limit=10):
    # Define the range for x and y values
    x = np.linspace(-limit, limit, 100)
    y = np.linspace(-limit, limit, 100)

    # Create a meshgrid for x and y values
    X, Y = np.meshgrid(x, y)

    # Apply the function to every point in the meshgrid
    Z = f([X, Y])

    # Create a 2D plot
    plt.figure()

    # Plot the function values with color
    plt.contourf(X, Y, Z, levels=100, cmap='viridis')
    plt.colorbar(label='Z value')

    # Extract x and y coordinates from the path
    path_x = [point[0] for point in path]
    path_y = [point[1] for point in path]

    # Plot the path
    plt.scatter(path_x, path_y, color='r', s=50)  # Red dots
    plt.plot(path_x, path_y, color='r')  # Connect dots with red line

    # Set labels and title
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')
    plt.title('2D Function Plot with Path and Color')

    plt.show()


from matplotlib.colors import LinearSegmentedColormap


def get_color(index, total_triangles):
    colors = np.linspace(0, 1, total_triangles)  # Creates a gradient scale from 0 to 1
    return f"hsl({colors[index] * 240}, 100%, 50%)"  # Hue ranges from blue to red


def create_gradient_colormap(n_colors):
    """
    Create a colormap with a gradient from green to red.

    Parameters:
    - n_colors: The number of discrete colors in the gradient.

    Returns:
    - A LinearSegmentedColormap instance.
    """
    # Define start and end colors in RGB.
    start_color = (0, 1, 0)  # Green
    end_color = (1, 0, 0)  # Red

    # Generate the colors for the gradient.
    colors = [np.linspace(start_color[i], end_color[i], n_colors) for i in range(3)]
    colors = np.stack(colors, axis=1)

    # Create the colormap.
    cmap = LinearSegmentedColormap.from_list("custom_gradient", colors)

    return cmap


def plot_2d_and_3d_side_by_side(f, path, limit=10, title='', save=False, issimplex=False, text=''):
    x = np.linspace(-limit, limit, 100)
    y = np.linspace(-limit, limit, 100)
    X, Y = np.meshgrid(x, y)
    Z = f([X, Y])

    fig = plt.figure(figsize=(25, 8))  # Adjusted figure size to accommodate the third plot

    # 2D plot
    ax1 = fig.add_subplot(131)
    contour = ax1.contourf(X, Y, Z, levels=100, cmap='viridis')
    fig.colorbar(contour, ax=ax1, label='Z value')

    if issimplex:
        cmap = create_gradient_colormap(len(path))
        for i, simplex in enumerate(path):
            triangle_points = [point for point in simplex]
            x_points = [point[0] for point in triangle_points]
            y_points = [point[1] for point in triangle_points]
            x_points.append(triangle_points[0][0])  # Closing the triangle by adding the first point at the end
            y_points.append(triangle_points[0][1])
            ax1.plot(x_points, y_points, color=cmap(i / 9), alpha=0.5)
    elif len(path) > 0:
        path_x = [point[0] for point in path]
        path_y = [point[1] for point in path]
        ax1.scatter(path_x, path_y, color='r', s=50)
        ax1.plot(path_x, path_y, color='r')

    ax1.set_xlabel('X Axis')
    ax1.set_ylabel('Y Axis')
    ax1.set_title(title)

    # 3D plot
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.2)
    if issimplex:
        cmap = create_gradient_colormap(len(path))
        for i, simplex in enumerate(path):
            triangle_points = [[point[0], point[1], f(point)] for point in simplex]
            triangle = Poly3DCollection([np.array(triangle_points)], color=cmap(i / 9), alpha=0.5)
            ax2.add_collection3d(triangle)
    elif len(path) > 0:
        path_z = np.array([f(point) for point in path]).flatten()
        ax2.scatter(path_x, path_y, path_z, color='r', s=50)
        ax2.plot(path_x, path_y, path_z, color='r')
    ax2.set_xlabel('X Axis')
    ax2.set_ylabel('Y Axis')
    ax2.set_zlabel('Z Axis')

    # Text block plot
    ax3 = fig.add_subplot(133)
    ax3.axis('off')
    ax3.text(0.5, 0.5, text, ha='center', va='center', fontsize=12,
             bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10})

    plt.tight_layout()
    if save:
        plt.savefig(f'images/{title}.png')
    else:
        plt.show()
