import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

def plot_vector_on_sphere(theta: float, phi: float) -> None:
    """
    Plot a vector on a unit sphere given spherical coordinates.
    
    Parameters:
    - theta: azimuthal angle in radians (0 to 2π)
    - phi: polar angle in radians (0 to π)
    
    Note: In physics, φ (phi) is often the azimuthal angle and θ (theta) is the polar angle,
    but in mathematics and many programming contexts, this convention is reversed.
    """

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    

    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    
    ax.plot_surface(x, y, z, color='lightblue', alpha=0.3)
    
    
    
    x_vec = np.sin(phi) * np.cos(theta)
    y_vec = np.sin(phi) * np.sin(theta)
    z_vec = np.cos(phi)
    

    ax.quiver(0, 0, 0, x_vec, y_vec, z_vec, color='red', arrow_length_ratio=0.1, linewidth=3)
    

    ax.scatter([x_vec], [y_vec], [z_vec], color='red', s=100)
    

    ax.quiver(0, 0, 0, 1.5, 0, 0, color='black', arrow_length_ratio=0.05, linewidth=2)
    ax.quiver(0, 0, 0, 0, 1.5, 0, color='black', arrow_length_ratio=0.05, linewidth=2)
    ax.quiver(0, 0, 0, 0, 0, 1.5, color='black', arrow_length_ratio=0.05, linewidth=2)
    ax.text(1.6, 0, 0, "X", fontsize=12)
    ax.text(0, 1.6, 0, "Y", fontsize=12)
    ax.text(0, 0, 1.6, "Z", fontsize=12)
    

    ax.text(x_vec*1.1, y_vec*1.1, z_vec*1.1, f"({x_vec:.2f}, {y_vec:.2f}, {z_vec:.2f})", fontsize=10)
    

    ax.text2D(0.05, 0.95, f"θ (theta) = {theta:.2f} rad\nφ (phi) = {phi:.2f} rad", 
              transform=ax.transAxes, fontsize=12)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Vector on Unit Sphere')
    

    ax.set_box_aspect([1, 1, 1])
    
    plt.tight_layout()
    plt.show()


def resolve_protocol() -> tuple[float, float]:
    """""
    Calculate the angles theta and phi for a specific vector on the unit sphere.
    """
    x1 = 1  
    x2 = 0
    y1 = 1*math.sqrt(3)
    y2 = 1*math.sqrt(3)
    c = math.sqrt(1/((complex(x1, x2)*complex(x1, -1*x2)).real + (complex(y1, y2)*complex(y1, -1*y2)).real))
    theta = 2*math.acos(c/x1)
    phi = math.asin(c*y2/math.sin(theta/2))
    return theta, phi

if __name__ == "__main__":

    theta, phi = resolve_protocol()
    plot_vector_on_sphere(theta, phi)
