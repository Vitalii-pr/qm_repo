import re
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
    

    ax.quiver(0, 0, 0, 0, 0, 1.5, color='black', arrow_length_ratio=0.05, linewidth=2)  # X-axis moves to Z
    ax.quiver(0, 0, 0, 1.5, 0, 0, color='black', arrow_length_ratio=0.05, linewidth=2)  # Z-axis moves to X
    ax.quiver(0, 0, 0, 0, 1.5, 0, color='black', arrow_length_ratio=0.05, linewidth=2)  # Y-axis remains Y
    
    ax.text(0, 0, 1.6, "X", fontsize=12)  # X is now in the Z direction
    ax.text(1.6, 0, 0, "Z", fontsize=12)  # Z is now in the X direction
    ax.text(0, 1.6, 0, "Y", fontsize=12)  # Y remains the same

    

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



def parse_equation(equation: str):
    """
    Parses a quantum state equation to extract x1, x2, y1, y2.
    Expected format: ψ = (x1 + i x2)|0⟩ + (y1 + i y2)|1⟩
    """
    # Regex to extract complex coefficients of |0⟩ and |1⟩
    matches = re.findall(r"\((-?\d*\.?\d*)\s*([\+\-])\s*i\s*(-?\d*\.?\d*)\)\|[01]⟩", equation)

    if len(matches) != 2:
        raise ValueError("Invalid equation format. Expected 2 complex coefficients.")


    def parse_complex(match):
        real, sign, imag = match
        imag = float(imag) * (-1 if sign == "-" else 1)  
        return float(real), imag

    (x1, x2) = parse_complex(matches[0])  
    (y1, y2) = parse_complex(matches[1]) 

    return x1, x2, y1, y2


def resolve_protocol(x1, x2, y1, y2) -> tuple[float, float]:
    """""
    Calculate the angles theta and phi for a specific vector on the unit sphere.
    """
    c = math.sqrt(1/((complex(x1, x2)*complex(x1, -1*x2)).real + (complex(y1, y2)*complex(y1, -1*y2)).real))
    theta = 2*math.acos(c/x1)
    phi = math.asin(c*y2/math.sin(theta/2))
    return theta, phi

if __name__ == "__main__":
    equation = "ψ = (3 + i 0)|0⟩ + (0 + i 1)|1⟩"
    x1, x2, y1, y2 = parse_equation(equation)
    print(f"Parsed coefficients: x1={x1}, x2={x2}, y1={y1}, y2={y2}")
    theta, phi = resolve_protocol(x1, x2, y1, y2)
    plot_vector_on_sphere(theta, phi)
