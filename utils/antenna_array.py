import numpy as np

def get_mimo_ula_steering_vector(theta:float, N:int, spacing:float=0.5) -> np.ndarray:
    """
    Get the steering vector for a given angle and number of antennas.
    Args:
        theta (float): The angle in degrees.
        N (int): The number of antennas.
        spacing (float, optional): The spacing between antennas. Defaults to 0.5.
    Returns:
        np.ndarray: The steering vector.
    """
    antenna_indices = np.arange(N).reshape(-1, 1)
    a_theta = np.exp(1j * 2 * np.pi * antenna_indices * spacing * np.sin(np.deg2rad(theta)))
    
    return a_theta

def get_fas_ula_steering_vector(theta:float, N:int, spacing:float) -> np.ndarray:
    """
    Get the steering vector for a FAS ULA.
    Args:
        theta (float): The angle of the steering vector.
        N (int): The number of antennas.
        spacing (float): The spacing between the antennas.
        return get_steering_vector(theta, N, spacing)
    """
    antenna_indices = np.arange(-(N - 1) / 2, (N - 1) / 2 + 1).reshape(-1, 1)
    a_theta = np.exp(1j * 2 * np.pi * antenna_indices * spacing * np.sin(np.deg2rad(theta)))    

    return a_theta

def get_mimo_upa_steering_vector(phi:float, theta:float, Nx:int, Nz:int, spacing=0.5):
    """
    Get the steering vector for a given angle and number of antennas.
    Args:
        phi (float): The azimuth angle in degrees.
        theta (float): The elevation angle in degrees.
        Nx (int): The number of antennas in the x direction.
        Nz (int): The number of antennas in the z direction.
        spacing (float, optional): The spacing between antennas. Defaults to 0.5.
    Returns:
        np.ndarray: The steering vector.
    """
    m = np.arange(Nx).reshape(-1, 1)
    a_x = np.exp(1j * 2 * np.pi * m * spacing * np.sin(np.deg2rad(phi)) * np.cos(np.deg2rad(theta)))
    n = np.arange(Nz).reshape(-1, 1)
    a_z = np.exp(1j * 2 * np.pi * n * spacing * np.sin(np.deg2rad(theta)))
    a_phi_theta = np.kron(a_x, a_z)
    
    return a_phi_theta