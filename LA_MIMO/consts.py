from utils.math import dBm2pow, dB2pow
import numpy as np
from Channel.mimo import MIMO_Channel
# np.random.seed(10)  # fix the random parameters

# --------------------- System Parameters ----------------------
NUM_OF_USERS = 3  # Communication user number
NUM_OF_ANTENNAS = 8  # Transmit antenna number
FRAME_LENGTH = 30  # ISAC signal frame length

# --------------------- Power Parameters -----------------------
POWER = dBm2pow(30)  # Transmit power 
CHANNEL_NOISE = dBm2pow(0)  # Communication noise power
SENSING_NOISE = dBm2pow(0)  # Radar sensing noise power
QOS_THRESHOLD = dB2pow(12)  # QoS threshold
REFLECTION_COEFFICIENT = dBm2pow(0) # Reflection coefficient

# --------------------- Targets Parameters ---------------------
DOA = np.deg2rad([-30, 0, 30]).reshape(1, -1)  # direction of arrival
NUM_OF_TARGETS = DOA.shape[1]  # Targets number

# --------------------- Steering Vector ------------------------
antenna_indices = np.arange(0, NUM_OF_ANTENNAS).reshape(-1, 1)
SPACING = 0.5
STEERING_VECTOR = np.exp(1j * 2 * np.pi * antenna_indices * SPACING * np.sin(DOA))
STEERING_VECTOR_DIFF = 1j * 2 * np.pi * antenna_indices * SPACING * np.cos(DOA) * STEERING_VECTOR

# --------------------- Channel --------------------------------
CHANNEL= MIMO_Channel(N=NUM_OF_ANTENNAS, N_user=NUM_OF_USERS).get_channel()

# --------------------- Parameters ----------------------
Parameters = {
    "K": NUM_OF_USERS,
    "M": NUM_OF_TARGETS,
    "N": NUM_OF_ANTENNAS,
    "Delta_d": SPACING,
    "L": FRAME_LENGTH,
    "spacing": SPACING,
    "P": POWER,
    "Gamma": QOS_THRESHOLD,
    "sigmaC2": CHANNEL_NOISE,
    "sigmaR2": SENSING_NOISE,
    "rc": REFLECTION_COEFFICIENT,
    "theta": DOA,
    "a": STEERING_VECTOR,
    "a_diff": STEERING_VECTOR_DIFF,
    "H": CHANNEL
}

