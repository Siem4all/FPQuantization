import numpy as np
import matplotlib.pyplot as plt

# Suppress scientific notation
np.set_printoptions(suppress=True)

# Generate randomly distributed parameters
weights = np.random.uniform(low=-127, high=127, size=50)

# Make sure important values are at the beginning for better debugging
weights[0] = weights.max() + 1
weights[1] = weights.min() - 1
weights[2] = 0

# Round each number to the second decimal place
weights = np.round(weights, 2)


def clamp(weights_q: np.array, lower_bound: int, upper_bound: int) -> np.array:
    weights_q[weights_q < lower_bound] = lower_bound
    weights_q[weights_q > upper_bound] = upper_bound
    return weights_q


def asymmetric_quantization(weights: np.array, bits: int) -> tuple[np.array, float, int]:
    # Calculate the scale and zero point
    alpha = np.max(weights)
    beta = np.min(weights)
    scale = (alpha - beta) / (2 ** bits - 1)
    zero = -1 * np.round(beta / scale)
    lower_bound, upper_bound = 0, 2 ** bits - 1
    # Quantize the parameters
    quantized = clamp(np.round(weights / scale + zero), lower_bound, upper_bound).astype(np.int32)
    return quantized, scale, zero


def asymmetric_dequantize(weights_q: np.array, scale: float, zero: int) -> np.array:
    return (weights_q - zero) * scale


def symmetric_dequantize(weights_q: np.array, scale: float) -> np.array:
    return weights_q * scale


def symmetric_quantization(weights: np.array, bits: int) -> tuple[np.array, float]:
    # Calculate the scale
    alpha = np.max(np.abs(weights))
    scale = alpha / (2 ** (bits - 1) - 1)
    lower_bound = -2 ** (bits - 1)
    upper_bound = 2 ** (bits - 1) - 1
    # Quantize the parameters
    quantized = clamp(np.round(weights / scale), lower_bound, upper_bound).astype(np.int32)
    return quantized, scale


def quantization_error(weights: np.array, weights_q: np.array):
    # calculate the MSE
    return abs(weights - weights_q)


(asymmetric_q, asymmetric_scale, asymmetric_zero) = asymmetric_quantization(weights, 8)
(symmetric_q, symmetric_scale) = symmetric_quantization(weights, 8)

print(f'Original:')
print(np.round(weights, 2))
print(f'Symmetric scale: {symmetric_scale}')
print(symmetric_q)
# Dequantize the parameters back to 32 bits
params_deq_symmetric = symmetric_dequantize(symmetric_q, symmetric_scale)
print(f'Dequantize Symmetric:')
print(np.round(params_deq_symmetric, 2))
# Calculate the quantization error
result=np.round(quantization_error(weights, params_deq_symmetric), 2)
# Plotting the discrete graph
plt.stem(weights, result, linefmt='b-', markerfmt='bo', basefmt='r-')
plt.xlabel('Weights')
plt.ylabel('Result (Weight - Weights_q)')
plt.title('Qunatization Error')
plt.grid(True)
plt.show()
