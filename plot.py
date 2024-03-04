import numpy as np
import matplotlib.pyplot as plt

# Define the original floating-point weight
original_weight = np.linspace(-0.100, 0.100, 1000)

# Sort the original weights
original_weight_sorted = np.sort(original_weight)

# Define the number of bits for each quantized format
num_bits_format1 = 8
num_bits_format2 = 8
num_bits_int8 = 8

# Define the ranges for the different formats
min_float = -0.100
max_float = 0.100

# Calculate the scales for each format
scale_format1 = (2**(num_bits_format1 - 1) - 1) / max(abs(min_float), abs(max_float))
scale_format2 = (2**(num_bits_format2 - 1) - 1) / max(abs(min_float), abs(max_float))
scale_int8 = (2**(num_bits_int8 - 1) - 1) / max(abs(min_float), abs(max_float))

# Quantize a floating-point weight to each format
def quantize_format1(weight):
    return weight * scale_format1

def quantize_format2(weight):
    return weight * scale_format2

def quantize_int8(weight):
    return weight * scale_int8

# Calculate the quantization error for each format
quantized_weight_format1 = quantize_format1(original_weight_sorted)
quantized_weight_format2 = quantize_format2(original_weight_sorted)
quantized_weight_int8 = quantize_int8(original_weight_sorted)

# Calculate quantization errors for each format
quantization_error_format1 = original_weight_sorted - quantized_weight_format1
quantization_error_format2 = original_weight_sorted - quantized_weight_format2
quantization_error_int8 = original_weight_sorted - quantized_weight_int8

# Plotting the graph
plt.figure(figsize=(12, 6))
plt.plot(original_weight_sorted, quantization_error_format1, label='Format 1: 8 bits (1 sign bit, 3 exponent bits, 4 mantissa bits)')
plt.plot(original_weight_sorted, quantization_error_format2, label='Format 2: 8 bits (1 sign bit, 4 exponent bits, 3 mantissa bits)')
plt.plot(original_weight_sorted, quantization_error_int8, label='INT8')
plt.xlabel('Original Weight')
plt.ylabel('Quantization Error')
plt.title('Quantization Error vs. Original Weight')
plt.legend()
plt.grid(True)
plt.show()
