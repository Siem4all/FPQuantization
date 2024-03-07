import numpy as np
import os
import matplotlib.pyplot as plt
from SingleCntrSimulator import main, getAllValsFP, getAllValsF2P

# Call the functions with the parameters and store their results in a dictinary
getAllF2PAndFPVals = {'F2P': getAllValsF2P(flavor='lr', signed=True, cntrSize=7, hyperSize=2),
                      'FP': getAllValsFP(cntrSize=7, expSizes=[4], signed=True)}

# vector to quantize
vec2quantize = np.random.uniform(low=-200, high=200, size=90)

# color of the plots
colorOfFP = {'F2P': 'green', 'FP': 'blue'}
file_path = '../res/images/F2PvsFP.png'

class Quantizer:
    def __init__(self, vec2quantize, quantizedRange):
        self.vec2quantize = vec2quantize
        self.quantizedRange = quantizedRange
        self.vec2quantize = np.sort(self.vec2quantize)

        np.set_printoptions(suppress=True)

    def calcQuantRange(self, quantScheme):
        quantSchemeMax = np.max(quantScheme)
        quantSchemeMin = np.min(quantScheme)
        return [quantSchemeMin, quantSchemeMax]

    def clamp(self, vec2quantize_q, lower_bound, upper_bound):
        vec2quantize_q[vec2quantize_q < lower_bound] = lower_bound
        vec2quantize_q[vec2quantize_q > upper_bound] = upper_bound
        return vec2quantize_q

    def asymmetric_quantization(self):
        rmin, rmax = self.calcQuantRange(self.vec2quantize)
        qmin, qmax = self.calcQuantRange(self.quantizedRange)
        scale = (rmax - rmin) / (qmax - qmin)
        zero_point = qmin - (rmax / scale)
        quantized = self.clamp(np.round(self.vec2quantize / scale + zero_point), qmin, qmax).astype(
            np.int32)
        return quantized, scale, zero_point

    def symmetric_quantization(self):
        qmin, qmax = self.calcQuantRange(self.quantizedRange)
        scale = np.max(np.abs(self.vec2quantize)) / qmax
        quantized = self.clamp(np.round(self.vec2quantize / scale), qmin, qmax).astype(np.int32)
        return quantized, scale

    def symmetric_dequantize(self, vec2quantize_q, scale):
        return vec2quantize_q * scale

    def quantization_error(self, symmetric_deg):
        return abs(self.vec2quantize - symmetric_deg)

    def runSimulation(self):
        (symmetric_q, symmetric_scale) = self.symmetric_quantization()

        symmetric_deg = self.symmetric_dequantize(symmetric_q, symmetric_scale)

        error = self.quantization_error(symmetric_deg)
        return self.vec2quantize, error


# Create an instance of the Quantizer class
if __name__ == '__main__':
    main()
    for mode, quantizedRange in getAllF2PAndFPVals.items():
        quantizer = Quantizer(vec2quantize, quantizedRange)
        (vec2quantize, error)= quantizer.runSimulation()
        plt.stem(vec2quantize, error, linefmt=colorOfFP[mode], markerfmt=colorOfFP[mode], label=mode, basefmt='r-')

plt.xlabel('vec2quantize')
plt.ylabel('|Quantization error|')
plt.title('Symetric Quantization Error')
plt.grid(True)
plt.legend()
# Check if the file exists and delete it if found
if os.path.exists(file_path):
    os.remove(file_path)

# Save the plot as F2PvsFP.png in the res/images directory
plt.savefig(file_path)
plt.show()
