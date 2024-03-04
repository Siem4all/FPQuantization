import math
import random
import openpyxl

class Main:
    def __init__(self, numBits, exponentSize):
        self.numBits = numBits
        self.exponentSize = exponentSize
        self.mantisaSize = numBits - exponentSize - 1
        self.biasExponent = 2 ** (self.exponentSize - 1)

    def cnvrtBinaryToFP(self, weight):
        signvalue=int(weight[0:1], 2)
        exponentvalue=int(weight[1:self.exponentSize+1], 2)
        mantisaBits=weight[self.exponentSize+1:self.numBits]
        if(self.exponentSize>0):
            value = ((-1) ** signvalue) * 2 ** (exponentvalue - self.biasExponent) * (1 + sum([int(mantisaBits[i], 2) / (2 ** (i+1)) for i in range(self.mantisaSize)]))
        else:
            pass
        return value

    def calQuantizaionError(self, weight):

        qgrid = random.uniform(-1.00, 1.00)
        wvalue=self.cnvrtBinaryToFP(weight)
        sigmaiod = wvalue / 225
        return abs(qgrid-sigmaiod), sigmaiod


def signedIntToBinary(num, numBits):
    # Calculate the two's complement of the number
    if (num < 0):
        two_comp = (~abs(num) + 1) & ((1 << numBits) - 1)
        return format(two_comp, '0' + str(numBits) + 'b')
    else:
        format_specifier = '0' + str(numBits - 1) + 'b'
        binary = format(num & ((1 << numBits) - 1), format_specifier)
        return '0' + binary  # Append '0' at the beginning


if __name__ == '__main__':
    weights = []
    numBits = 8
    filename = 'results.xlsx'
    workbook = openpyxl.Workbook()
    sheet = workbook.active

    for num in range(-128, 127, 3):
        result = signedIntToBinary(num, numBits)
        weights.append(result)
    for exponent in [3, 4]:
        header = [f'{exponent}E', 'x']
        sheet.append(header)
        fp8=Main(numBits, exponent)
        for weight in weights:
           qerror=fp8.calQuantizaionError(weight)
           sheet.append(qerror)
    workbook.save(filename)
    print(f"Results have been written to {filename} successfully.")

