import numpy as np

def quantizeSimple(data, scaling):
    """
    computes the foward quantization similar to jpeg

    Inputs:
    data: the raw data to be quantized
    scaling: the scaling coeffs for each data point

    Output:
    quantized_data: integer values representing the quantized data
    """

    quantized_data = np.round(data/scaling)
    return quantized_data

def inverseQuantizeSimple(quantized,scaling):
    """
    computes the inverse quantization similar to jpeg

    Inputs:
    quantized: the integer values representing quantized data
    scaling: the scaling coeffs for each data point

    Output:
    recovered_data: approximated data after quantization
    """

    recovered_data = quantized * scaling
    return recovered_data

def test_quantization():
    """
    This function peforms forward and reverse quantization on some dummy data 
    and checks that the error falls within an acceptable range.
    """
    # make some dummy data
    data = 100*np.random.random(size=100)
    # make summy scaling coefficients
    scaling = np.random.randint(1, 11, size=100)
    # forward quantization
    data_quant = quantizeSimple(data,scaling)
    # reverse quantization
    data_recovered = inverseQuantizeSimple(data_quant,scaling)

    # assert that the difference is less than what the scaling would expect
    assert np.all(np.abs(data_recovered-data)<=scaling/2)


if __name__ == "__main__":
    test_quantization()

