# combine_weights.py
import os

files = [
    'hilbert_scan.indices.bin',
    'uproject.weight.bin',
    'uproject.bias.bin',
    's4_1.log_dt.bin',
    's4_1.log_A_real.bin',
    's4_1.A_imag.bin',
    's4_1.C.bin',
    's4_1.D.bin',
    's4_2.log_dt.bin',
    's4_2.log_A_real.bin',
    's4_2.A_imag.bin',
    's4_2.C.bin',
    's4_2.D.bin',
    's4_3.log_dt.bin',
    's4_3.log_A_real.bin',
    's4_3.A_imag.bin',
    's4_3.C.bin',
    's4_3.D.bin',
    'fc.weight.bin',
    'fc.bias.bin'
]

with open('model_weights.bin', 'wb') as outfile:
    for fname in files:
        with open(f'model_params/{fname}', 'rb') as infile:
            outfile.write(infile.read())

print(" Combined model_weights.bin created!")