import os
import math
import time

REG_IFB_MPHY = 42
REG_PB_MPHY = 43
REG_GEN_EYESCAN = 63
CHANNEL_NUM = 2
INTF_ID = 0

try:
    import SBriPyMonitor as smon
    if not smon.connect(): raise RuntimeError()
except:
    print('Unable to load SBri module or connect to SBri monitor. Only simulation mode is supported.')

def decorate_address(param24, param16, addr):
    return addr | ((param16 & 0xff) << 16) | ((param24 & 0xff) << 24)
    
def generate_eyescan(vert_step, horz_step):
    smon.write(decorate_address(INTF_ID, CHANNEL_NUM, REG_GEN_EYESCAN), 0x10000 | (vert_step << 8) | horz_step, 'confettikanone')
    
def get_mphy_reg_val(mphy_reg):
    return smon.read(mphy_reg.get_address_decorated())
    
def set_mphy_reg_val(mphy_reg, val):
    smon.write(mphy_reg.get_address_decorated(), val, 'confettikanone')

def get_values_from_csv_line(line):
    vals = []
    for val in line.split(','):
        if val not in ['\n', '']: vals.append(float(val))
    return vals

def parse_eyediagram_file(file):
    x_values = []
    y_values = []
    with open(file, 'r') as file:
        x_values = get_values_from_csv_line(file.readline())
        y_values = get_values_from_csv_line(file.readline())
    return x_values, y_values