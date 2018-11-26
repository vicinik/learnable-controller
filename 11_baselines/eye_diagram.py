#!/usr/bin/python3

from mphy_regs import sbri_rx1_mphy_regs
import os
import math
import time

try:
    import SBriPyMonitor as smon
except:
    print('unable to load module SBriPyMonitor. exiting...')
    exit()
if not smon.connect():
    print('could not connect to sbri monitor')
    exit()

# RX1 MPHY manipulation: dev 1, Byte 2, pb->tx, ifb->rx

REG_IFB_MPHY = 42
REG_PB_MPHY = 43
REG_GEN_EYESCAN = 63
CHANNEL_NUM = 2
INTF_ID = 0

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

def avg(vals):
    return float(sum(vals)) / max(len(vals), 1)

def gather_eyescan_data(regs, logfile, wrstr):
    for reg in regs:
        print('{} : {}'.format(reg.to_string(), hex(get_mphy_reg_val(reg))))
    print('---')
    print('Generating eyescan...')
    generate_eyescan(16, 4)
    x_values, y_values = parse_eyediagram_file('../../../70_sscript_tc/release_regression_v2/last_eyescan.csv')
    logfile.write('{}{};{}\n'.format(wrstr,x_values,y_values))
    print('Avg. X: {}, Avg. Y: {}'.format(avg(x_values)*1000, avg(y_values)*1000))
    print('===')

def measure_all_permutations_recursively(regs, logfile, wrstr='', idx=0):
    reg = regs[idx]
    for i in reg.mphy_reg.val_range:
        wrstr_loc = wrstr + str(i) + ';'
        old_data = get_mphy_reg_val(reg)
        data = reg.get_value_decorated(old_data, i)
        set_mphy_reg_val(reg, data)
        if idx == len(regs) - 1:
            gather_eyescan_data(regs, logfile, wrstr_loc)
        else:
            measure_all_permutations_recursively(regs, logfile, wrstr_loc, idx+1)
  
def main():
    regs = [sbri_rx1_mphy_regs[1],sbri_rx1_mphy_regs[2],sbri_rx1_mphy_regs[4],sbri_rx1_mphy_regs[6]]
    logfile = open('2_input_system.csv', 'w')
    start = time.time()
    measure_all_permutations_recursively(regs, logfile)
    time_passed = time.time() - start
    logfile.close()

    print('All done! Time passed for all measurements: {}s'.format(time_passed))
    
    
if __name__=='__main__':
    main()