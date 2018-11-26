PB = 43
IFB = 42

class MphyReg:
    def __init__(self, name, address, val_range):
        self.name = name
        self.address = address
        self.val_range = val_range
        
class SbriMphyReg:
    def __init__(self, lane, board_num, device, bytesel, mphy_reg):
        self._lane = lane
        self._board_num = board_num
        self._device = device
        self._bytesel = bytesel
        self.mphy_reg = mphy_reg
        
    def _decorate_address(self, param24, param16, addr):
        return addr | ((param16 & 0xff) << 16) | ((param24 & 0xff) << 24)
        
    def get_address_decorated(self):
        return self._decorate_address(self._device, self.mphy_reg.address, self._board_num)
        
    def get_value_decorated(self, curr_data, new_data):
        return (curr_data & ~(0xFF << self._bytesel*8)) | ((new_data & 0xFF) << self._bytesel*8)
        
    def to_string(self):
        return '{}.{}'.format(self._lane, self.mphy_reg.name)
        
mphy_reg_dc_gain = MphyReg('dc_gain', 3, range(0,8))
mphy_reg_eq_data_rate = MphyReg('eq_data_rate', 4, range(0, 4))
mphy_reg_eq_dc_gain = MphyReg('eq_dc_gain', 5, range(0, 4))
mphy_reg_eq = MphyReg('eq', 6, range(0, 11))
mphy_reg_la_swing = MphyReg('la_swing', 7, range(0, 4))
mphy_reg_sig_thresh = MphyReg('sig_thresh', 8, range(0, 8))
mphy_reg_sig_glitchrm = MphyReg('sig_glitchrm', 9, range(0, 4))
mphy_reg_tx_swing = MphyReg('tx_swing', 11, range(0, 8))
mphy_reg_tx_deemp = MphyReg('tx_deemp', 12, range(0, 8))
mphy_reg_tx_slew = MphyReg('tx_slew', 13, range(0, 4))
mphy_reg_tx_emp_delay = MphyReg('tx_emp_delay', 14, range(0, 4))

sbri_rx1_mphy_regs = [
    SbriMphyReg('rx1', IFB, 1, 2, mphy_reg_dc_gain),
    SbriMphyReg('rx1', IFB, 1, 2, mphy_reg_eq_data_rate),
    SbriMphyReg('rx1', IFB, 1, 2, mphy_reg_eq_dc_gain),
    SbriMphyReg('rx1', IFB, 1, 2, mphy_reg_eq),
    SbriMphyReg('rx1', IFB, 1, 2, mphy_reg_la_swing),
    SbriMphyReg('rx1', IFB, 1, 2, mphy_reg_sig_thresh),
    SbriMphyReg('rx1', IFB, 1, 2, mphy_reg_sig_glitchrm),
    SbriMphyReg('rx1', PB, 1, 2, mphy_reg_tx_swing),
    SbriMphyReg('rx1', PB, 1, 2, mphy_reg_tx_deemp),
    SbriMphyReg('rx1', PB, 1, 2, mphy_reg_tx_slew),
    SbriMphyReg('rx1', PB, 1, 2, mphy_reg_tx_emp_delay),
]