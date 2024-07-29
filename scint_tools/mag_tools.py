import numpy as np
from scipy import interpolate
import dateutil.parser as dp

def smooth_outliers(input_data):
    output_data = np.copy(input_data)
    invalid_indices = np.where(np.isclose(input_data, 99999.0))[0]
    for i in range(len(invalid_indices)):
        first_index = invalid_indices[i]
        next_index = np.copy(first_index)
        while np.isin(next_index, invalid_indices):
            next_index=next_index+1
            i = i+1
        output_data[first_index:next_index] = (output_data[first_index-1]+output_data[next_index+1])/2
    return output_data

def get_mag_data(fname, skiplines = 15, interp_outliers=False):
    with open(fname, 'r') as fp:
        for count, line in enumerate(fp):
            pass

    count = count+1
    timestamps_mag = np.zeros(count-skiplines)
    D_field = np.zeros(count-skiplines)
    H_field = np.zeros(count-skiplines)
    Z_field = np.zeros(count-skiplines)
    F_field = np.zeros(count-skiplines)

    line_number = 0
    with open(fname, 'r') as f:
        for line in f.readlines():
            if line_number>=skiplines:
                line = line.strip()
                line = line.split()
                time_info = line[0]+'T'+line[1]+'Z'
                parsed_t = dp.parse(time_info)
                t_in_seconds = parsed_t.strftime('%s')

                timestamps_mag[line_number-skiplines] = float(t_in_seconds)
                D_field[line_number-skiplines] = float(line[3])
                H_field[line_number-skiplines] = float(line[4])
                Z_field[line_number-skiplines] = float(line[5])
                F_field[line_number-skiplines] = float(line[6])
            line_number = line_number+1

    if interp_outliers:
        D_field = smooth_outliers(D_field)
        H_field = smooth_outliers(H_field)
        Z_field = smooth_outliers(Z_field)
        F_field = smooth_outliers(F_field)

    return timestamps_mag, D_field, H_field, Z_field, F_field

def get_mag_data_interp(fname, times_interp, skiplines = 15):
    with open(fname, 'r') as fp:
        for count, line in enumerate(fp):
            pass

    count = count+1
    timestamps_mag = np.zeros(count-skiplines)
    D_field = np.zeros(count-skiplines)
    H_field = np.zeros(count-skiplines)
    Z_field = np.zeros(count-skiplines)
    F_field = np.zeros(count-skiplines)

    line_number = 0
    with open(fname, 'r') as f:
        for line in f.readlines():
            if line_number>=skiplines:
                line = line.strip()
                line = line.split()
                time_info = line[0]+'T'+line[1]+'Z'
                parsed_t = dp.parse(time_info)
                t_in_seconds = parsed_t.strftime('%s')

                timestamps_mag[line_number-skiplines] = float(t_in_seconds)
                D_field[line_number-skiplines] = float(line[3])
                H_field[line_number-skiplines] = float(line[4])
                Z_field[line_number-skiplines] = float(line[5])
                F_field[line_number-skiplines] = float(line[6])
            line_number = line_number+1

    interp_func = interpolate.interp1d(timestamps_mag, D_field)
    D_i = interp_func(times_interp)

    interp_func = interpolate.interp1d(timestamps_mag, H_field)
    H_i = interp_func(times_interp)

    interp_func = interpolate.interp1d(timestamps_mag, Z_field)
    Z_i = interp_func(times_interp)

    interp_func = interpolate.interp1d(timestamps_mag, F_field)
    F_i = interp_func(times_interp)

    return D_i, H_i, Z_i, F_i