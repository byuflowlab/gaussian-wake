import numpy as np
import cPickle as pickle

if __name__ == "__main__":

    filename = "NREL5MWCPCT.p"

    data = pickle.load(open(filename, "rb"))
    cpct_data = np.zeros([data['wind_speed'].size, 3])
    cpct_data[:, 0] = data['wind_speed']
    cpct_data[:, 1] = data['CT']
    cpct_data[:, 2] = data['CP']

    print data

    np.savetxt('NREL5MWCPCT_dict.txt', np.c_[cpct_data[:, 0], cpct_data[:, 1], cpct_data[:, 2]], header='wind speed, CP, CT')