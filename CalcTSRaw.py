import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy import integrate

class CalcTSRaw:
    def __init__(self):
        pass

    def get_TSraw(self, filename):
        data = np.loadtxt(filename, delimiter=",", skiprows=18)
        return data

    def average_raw(self, numst, numed, isAvg="YES"):
        buf = np.zeros((10000,5))
        for i in range(numst, numed+1):
            filename = "/Users/kemmochi/SkyDrive/Document/Study/Fusion/RT1/Thomson/ScatteredLight/20170907/tek%04dALL.csv" % i
            data = self.get_TSraw(filename)
            buf += data
        buf /= numed-numst+1

        if isAvg == "YES":
           buf = self.rm_offset(buf)

        #buf[:,1] *= 0.02
        #buf[:,2] *= 0.01
        #buf[:,3] *= 0.02
        #buf[:,4] *= 0.01

        return buf

    def rm_offset(self, data):
        return data - np.mean(data[:4000, :], axis=0)

    def integrate_raw(self):
        pass


    def get_TSraw_csv(self):
        filename = "tek0098ALL.csv"
        f = open(filename, 'r')

        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            print(row)

        f.close()


if __name__ == "__main__":
    CTSR = CalcTSRaw()
    #scl_fix1_1 = CTSR.average_raw(10, 19, isAvg="YES")
    scl_fix1_2 = CTSR.average_raw(20, 29, isAvg="YES")
    scl_fix2 = CTSR.average_raw(33, 42, isAvg="YES")
    scl_fix3 = CTSR.average_raw(57, 66, isAvg="YES")
    scl_fix4 = CTSR.average_raw(71, 80, isAvg="YES")
    scl_fix5 = CTSR.average_raw(82, 91, isAvg="YES")
    #strl_0 = CTSR.average_raw(0, 9, isAvg="YES")
    #strl_1 = CTSR.average_raw(30, 32, isAvg="YES")
    strl_2 = CTSR.average_raw(44, 52, isAvg="YES")
    strl_3 = CTSR.average_raw(91, 101, isAvg="YES")
    #scl_1st = CTSR.average_raw(99, 99)
    #scl_fix1 = CTSR.average_raw(51, 60, isAvg="YES")
    #scl_fix2 = CTSR.average_raw(72, 81, isAvg="YES")
    #scl_fix3 = CTSR.average_raw(88, 97, isAvg="YES")
    #scl_fix4 = CTSR.average_raw(98, 107, isAvg="YES")
    #strl = CTSR.average_raw(60, 71, isAvg="YES")
    #strl_2 = CTSR.average_raw(108, 111)
    #strl_112 = CTSR.average_raw(112, 112)
    chnum = 1
    chname = ["chA", "chE", "chB", "chF"]
    int_raw = np.zeros(7)
    mag_i_max1 = np.array([0.0017796, 0.0004783, 0.00014029, 0.0011915, 0.00016026, 0, 0])
    IF2den = np.array([0.70564, 0.96754, 1.0938, 0.78616, 1.0898, 1, 1])
    int_raw[0] = np.sum(scl_fix1_2[5100:5500, chnum])
    int_raw[1] = np.sum(scl_fix2[5100:5500, chnum])
    int_raw[2] = np.sum(scl_fix3[5100:5500, chnum])
    int_raw[3] = np.sum(scl_fix4[5100:5500, chnum])
    int_raw[4] = np.sum(scl_fix5[5100:5500, chnum])
    int_raw[5] = np.sum(strl_2[5100:5500, chnum])
    int_raw[6] = np.sum(strl_3[5100:5500, chnum])
    int_raw = np.abs(int_raw)
    #plt.plot(mag_i_max1, int_raw, 'o', label='mag_i_max1')
    plt.figure(figsize=(8, 3))
    #plt.plot(IF2den, int_raw, 'o', ms=10, label='IF2den')
    #plt.plot(mag_i_max1/IF2den, int_raw, 'ro', ms=10, label='mag_i_max1/IF2den')
    plt.plot(mag_i_max1/(IF2den*IF2den), int_raw, 'ko', ms=10, label='mag_i_max1/IF2den^2')
    #plt.plot(mag_i_max1, int_raw, 'go', ms=10, label='mag_i_max1')
    #plt.xlabel("$\mathbf{n_eL [10^{17}m^{-3}]}$")
    plt.xlabel("mag_i_max1/IF2den^2")
    plt.ylabel("Signal")
    plt.legend()
    #plt.xlim(-0.00001,0.00014)
    plt.show()
    #print(int_scl_fix1_2)
    plt.figure(figsize=(10, 2.5))
    plt.subplots_adjust(bottom=0.2)
    #plt.plot(scl_fix1_1[:,0], scl_fix1_1[:, chnum], label=("fix1_1 %s" % chname[chnum-1]))
    #plt.plot(scl_fix1_2[:,0], scl_fix1_2[:, chnum], label=("fix1_2 %s" % chname[chnum-1]))
    #plt.plot(scl_fix2[:,0], scl_fix2[:, chnum], label=("fix2 %s" % chname[chnum-1]))
    #plt.plot(scl_fix3[:,0], scl_fix3[:, chnum], label=("fix3 %s" % chname[chnum-1]))
    #plt.plot(scl_fix4[:,0], scl_fix4[:, chnum], label=("fix4 %s" % chname[chnum-1]))
    #plt.plot(scl_fix5[:,0], scl_fix5[:, chnum], label=("fix5 %s" % chname[chnum-1]))
    #plt.plot(strl_0[:,0],strl_0[:, chnum], label=("w/o plasma_0 %s" % chname[chnum-1]))
    #plt.plot(strl_2[:,0],strl_2[:, chnum], label=("w/o plasma_2 %s" % chname[chnum-1]))
    #plt.plot(strl_3[:,0],strl_3[:, chnum], label=("w/o plasma_3 %s" % chname[chnum-1]))
    #plt.plot(scl_1st[:,0], scl_1st[:,1], label="scattered light(fix4 1shot)")
    #plt.plot(scl_fix1[:,0], scl_fix1[:, chnum], label=("fix1 %s" % chname[chnum-1]))
    #plt.plot(scl_fix2[:,0], scl_fix2[:, chnum], label=("fix2 %s" % chname[chnum-1]))
    #plt.plot(scl_fix3[:,0], scl_fix3[:, chnum], label=("fix3 %s" % chname[chnum-1]))
    #plt.plot(scl_fix4[:,0], scl_fix4[:, chnum], label=("fix4 %s" % chname[chnum-1]))
    #plt.plot(strl_1[:,0],strl_1[:, chnum], label=("w/o plasma %s" % chname[chnum-1]))
    #plt.plot(strl[:,0],strl[:, chnum], label=("w/o plasma %s" % chname[chnum-1]))
    #plt.plot(strl_112[:,0],strl_112[:,2], label="stray light 112")
    plt.xlabel("Time [sec]")
    plt.ylabel("Signal [V]")
    plt.xlim(-2e-6, 2e-6)
    plt.ylim(-0.01, 0.005)
    plt.legend()
    #filename = "figure/RT1_TS_20170907_%s_80av_rmoffset.pdf" % chname[chnum-1]
    #plt.savefig(filename)
    #plt.show()
