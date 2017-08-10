import numpy as np
from scipy import integrate
from RSCalib import RSCalib
import matplotlib.pyplot as plt

class TSCalib:
    """トムソン較正用クラス
    """

    def __init__(self, LOADorCALC):
        """較正のための初期値設定
        """
        # 各諸元設定
        self.ntct = 100  # 温度計算の分割数
        self.nrat = 1000  # ??の分割数
        self.ll = 5  # フィルタ数
        self.maxch = 25  # 空間チャンネル数
        self.maxword = 440 #モノクロの較正波長数
        self.nfil = 5  # フィルタ数
        self.maxm = 10
        self.maxlaser = 2  # 最大レーザー台数
        self.maxfl = 10
        self.nlaser = 2  # 使用レーザー数
        self.m = 2
        self.tt = 297.15  # 較正時のガス温度
        self.maxdata = 160  # 最大取り込みチャンネル数
        self.inj_angle = np.pi / 9  # 入射角度[rad]
        self.worder = np.array([  # V792のデータ順を並び替える配列
                      0, 2, 4, 6, 8, 10,
                      12, 14, 16, 18, 20, 22,
                      24, 26, 28, 30, 1, 3,
                      5, 7, 9, 11, 13, 15,
                      17, 19, 21, 23, 25, 27,
                      32, 34, 36, 38, 40, 42,
                      44, 46, 48, 50, 52, 54,
                      56, 58, 60, 62, 33, 35,
                      37, 39, 41, 43, 45, 47,
                      49, 51, 53, 55, 57, 59,
                      64, 66, 68, 70, 72, 74,
                      76, 78, 80, 82, 84, 86,
                      88, 90, 92, 94, 65, 67,
                      69, 71, 73, 75, 77, 79,
                      81, 83, 85, 87, 89, 91,
                      96, 98, 100, 102, 104, 106,
                      108, 110, 112, 114, 116, 118,
                      120, 122, 124, 99, 97, 126,
                      101, 103, 105, 107, 109, 111,
                      113, 115, 117, 119, 121, 123,
                      128, 130, 132, 134, 136, 138,
                      140, 142, 144, 146, 148, 150,
                      152, 154, 156, 158, 129, 131,
                      133, 135, 137, 139, 141, 143,
                      145, 147, 149, 151, 153, 155,
                      29, 31, 61, 63, 93, 95, 125, 127, 157, 159])
        self.int_range = np.array([  # 各チャンネルの積分範囲、フィルターの透過波長領域に対応
                         360, 380,   #ch1
                         250, 350,   #ch2
                         10, 170,    #ch3
                         330, 370,   #ch4
                         140, 290,   #ch5
                         10, 400])   #ch6
        self.init_wlength = 685  # モノクロメータの初期波長[nm]
        #self.i1 = np.array([0, 0, 0, 0, 0, 3, 3, 3, 3, 1, 1, 1, 4, 4, 2])
        #self.i2 = np.array([3, 1, 4, 2, 5, 1, 4, 2, 5, 4, 2, 5, 2, 5, 5])
        self.i1 = np.array([0, 0, 0, 0, 3, 3, 3, 1, 1, 4])
        self.i2 = np.array([3, 1, 4, 2, 1, 4, 2, 4, 2, 2])
        self.num_ratio = int((self.nfil - 1) * self.nfil / 2)  # チャンネルの信号比の組み合わせ数
        self.te = np.exp(np.log(10000) * np.arange(self.ntct)/99)  # 計算温度範囲[eV] ntctと同数
        #self.PATH = '/Users/kemmochi/SkyDrive/Document/Study/Thomson/DATE/Polychrometer/Data/data of polychrometors for calibration/2016/'
        self.PATH = ''
        self.FILE_NAME = 'w_for_alignment_Aug2016.txt'
        self.LOADorCALC = LOADorCALC    #新たに較正データを読み込むか，計算済みのデータを読み込むか
        self.tfp = 2.0  #較正時の閾値
        self.pnconv = 9.6564e5

    def main(self):
        relte = self.cnt_photon_ltdscp()
        dTdR = self.cal_dTdR(relte)
        coft, cof = self.cal_cof(dTdR)

        np.savez("coft_cof_relte", coft=coft, cof=cof, relte=relte)

        return coft, cof, relte

    def sort_rawdata(self, st_raw, raw, worder):
        """V792のデータ順序を読み取りやすいように入れ替え
        """
        for i in range(self.maxdata):
            st_raw[:, i] = raw[:, worder[i]]

    def load_calib_values(self):
        """ポリクロ波長較正データの読み出し
        """
        st_rwdata = np.zeros((self.maxword, self.maxdata))
        clbdata = np.zeros((self.maxword, self.maxdata))

        for j in range(self.maxch):
            file_name = "Th_Raw_HJ" + str(j + 1) + ".txt"
            rwdata = np.loadtxt(self.PATH + file_name, comments='#')
            self.sort_rawdata(st_rwdata, rwdata, self.worder)
            clbdata[:, (self.nfil + 1) * j:(self.nfil + 1) * (j + 1) - 1] = st_rwdata[:,
                                                                            (self.nfil + 1) * j:(self.nfil + 1) * (
                                                                            j + 1) - 1]
        #            clbdata[:, 6*j:6*(j+1)-1] = st_rwdata[:, 6*j:6*(j+1)-1]

        return clbdata

    def clb_wo_offset(self):
        """較正データのオフセットを差し引く

        返り値: オフセットを差し引いたポリクロの波長感度
        """
        clbdata = self.load_calib_values()
        for i in range(self.maxdata):
            clbdata[:, i] -= np.average(clbdata[380:439, i])
        #        clbdata -= np.average(clbdata[380:439,:])
        return clbdata

    def cnt2vol(self):
        """取り込み値を電圧値に変換

        返り値: モノクロの光量で較正したポリクロの波長感度
        """
        light_power = np.loadtxt(self.PATH + self.FILE_NAME)
        light_power = light_power[np.newaxis, :]
        clbdata = clbdata / light_power.T
        #plt.plot(clbdata[:,1])
        #plt.show()
        np.save(self.PATH + 'clbdata.npy',clbdata)

        return clbdata

    def thomson_shape(self):
        """トムソン散乱断面積を計算

        返り値:
            w1: 各温度におけるトムソン散乱断面積のスペクトル(横440 x 縦160)
        """
        w1 = np.zeros((self.maxword, self.ntct))
        IS = np.cos(self.inj_angle)

        wlength = np.arange(self.maxword) + self.init_wlength  # 較正波長領域[nm]

        w2 = 1064 / wlength
        x = 51193 / self.te
        x = x[np.newaxis, :]

        #        K2 = ((x / (2 * 3.14159265358979)) ** 0.5) * np.exp(x) * (1 + (15 / (8 * x)))
        q = (1 - (1 - IS) / x.T) ** 2
        #        A = (1 - 2 * w2 * IS + w2 ** 2) ** (-0.5)
        #        B = (1 + ((w2 - 1) ** 2) / (2 * w2 * (1 - IS))) ** 0.5

        w1 = np.log(q) + 0.5 * np.log((x.T / (2 * np.pi))) + x.T + np.log(1 + (15 / (8 * x.T))) + 2 * np.log(
            w2) - x.T * np.sqrt(1 + ((w2 - 1) ** 2) / (2 * w2 * (1 - IS))) - 0.5 * np.log(1 - 2 * w2 * IS + w2 ** 2)
        w1 = np.exp(w1)

        #        plt.plot(w1[80,:])
        #        plt.contourf(np.log(w1+1))
        return w1

    def cnt_photon_ltdscp(self):
        """
        各温度にポリクロの各チャンネルに入ってくるフォトン数を計算
        "LOAD": 計算済みの較正データを読み込む
        "CALC": 新たにポリクロの較正データを読み込む
        """
        if(self.LOADorCALC == "LOAD"):
            clbdata = np.load(self.PATH + 'clbdata.npy')
        else:
            clbdata = self.cnt2vol()
        thomson_shape = self.thomson_shape()


#        int_clbdata = np.zeros((self.maxdata, self.ntct))
        int_clbdata = np.zeros((self.maxch, self.nfil+1, self.ntct))
#        relte = np.zeros((self.num_ratio * self.maxch, self.ntct))
        relte = np.zeros((self.maxch, self.num_ratio, self.ntct))

        ll = 0

        for i in range(self.ntct):
            for j in range(self.nfil + 1):
                for k in range(self.maxch):
                    buff = clbdata[:, (self.nfil + 1) * k + j] * thomson_shape[i, :]
                    int_clbdata[k, j, i] = integrate.trapz(
                        buff[self.int_range[2 * j]:self.int_range[2 * j + 1]])

        for i in range(self.ntct):
            for j in range(self.maxch):
                for k in range(self.num_ratio):
                    if(np.abs(int_clbdata[j, self.i2[k], i]) < 1.0e-35):
                        relte[j, k, i] = np.nan
                    else:
                        relte[j, k, i] = int_clbdata[j, self.i1[k], i]/int_clbdata[j, self.i2[k], i]
        #        return int_clbdata
#        plt.ylim(0, 1e3)
        #plt.plot(rrelte[:, 50])
#        plt.plot(relte[75, :])
#        plt.plot(relte[76, :])
#        plt.plot(relte[77, :])
#        plt.plot(relte[78, :])
#        plt.plot(relte[79, :])
        #plt.show()
        #rrelte = self.make_ratio_relte(relte)
        return relte

    def differ(self, data, n, dx, ddata):
        """差分を計算
        """
        ddata[0] = (data[1] - data[0]) / dx

        for i in range(1, n-1):
            ddata[i] = (data[i+1] - data[i-1])/(2*dx)

        ddata[n-1] = (data[n-1] - data[n-2]) / dx

    def cal_dTdR(self, relte):
#        dTdR = np.zeros((self.num_ratio * self.maxch, self.ntct))
        dTdR = np.zeros((self.maxch, self.num_ratio, self.ntct))
#        dTdR[:,:self.ntct-1] = ((self.te[1:]-self.te[:self.ntct-1])/(relte[:,1:]-relte[:,:self.ntct-1]))**2
        dTdR[:, :, :self.ntct-1] = ((self.te[1:]-self.te[:self.ntct-1])/(relte[:, :, 1:]-relte[:, :, :self.ntct-1]))**2

        return dTdR

    def cal_cof(self, dTdR):
#        coft = np.zeros((self.num_ratio * self.maxch, self.ntct))
#        cof = np.zeros((self.num_ratio * self.maxch, self.ntct))
        coft = np.zeros((self.maxch, self.num_ratio, self.ntct))
        cof = np.zeros((self.maxch, self.num_ratio, self.ntct))

        coft = self.te / dTdR
        cof = 1/dTdR

        return coft, cof

    def calc_calib(self):
        numfil = np.zeros((self.maxch, self.maxlaser))
        numfil = self.nlaser
        calib = np.zeros((self.maxch, self.maxlaser))
        sramd = np.arange(685, 1124, 1)
        RSC = RSCalib()
        calibfac = RSC.calib_Raman()
        calib = np.where(numfil == 0, calibfac, 0)
        np.where(np.abs(calibfac < 1.02-30, 0, self.pnconv/calibfac/self.tt))


if __name__ == "__main__":
    test = TSCalib(LOADorCALC="LOAD")
    test.main()

    print("Successfully Run the program !!")
