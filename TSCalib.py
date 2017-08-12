import numpy as np
from scipy import integrate
from RSCalib import RSCalib
import matplotlib.pyplot as plt
np.seterr(divide='ignore', invalid='ignore')

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
        self.m = 2

    def main(self):
        clbdata, relte = self.cnt_photon_ltdscp()
        dTdR = self.cal_dTdR(relte)
        coft, cof = self.cal_cof(dTdR)

        np.savez("coft_cof_relte", coft=coft, cof=cof, relte=relte)

        self.calc_calib(clbdata)

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
        clbdata = self.clb_wo_offset()
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
        return clbdata, relte

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

    def calc_calib(self, clbdata):
        tfp = 2.0   #threshold parameter
        numfil = np.ones((self.maxch, self.maxlaser))
        numfil *= self.nlaser
        calib = np.zeros((self.maxch, self.maxlaser))
        sramd = np.arange(self.init_wlength, self.init_wlength+self.maxword, 1)
        RSC = RSCalib()
        calibfac, pcof = RSC.calib_Raman()
        calib = np.where(numfil == 0, calibfac, 0)
        np.where(np.abs(calibfac) < 1.02-30, 0, self.pnconv/calibfac/self.tt)
        cross = self.get_cross()
        ramd_max, ramd_min = self.set_ramd_range()
        clbdata = clbdata[:, :self.maxdata-10].reshape((self.maxword, self.maxch, self.nfil+1))   #(Adata[:, :150], (3, 6, 25))
        dfilter = self.red_filter(clbdata, sramd)

        calib[:, :] = 0.0
        for ilaser in range(self.nlaser):
            for ich in range(self.maxch):
                for iline in range(len(cross[:,0])):
                    p = self.getNearestValue(sramd, cross[iline, 2])
                    temp = calibfac[ich, ilaser] * cross[iline, 1]
                    if(numfil[ich, ilaser] != 0):
                        rmin = ramd_min[0, ich]
                        rmax = ramd_max[0, ich]
                        if(cross[iline, 2] < rmin or cross[iline, 2] > rmax):
                            temp = 0.0
                        else:
                            tempf = self.splint(sramd, clbdata[:, ich, 0], dfilter[:, ich, 0], self.maxword, cross[iline, 2])
                            temp *= clbdata[p, ich, 0]
                            if(tempf < tfp):
                                tempf = 0.0
                                temp *= tempf
                        temp /= self.poly(pcof, self.m, cross[iline, 2])
                        calib[ich, ilaser] = calib[ich, ilaser] + temp

        return calib


    def red_filter(self, clbdata, sramd):
        dfilter = np.zeros((self.maxword, self.maxch, self.nfil+1))
        for ich in range(self.maxch):
            yp1 = 0.0
            ypn = 0.0
            for ifil in range(self.nfil):
                 self.spline(sramd, clbdata[:, ich, ifil], self.maxword, yp1, ypn, dfilter[:, ich, ifil])

        return dfilter


    def poly(self, coef, n, x):
        poly = 0.0
        for i in range(n):
            poly += coef[i]*x**i

        return poly


    def splint(self, xa, ya, y2a, n, x):
        klo = 0
        khi = n
        while khi - klo > 1:
            k = (khi + klo)/2.0
            if(xa[k] > x):
                khi = k
            else:
                klo = k

        h = xa[khi] - xa[klo]
        a = (xa[khi] - x)/h
        b = (x - xa[klo])/h
        if(np.abs(y2a[klo]) > 1.0e33):
            y = (ya[klo] + ya[khi])/2.0

        y = a*ya[klo] + b*ya[khi] + ((a**3 - a)*y2a[klo] + (b**3-b)*y2a[khi])*h**2/6.0

        return y


    def spline(self, x, y, n, yp1, ypn, y2):
        u = x
        if(yp1 > 0.99e30):
            y2[0] = 0.0
            u[0] = 0.0
        #else:
        elif(x[1]-x[0] != 0):
            y2[0] = -0.5
            u[0] = (3.0/(x[1] - x[0]))*((y[1] - y[0])/(x[1] - x[0]) - yp1)

        sig = (x[1:n-2] - x[0:n-3])/(x[2:n-1] - x[0:n-3])
        p = sig*y2[0:n-3] + 2.0
        y2[1:n-2] = (sig - 1.0)/p
        u[1:n-2] = (y[2:n-1] - y[1:n-2])/(x[2:n-1] - x[1:n-2]) - (y[1:n-2] - y[0:n-3])/(x[1:n-2] - x[0:n-3])
        u[1:n-2] = (6.0*u[1:n-2]/(x[2:n-1] - x[0:n-3]) - sig*u[0:n-3])/p

        if(ypn > 0.99e30):
            qn = 0.0
            un = 0.0
        else:
            qn = 0.5
            un = (3.0/(x[n-1] - x[n-2]))*(ypn - (y[n-1] - y[n-2])/(x[n-1] - x[n-2]))

        y2[n-1] = (un - qn*u[n-2])/(qn*y2[n-2] + 1.0)

        y2[0:n-2] = y2[0:n-2]*y2[1:n-1] + u[0:n-2]


    def get_cross(self):
        """
        窒素のラマン散乱ラインを読み込む
        :return:
        cross[:, 1]: 散乱断面積
        cross[:, 2]: 波長
        """
        file_path = ""
        file_name = "cross.dat"
        cross = np.loadtxt("cross.dat", skiprows=3)

        #plt.plot(cross[:, 2], cross[:, 1])
        #plt.show()

        return cross


    def getNearestValue(self, list, num):
        """
        概要: リストからある値に最も近い値を返却する関数
        ThAnalysTeNe.pyにも同じ関数あり
        @param list: データ配列
        @param num: 対象値
        @return 対象値に最も近い値
        """

        # リスト要素と対象値の差分を計算し最小値のインデックスを取得
        buf_list = np.where(list != list, 0, list)
        #idx = np.abs(np.asarray(list[10:]) - num).argmin()
        #idx = np.abs(np.asarray(list) - num).argmin()
        idx = np.abs(np.asarray(buf_list) - num).argmin()
        return idx  #list[idx]

    def set_ramd_range(self):
        ramd_max = np.zeros((self.nfil, self.maxch))
        ramd_min = np.zeros((self.nfil, self.maxch))

        ramd_max[0, :] = 1060
        ramd_max[1, :] = 1025
        ramd_max[2, :] = 845
        ramd_max[3, :] = 1048
        ramd_max[4, :] = 964

        ramd_min[0, :] = 1050
        ramd_min[1, :] = 952
        ramd_min[2, :] = 696
        ramd_min[3, :] = 1022
        ramd_min[4, :] = 840

        return ramd_max, ramd_min

if __name__ == "__main__":
    test = TSCalib(LOADorCALC="LOAD")
    test.main()
    test.get_cross()

    print("Successfully Run the program !!")
