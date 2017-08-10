import numpy as np
import matplotlib.pyplot as plt


class RSCalib:
    def __init__(self):
        """較正のための初期値設定
        """
        # 各諸元設定
        self.ntct = 100  # 温度計算の分割数
        self.nrat = 1000  # ??の分割数
        self.ll = 5  # フィルタ数
        self.maxch = 25  # 空間チャンネル数
        self.maxword = 1000    #ラマン散乱光信号の取得タイミング数
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
        self.PONUM = 2   #YAGレーザーのパワーを計測しているポリクロ番号
        self.CH = 6  #YAGレーザーのパワーを計測しているポリクロのチャンネル番号
        self.PATH = '/Users/kemmochi/SkyDrive/Document/Study/Thomson/Data Analysis/Raman/2015/'
        self.FILE_NAME = 'w_for_alignment_Aug2016.txt'
        self.N2Pressure = np.array([0, 50, 89, 191, 295, 383, 386])     #Raman散乱光計測時の窒素圧力{Torr]

    def calib_Raman(self):
        """
        ラマン散乱光の計測データからトムソン散乱の密度較正に用いる較正係数を計算する
        信号量　vs 窒素ガス圧 の傾きを計算している
        :return:
        """
        rs_mean = np.zeros((self.maxch, len(self.N2Pressure), self.nlaser))
        rs_data, power = self.load_raman()
        calibfact = np.zeros((self.maxch, self.nlaser))
        for ils in range(self.nlaser):
            rs_mean[:, :, ils] = np.mean(rs_data[:, :, :, ils], axis = 0)
        power_mean = np.mean(power, axis = 0)
        pcof = np.mean(power_mean, axis = 0)
        rs_per_power = rs_mean/power_mean
        A = np.vstack([self.N2Pressure, np.ones(len(self.N2Pressure))]).T
        for ils in range(self.nlaser):
            for ich in range(self.maxch):
                calibfact[ich, ils], buf = np.linalg.lstsq(A, rs_per_power[ich, :, ils])[0]

        #plt.plot(rs_data[:, :, 5, 1])
        #plt.plot(power[:, :, 0])
        #plt.plot(rs_mean[:, :, 0])
        #plt.plot(power_mean[:, :])
        #plt.plot(rs_per_power[:, :, 1])
        #plt.plot(calibfact)
        #plt.show()
        return calibfact, pcof

    def load_raman(self):
        """ラマン散乱データの読み出し
        """
        #rs_data = np.zeros((self.nlaser, len(self.N2Pressure), self.maxch, self.maxword))
        #power = np.zeros((self.nlaser, len(self.N2Pressure), self.maxword))
        rs_data = np.zeros((self.maxword, self.maxch, len(self.N2Pressure), self.nlaser))
        power = np.zeros((self.maxword, len(self.N2Pressure), self.nlaser))

        for ils in range(self.nlaser):
            for igp in range(len(self.N2Pressure)):
                file_name = "L%d_%dTorr_2.txt" % (ils+1, self.N2Pressure[igp])
                rwdata = np.loadtxt(self.PATH + file_name, comments='#')
                st_rwdata = self.sort_rawdata(rwdata)
                #clbdata = st_rwdata[:, :self.maxdata-10].reshape((self.maxword, self.nfil+1, self.maxch))   #(Adata[:, :150], (3, 6, 25))
                clbdata = st_rwdata[:, :self.maxdata-10].reshape((self.maxword, self.maxch, self.nfil+1))   #(Adata[:, :150], (3, 6, 25))
                rs_data[:, :, igp, ils] = clbdata[:, :, 0]
                power[:, igp, ils] = clbdata[:, self.PONUM - 1, self.CH - 1]

        return rs_data, power

    def sort_rawdata(self, rawdata):
        """
        散乱光の生データを計測位置・ポリクロチャンネルごとに並び替えて配列に格納する
        :param rawdata:
        :return:
        """
        st_rwdata = np.array(rawdata)
        for i in range(self.maxdata):
            st_rwdata[:, i] = rawdata[:, self.worder[i]]

        return st_rwdata

if __name__ == "__main__":
    RSC = RSCalib()
    RSC.calib_Raman()
