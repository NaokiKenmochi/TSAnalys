import numpy as np
import matplotlib.pyplot as plt
import time

# 定数の設定
calib_settings = {
    'ntct': 100,  # 温度計算の分割数
    'nrat': 100,  # ??の分割数
    'll': 5,  # フィルタ数
    'maxch': 25,  # 空間チャンネル数
    'maxword': 440, #モノクロの較正波長数
    'nfil': 5,  # フィルタ数
    'maxm': 10,
    'maxlaser': 2,  # 最大レーザー台数
    'maxfl': 10,
    'nlaser': 2,  # 使用レーザー数
    'm': 2,
    'num_bg': 3,    #散乱光１タイミングあたりの背景光の取り込み数
    'num_sig': 30,  #散乱光の取り込み数
    'TT': 297.15,  # 較正時のガス温度
    'maxdata': 160,  # 最大取り込みチャンネル数
    'inj_angle': np.pi*8/ 9,  # 入射角度[rad]
    'init_wlength': 685,  # モノクロメータの初期波長[nm]
    'worder': np.array([  # V792のデータ順を並び替える配列
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
        29, 31, 61, 63, 93, 95, 125, 127, 157, 159]),
    'int_range': np.array([  # 各チャンネルの積分範囲、フィルターの透過波長領域に対応
        360, 380,   #ch1
        250, 350,   #ch2
        10, 170,    #ch3
        330, 370,   #ch4
        140, 290,   #ch5
        10, 400])   #ch6
}

class ThAnalysTeNe():
    """
    PolychromatorタイプのYAG-TS計測データを解析するクラス
    電子温度・誤差を計算する
    較正データはTSCalibクラスで計算する
    計測日時(yyyymmdd)，ショット番号，較正係数を計算するかロードするか指定する
    """
    def __init__(self, date, shotNo, CALIBorLOAD, **kwargs):
        self.date = date    #計測日時 yyyymmdd形式
        self.shotNo = shotNo    #ショット番号
        self.CALIBorLOAD = CALIBorLOAD  #較正係数を新たに計算するかロードするか
        #self.PATH = '/Users/kemmochi/SkyDrive/Document/Study/Thomson/DATE/Scattered Light/'
        self.PATH = ''

        """較正のための初期値設定
        """

        # 各諸元設定
        self.ntct = kwargs['ntct']
        self.nrat = kwargs['nrat']
        self.ll = kwargs['ll']
        self.maxch = kwargs['maxch']
        self.maxword = kwargs['maxword']
        self.nfil = kwargs['nfil']
        self.maxm = kwargs['maxm']
        self.maxlaser = kwargs['maxlaser']
        self.maxfl = kwargs['maxfl']
        self.nlaser = kwargs['nlaser']
        self.m = kwargs['m']
        self.num_bg = kwargs['num_bg']
        self.num_sig = kwargs['num_sig']
        self.tt = kwargs['TT']
        self.maxdata = kwargs['maxdata']
        self.inj_angle = kwargs['inj_angle']
        self.worder = kwargs['worder']
        self.int_range = kwargs['int_range']
        self.init_wlength = kwargs['init_wlength']
        #self.i1 = np.array([0, 0, 0, 0, 0, 3, 3, 3, 3, 1, 1, 1, 4, 4, 2])
        #self.i2 = np.array([3, 1, 4, 2, 5, 1, 4, 2, 5, 4, 2, 5, 2, 5, 5])
        self.i1 = np.array([0, 0, 0, 0, 3, 3, 3, 1, 1, 4])
        self.i2 = np.array([3, 1, 4, 2, 1, 4, 2, 4, 2, 2])
        self.tmd = 1.0e-3
        self.tmd2 = 1.0e-40
        self.tmds = 1.0e-20     #密度計測の閾値
        self.num_comb = int(self.ll*(self.ll-1)/2)
        self.num_ratio = int((self.nfil - 1) * self.nfil / 2)  # チャンネルの信号比の組み合わせ数
        self.temax = 10000  #[eV]
        self.temin = 10 #[eV]
        self.te = np.exp(np.log(self.temax) * np.arange(self.ntct)/(self.ntct-1))  # 計算温度範囲[eV] ntctと同数
        #        self.nte = self.cal_Te(self.nrat)   # 計算温度範囲[eV] nratと同数
        self.nte = np.exp(np.log(self.temax) * np.arange(self.nrat)/(self.nrat-1))      # 計算温度範囲[eV] nratと同数
        self.PONUM = 2   #YAGレーザーのパワーを計測しているポリクロ番号
        self.CH = 6  #YAGレーザーのパワーを計測しているポリクロのチャンネル番号
        self.THRE = 1.0
        self.THRE_1 = [1500, 2000]
        self.THRE_2 = [1500, 2000]
        self.COF = [1, 1]
        self.KARI_POWER = [1078, 1502]
        self.nulaser = [1, 0]   #２台のレーザーの発振順序
        self.nec = 1.0



    def main(self):
        """
        電子温度，温度誤差を返す
        グラフ化はmultiplot(self)を実効
        :return:
        """
        if(self.CALIBorLOAD == 'CALIB'):
            from TSCalib import TSCalib
            TSC = TSCalib("LOAD")
            coft, cof, relte, cofne, ecofne = TSC.main()
        else:
            coft_cof_relte_cofne_ecofne = np.load("coft_cof_relte_cofne_ecofne.npz")
            coft = coft_cof_relte_cofne_ecofne["coft"]
            cof = coft_cof_relte_cofne_ecofne["cof"]
            relte = coft_cof_relte_cofne_ecofne["relte"]
            cofne = coft_cof_relte_cofne_ecofne["cofne"]
            ecofne = coft_cof_relte_cofne_ecofne["ecofne"]

#        cofne = np.zeros((self.nrat, self.nfil, self.maxch, self.nlaser))
#        ecofne = np.zeros((self.nrat, self.nfil, self.maxch, self.nlaser))
        wobg, bg = self.remove_bg()
        wobg = self.remove_str_light(wobg, 0, 11)
        bg = self.remove_str_light(bg, 0, 11)
        err = self.cal_err(wobg, bg, 1)
        ratio = self.make_ratio_wobG(wobg, self.tmd2)
        rerr = self.cal_rerror(err, ratio)
        te, teerr = self.teanalys(cof, coft, relte, ratio, rerr, ierr=1)
        Adata, AEdata = self.make_RAdata(wobg, err)
        ne, neerr = self.neanalys(te, teerr, cofne, ecofne, Adata, AEdata, ierr=1)

        return te, teerr, ne, neerr

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


    def make_ratio_wobG(self, wobg, tmd2):
        """
        背景光を取り除いた散乱光データに対し，ポリクロチャンネル毎の比をとる
        np.whereを使えば高速化可能
        :param wobg:
        :param tmd2:
        :return:
        """
        ratio = np.zeros((self.num_sig, self.maxch, self.num_comb))
        for i in range(self.maxch):
            for j in range(self.num_comb):
#                ratio[:, i, j] = self.divide_waves(wobg, tmd2, self.num_comb*i+j, (self.ll+1)*i+self.i1[j], (self.ll+1)*i+self.i2[j])
                i2 = (self.ll+1)*i+self.i1[j]
                i3 = (self.ll+1)*i+self.i2[j]
                for k in range(self.num_sig):
                   if wobg[k, i3] > tmd2:
                       ratio[k, i, j] = wobg[k, i2]/wobg[k, i3]
                   else:
                       ratio[k, i, j] = 0.0

        return ratio

    def make_RAdata(self, wobg, err):
        #power = np.zeros((self.num_sig/self.maxlaser, self.maxlaser))
        #for i in range(self.maxlaser):
        #    power[:,i] = wobg[i::self.maxlaser, (self.ll+1)*(PONUM-1)+CH-1]
        power = wobg[:, (self.ll+1)*(self.PONUM-1)+self.CH-1]

        for j in range(self.maxlaser):
            np.where(power[j::self.maxlaser] < self.THRE_1[j], self.KARI_POWER[j], power[j::self.maxlaser])
            np.where(power[j::self.maxlaser] > self.THRE_2[j], self.COF[j]*power[j::self.maxlaser], power[j::self.maxlaser])
            #np.where(power[:, j] < THRE_1[j], KARI_POWER[j], power[:, j])
            #np.where(power[:, j] > THRE_2[j], COF[j]*power[:, j], power[:, j])

        Adata = wobg
        AEdata = np.abs(err*wobg)
        np.where(Adata < self.THRE, 0, Adata)
        np.where(AEdata < self.THRE, 0, AEdata)
        Adata = (Adata.T/power).T
        AEdata = (AEdata.T/power).T
        #plt.plot(power[:])
        #plt.plot(power[:,1])
        #plt.show()

        Adata = Adata[:, :self.maxdata-10].reshape((self.num_sig, self.maxch, self.nfil+1))   #(Adata[:, :150], (3, 6, 25))
        AEdata = Adata[:, :self.maxdata-10].reshape((self.num_sig, self.maxch, self.nfil+1))   #(Adata[:, :150], (3, 6, 25))
        Adata = Adata[:, :, :self.nfil]
        AEdata = AEdata[:, :, :self.nfil]
        Adata = np.transpose(Adata, (0, 2, 1))
        AEdata = np.transpose(AEdata, (0, 2, 1))

        return Adata, AEdata


    def cal_err(self, wobg, bg, const):
        return np.sqrt((const**2*(np.abs(wobg) + np.abs(bg) + bg**2)))/wobg

    def cal_rerror(self, err, ratio):
        rerr = np.zeros((self.num_sig, self.maxch, self.num_comb))
        for i in range(self.maxch):
            for j in range(int(self.ll*(self.ll-1)/2)):
#                rerr[:, i, j] = self.cal_RErr(err, ratio, self.num_comb*i+j, (self.ll+1)*i+self.i1[j], (self.ll+1)*i+self.i2[j])
                rerr[:, i, j] = np.sqrt(err[:, (self.ll+1)*i+self.i1[j]]**2 + err[:, (self.ll+1)*i+self.i2[j]]**2) * ratio[:, i, j]

        return rerr


    def remove_bg(self):
        """
        散乱光の後に取得しているバックグラウンド信号を差し引く
        :return:
        """
        bg = np.zeros((self.num_sig, self.maxdata))
        wobg = np.zeros((self.num_sig, self.maxdata))
        err = np.zeros((self.num_sig, self.maxdata))
        rate = np.zeros((self.num_sig, 375))
        RErr = np.zeros((self.num_sig, 375))
#        rwdata = np.loadtxt(self.PATH + "%d/%d/Th_Raw_HJ%d.txt" % (int(self.date/10000), self.date-20000000, self.shotNo), comments='#')
        rwdata = np.loadtxt(self.PATH + "Th_Raw_HJ%d.txt" % (self.shotNo), comments='#')
        st_rwdata = self.sort_rawdata(rwdata)
        for i in range(self.num_sig):
            #bg[i] = (st_rwdata[4 * i + 1] + st_rwdata[4 * i + 2] + st_rwdata[4 * i + 3]) / 3
            #wobg[i] = st_rwdata[4 * i] - bg[i]
            for j in range(self.num_bg):
                bg[i] += st_rwdata[(self.num_bg+1)*i+j+1]
        bg /= self.num_bg
        wobg = st_rwdata[::self.num_bg+1] - bg

        return wobg, bg


    def remove_str_light(self, wave, is1, is2):
        """
        生信号からプラズマがない時間の平均値を差し引く
        :param wave:
        :param is1:
        :param is2:
        :return:
        """
        wbuf = np.mean(wave[:, is1:is2], axis=1)
        wave -= wbuf.reshape((self.num_sig,1))

        return wave

    def teanalys(self, cof, coft, relte, ratio, rerr, ierr):
        """
        電子温度・誤差を計算する
        :param ratio:
        :param rerr:
        :param ierr:
        :return:
        """
        cte = np.zeros((self.num_sig, self.num_comb))
        ccte = np.zeros((self.num_sig, self.num_comb))
        tel = np.zeros((self.num_sig, self.num_comb, self.maxch))
        teerrl = np.zeros((self.num_sig, self.num_comb, self.maxch))
        te = np.zeros((self.num_sig, self.maxch))
        teerr = np.zeros((self.num_sig, self.maxch))


        for i in range(self.maxch):
            for j in range(self.num_comb):
                for k in range(self.num_sig):
                    buf1 = ratio[k, i, j]
                    if(ratio[k, i, j] == 0.0):
                        cte[k, j] = 0.0
                        ccte[k, j] = 0.0
                    else:
#                        nbuf = self.getNearestValue(relte[self.num_ratio*i+j, :], ratio[k, self.num_comb*i+j])
                        nbuf = self.getNearestValue(relte[i, j, :], ratio[k, i, j])
                        #nbuf -= 5
                        if(nbuf > self.nrat - 2 or nbuf < 2):
                            cte[k, j] = 0.0
                            ccte[k, j] = 0.0
                        else:
                            cte[k, j] = coft[i, j, nbuf]
                            ccte[k, j] = cof[i, j, nbuf]

            for k in range(self.num_sig):
                for j in range(self.num_comb):
                    if(np.abs(cte[k, j]) > self.tmd):
                        tel[k, j, i] = cte[k,j]/ccte[k,j]
                    else:
                        tel[k, j, i] = 0.0

                    if(np.abs(rerr[k, i, j]*ccte[k, j]) > self.tmd):
                        teerrl[k, j, i] = 1.0/((ccte[k, j]*np.abs(rerr[k, i, j])))
                    else:
                        teerrl[k, j, i] = 0.0

            for k in range(self.num_sig):
                buf = 0.0
                for j in range(self.num_comb):
                    if(np.abs(rerr[k, i, j]) > self.tmd):
                        buf += ccte[k, j]/(rerr[k, i,  j]**2)

                if(buf == 0.0):
                    teerr[k, i] = 0.0
                else:
                    teerr[k, i] = 1.0/buf

            for k in range(self.num_sig):
                buf = 0.0
                for j in range(self.num_comb):
                    if(np.abs(rerr[k, i, j]) > self.tmd):
                        buf += cte[k, j]/(rerr[k, i, j]**2)

                te[k, i] = buf*teerr[k, i]

            if(ierr == 1):
                teerr[:, i] = np.sqrt(np.abs(teerr[:, i]))  #ブロードキャストにできる
            else:
                for k in range(self.num_sig):
                    buf = 0.0
                    for j in range(self.num_comb):
                        if(np.abs(rerr[k, i, j]*ccte[k, j]) > self.tmd):
                            buf += (ccte[k,j]*te[k,i]-cte[k,j])/ccte[k,j]/(rerr[k, i, j]**2)

                    if(ierr == 2):
                        teerr[k, i] = np.sqrt(np.abs(teerr[k, i]*buf))
                    elif(ierr == 3):
                        buf += 1.0
                        teerr[k, i] = np.sqrt(np.abs(teerr[k, i]*buf))

        return te, teerr

    def neanalys(self, te, teerr, cofne, ecofne, Adata, AEdata, ierr):
        """
        Unit of ne: 10^19 m^{-3}
        :param te:
        :param teerr:
        :param cofne:
        :param ecofne:
        :param Adata:
        :param AEdata:
        :param ierr:
        :return:
        """
        cne = np.zeros((self.num_sig, self.nfil, self.maxch))
        ecne = np.zeros((self.num_sig, self.nfil, self.maxch))
        nef = np.zeros((self.num_sig, self.nfil, self.maxch))
        neerrf = np.zeros((self.num_sig, self.nfil, self.maxch))
        neerr2 = np.zeros((self.num_sig, self.nfil))
        ne = np.zeros((self.num_sig, self.maxch))
        neerr = np.zeros((self.num_sig, self.maxch))
        """
        np.whereを使えば高速化できると思う
        """
        for ich in range(self.maxch):
            ilnum = 0
            for idat in range(self.num_sig):
                nbuf = self.getNearestValue(self.nte, te[idat, ich])
                if(nbuf > self.nrat or nbuf < 2):
                    cne[idat, :, ich] = 0.0
                    ecne[idat, :, ich] = 0.0
                else:
                    ilaser = self.nulaser[ilnum]
                    if(ilnum > self.nlaser):
                        cne[idat, :, ich] = 0.0
                        ecne[idat, :, ich] = 0.0
                    else:
                        cne[idat, :, ich] = cofne[nbuf, :, ich, ilaser]
                        ecne[idat, :, ich] = ecofne[nbuf, :, ich, ilaser]
                ilnum += 1
                if(ilnum > self.nlaser-1):
                    ilnum -= self.nlaser

            nef = cne*Adata
            neerrf[:, :, ich] = np.abs(cne[:, :, ich]*AEdata[:, :, ich]) + np.abs(ecne[:, :, ich].T*teerr[:, ich]*Adata[:, :, ich].T).T
            neerr2[:, :] = neerrf[:, :, ich]**2

            for idat in range(self.num_sig):
                buf = 0.0
                for ifil in range(self.nfil):
                    if(neerr2[idat, ifil] > self.tmds):
                        buf += 1.0/neerr2[idat, ifil]
                if(np.abs(buf) < self.tmds):
                    neerr[idat, ich] = 0.0
                else:
                    neerr[idat, ich] = 1.0/buf

            for idat in range(self.num_sig):
                buf = 0.0
                for ifil in range(self.nfil):
                    if(neerr2[idat, ifil] > self.tmds):
                        buf += nef[idat, ifil, ich]/neerr2[idat, ifil]
                ne[idat, ich] = buf*neerr[idat, ich]

            if(ierr == 1):
                neerr[:, ich] = np.sqrt(np.abs(neerr[:, ich]))
            else:
                for idat in range(self.num_sig):
                    buf = 0.0
                    for ifil in range(self.nfil):
                        if(neerr2[idat, ifil] > self.tmds):
                            buf += (ne[idat, ich] - nef[idat, ifil, ich])**2/neerr2[idat, ifil]
                    if(ierr == 2):
                        neerr[idat, ich] = np.sqrt(np.abs(buf*neerr[idat, ich]))
                    elif(ierr == 3):
                        buf += 1
                        neerr[idat, ich] = np.sqrt(np.abs(buf*neerr[idat, ich]))

            ne[:, ich] = self.nec * ne[:, ich]
            neerr[:, ich] = self.nec * neerr[:, ich]

        return ne, neerr


    def getNearestValue(self, list, num):
        """
        概要: リストからある値に最も近い値を返却する関数
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

    def multiplot(self):
        """
        解析データをタイミング毎にならべてプロットする
        とりあえず温度の解析データのみ
        :return:
        """
        nRows = 5
        nCols = 6
        fig = plt.figure(figsize=(18, 10))
        te, teerr, ne, neerr = self.main()
        te /= 1000
        teerr /= 1000
        majorR = np.loadtxt("MajorR_2015.txt")
        majorR /= 1000

        ax = fig.add_subplot(nRows, nCols, 1)
        fig.subplots_adjust(hspace=0, wspace=0)

#        for j in range(self.num_sig):
#            plt.subplot(nRows, nCols, j+1, sharex=ax, sharey=ax)
        # Turn off tick labels where needed.
        index = 0
        for r in range(1, nRows+1):
            for c in range(1, nCols+1):
                index += 1
                # Turn off y tick labels for all but the first column.
                if((c == 1) and (index <= self.num_sig)):
                    ax1 = plt.subplot(nRows, nCols, index, sharex=ax, sharey=ax)
                    ax1.legend(fontsize=8)
                    ax1.set_ylabel("Te [keV]")
#                if((c == nCols) and (index <= self.num_sig)):
#                    ax2 = ax1.twinx()
#                    ax2 = plt.subplot(nRows, nCols, index, sharex=ax, sharey=ax)
#                    ax2.legend(fontsize=8)
#                    ax2.set_ylabel("ne [x10({19} m^{-3}]")
                if((r == nRows) and (index <= self.num_sig)):
                    ax1 = plt.subplot(nRows, nCols, index, sharex=ax, sharey=ax)
                    ax1.legend(fontsize=8)
#                    ticks = ax1.set_xticks([1.0, 1.1, 1.2, 1.3, 1.4])
                    #ax1.set_xticklabels(majorR, rotation = 45)
                    #ax1.set_xticklabels([1.1, 1.2, 1.3])
                    ax1.set_xlabel("Major R [m]")
                if((c != 1) and (index <= self.num_sig)):
                    ax1 = plt.subplot(nRows, nCols, index, sharex=ax, sharey=ax)
                    ax1.legend(fontsize=8)
                    plt.setp(ax1.get_yticklabels(), visible=False)
                if((c != nCols + 1) and (index <= self.num_sig)):
                    ax2 = ax1.twinx()
                    ax2 = plt.subplot(nRows, nCols, index, sharex=ax, sharey=ax)
                    ax2.legend(fontsize=8)
#                    ax2.set_yticklabels(())
                    plt.setp(ax2.get_yticklabels(), visible=False)
                # Turn off x tick lables for all but the bottom plot in each column.
                if((self.num_sig - index) >= nCols):
                    ax1 = plt.subplot(nRows, nCols, index, sharex=ax, sharey=ax)
                    ax1.legend(fontsize=8)
                    plt.setp(ax1.get_xticklabels(), visible=False)
        for j in range(self.num_sig):
            if(j == nCols):
                plt.title("Date: %s, Shot No.: %d" % (self.date, self.shotNo), loc='right', fontsize=36, fontname="Times New Roman")
            ax1 = plt.subplot(nRows, nCols, j+1, sharex=ax, sharey=ax)
            ax2 = ax1.twinx()
            #ax1.plot(majorR, te[j, :], marker='o', ls='None', label="%d ms" % (40 + 10*j))
            ax1.errorbar(majorR, te[j, :], fmt='o', ls='None', yerr=teerr[j, :], label="%d ms" % (40 + 10*j))
            ax2.errorbar(majorR, ne[j, :], fmt='o', ls='None', yerr=neerr[j, :], color="r")
            plt.setp(ax2.get_yticklabels(), visible=False)
            ax1.legend(fontsize=8)
            ax1.set_ylim(0, 4)
            ax2.set_ylim(0, 2e-5)
            ax1.set_xlim(1.02, 1.35)

        plt.show()

    def multiplot_sepTeNe(self):
        """
        解析データをタイミング毎にならべてプロットする
        とりあえず温度の解析データのみ
        :return:
        """
        nRows = 10
        nCols = 6
        fig = plt.figure(figsize=(18, 10))
        te, teerr, ne, neerr = self.main()
        te /= 1000
        teerr /= 1000
        majorR = np.loadtxt("MajorR_2015.txt")
        majorR /= 1000

        ax = fig.add_subplot(nRows, nCols, 1)
        fig.subplots_adjust(hspace=0, wspace=0)

        #        for j in range(self.num_sig):
        #            plt.subplot(nRows, nCols, j+1, sharex=ax, sharey=ax)
        # Turn off tick labels where needed.
        index = 0
        i = 0
        j = 0
        for r in range(1, nRows+1):
            for c in range(1, nCols+1):
                index += 1
                if(index == nCols + 1):
                    plt.title("Date: %s, Shot No.: %d" % (self.date, self.shotNo), loc='right', fontsize=36, fontname="Times New Roman")
                if r%2 == 0:
                    ax1 = plt.subplot(nRows, nCols, index, sharex=ax)#, sharey=ax1)
                    ax1.errorbar(majorR, te[i, :], fmt='o', ls='None', yerr=teerr[i, :], label="%d ms" % (40 + 10*i))
                    #ax1.plot(majorR, te[j, :], marker='o', ls='None', label="%d ms" % (40 + 10*j))
                    ax1.set_ylim(0, 0.1)
                    i += 1
                    if((c == 1) and (index <= 2*self.num_sig)):
                        ax1 = plt.subplot(nRows, nCols, index, sharex=ax)#, sharey=ax)
                        ax1.legend(fontsize=8)
                        ax1.set_ylabel("Te [keV]")
                    if((r == nRows) and (index <= 2*self.num_sig)):
                        ax1 = plt.subplot(nRows, nCols, index, sharex=ax)#, sharey=ax)
                        ax1.legend(fontsize=8)
                        ax1.set_xlabel("Major R [m]")
                    if((c != 1) and (index <= 2*self.num_sig)):
                        ax1 = plt.subplot(nRows, nCols, index, sharex=ax)#, sharey=ax)
                        ax1.legend(fontsize=8)
                        plt.setp(ax1.get_yticklabels(), visible=False)
                # Turn off x tick lables for all but the bottom plot in each column.
                    if((2*self.num_sig - index) >= nCols):
                        ax1 = plt.subplot(nRows, nCols, index, sharex=ax)#, sharey=ax)
                        ax1.legend(fontsize=8)
                        plt.setp(ax1.get_xticklabels(), visible=False)
                else:
                    ax2 = plt.subplot(nRows, nCols, index, sharex=ax)#, sharey=ax)
                    ax2.errorbar(majorR, ne[j, :], fmt='o', ls='None', yerr=neerr[j, :], color="r", label="%d ms" % (40 + 10*j))
                    ax2.set_ylim(0, 2e-6)
                    j += 1
                # Turn off y tick labels for all but the first column.
                    if((c == 1) and (index <= 2*self.num_sig)):
                        ax2 = plt.subplot(nRows, nCols, index, sharex=ax)#, sharey=ax)
                        ax2.legend(fontsize=8)
                        ax2.set_ylabel("ne []")
                    if((c != 1) and (index <= 2*self.num_sig)):
                        ax2 = plt.subplot(nRows, nCols, index, sharex=ax)#, sharey=ax)
                        ax2.legend(fontsize=8)
                        plt.setp(ax2.get_yticklabels(), visible=False)
                    if((2*self.num_sig - index) >= nCols):
                        ax2 = plt.subplot(nRows, nCols, index, sharex=ax)#, sharey=ax)
                        ax2.legend(fontsize=8)
                        plt.setp(ax2.get_xticklabels(), visible=False)
        ax1.set_xlim(1.02, 1.35)

        filename = "TeNe_%s_%d.pdf" % (self.date, self.shotNo)
        plt.savefig(filename)
        print("Save graph to " + filename)

#        plt.show()

if __name__ == "__main__":
    start = time.time()
    TA = ThAnalysTeNe(date=20161117, shotNo=64404, CALIBorLOAD="CALIB", **calib_settings)
    TA.main()
    TA.multiplot_sepTeNe()
    elapsed_time = time.time() - start
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")