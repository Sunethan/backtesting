import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib.pylab as pylab
import physics as phys
from scipy.optimize import curve_fit
import utils
import GenerationRate.ImpactIonization as II
import os
import glob
import time
import csv
from tempfile import NamedTemporaryFile
import shutil
import plotly.graph_objects as go
start_time = time.time()
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (7.5, 7),
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 'x-large',
          'ytick.labelsize': 'x-large'}
pylab.rcParams.update(params)
plt.rcParams.update({'font.size': 13})
import warnings
warnings.simplefilter("error")


class TDDB(object):
    def __init__(self, location):
        self._curve = utils.parse_csv(location, 2, 0, 0, 'Yes')
        self._IT = [Read(location, 'Standard', xlabel=cur, ylabel='same') for cur in self._curve]

    def statistics(self, criteria):
        tbd_arr = []
        for curve in self._IT:
            tbd_arr.append(utils.find_bd(curve.X, abs(curve.Y), criteria))
        avg = np.average(tbd_arr)
        std = np.std(tbd_arr)
        return avg, std


class WeibullPlot(object):
    def __init__(self, location, criteria):
        self._loc = location
        self._cri = criteria
        self._curve = utils.parse_csv(self._loc, 2, 0, 0, 'Yes')
        self._IV = [Read(self._loc, xlabel=curve, ylabel='same') for curve in self._curve]
        self._TimeSet = []

        for iv in self._IV:
            for j, current in enumerate(iv.y):
                if current >= self._cri:
                    t1 = np.log10(iv.y[j - 1])
                    t2 = np.log10(iv.x[j])
                    i1 = np.log10(iv.y[j - 1])
                    i2 = np.log10(iv.y[j])
                    slope = (i2 - i1) / (t2 - t1)
                    tc = t1 + (np.log10(self._cri) - i1) / slope
                    self._TimeSet.append(10 ** tc)
                    break
        self._TimeSet.sort()
        self._F = np.array([i / len(self._TimeSet) for i, data in enumerate(self._TimeSet)])

    def get_Time(self):
        return np.array(self._TimeSet[1:])

    def get_F(self):
        return self._F[1:]

    def fit(self):
        def f(x, a, b):  # this is your 'straight line' y=f(x)
            return a * x + b

        popt, pcov = curve_fit(f, np.log10(np.array(self._TimeSet[1:])), np.log(-np.log(1 - self._F[1:])))  # your data x, y to fit
        return [popt, pcov]


class ReadLog(object):
    def __init__(self, location=None, title='0', row=0, column=0, name=0, initial_value=None, delimiter=" ",
                 form=None, time_marker='Computing BE-step from', data_form='number', start_line=None):
        """

        :param location:
        :param title:
        :param row:
        :param column:
        :param name:
        :param initial_value:
        :param delimiter: if data_form is 'number', then delimiter is required.
        :param form:
        :param time_marker:
        :param data_form: 'number' or 'coordinate'.
        :param start_line: it is a string.
        """
        if type(title) is str:
            title = [title]
        if type(row) is int:
            row = [row]
        if type(column) is int:
            column = [column]
        if type(name) is str:
            name = [name]
        if not all(isinstance(x, list) for x in [title, row, column, name]):
            raise TypeError("Title, row, column and name should be the list.")

        if len(title) not in [1, len(row)]:
            raise ValueError("Title number should be 1 or number of rows.")
        if len(row) != len(column):
            if data_form == 'coordinate':
                pass
            else:
                raise ValueError("The number of rows and columns should be equal.")
        if len(row) != len(name):
            if data_form == 'coordinate':
                pass
            else:
                raise ValueError("The number of rows and names should be equal.")

        def time_format(line):
            if time_marker == 'Computing BE-step from':
                return float(line.split(delimiter)[6])
            elif time_marker == 'Computing step from':
                return float(line.split(delimiter)[5].split("=")[1])
            elif time_marker == 'Computing BDF2-step from':
                return float(line.split(delimiter)[6])
            elif time_marker == 'Computing TR-step from':
                return float(line.split(delimiter)[6])
            elif time_marker == 'Total time':
                return float(line.split(delimiter)[-2])
            elif time_marker == 'Solve time':
                return float(line.split(delimiter)[-2])
            else:
                raise ValueError("No such time_marker: %s" % time_marker)

        def data_format(line, j):
            def tranform(data, j):
                if form is None:
                    return float(data)
                else:
                    tmp = form[j]
                    if isinstance(form, list):
                        if tmp[0] == 'split':
                            return data.split(tmp[1])[tmp[2]]
                        else:
                            raise ValueError("Wrong form: %s" % tmp)
                    else:
                        raise ValueError("Wrong form: %s" % tmp)

            if data_form == 'number':
                return tranform(line.split(delimiter)[column[j]], j)
            elif data_form == 'coordinate':
                coordinate = line.split("(")[1].split(")")[0].split(", ")
                x = coordinate[0]
                y = coordinate[1]
                z = coordinate[2]
                return [x, y]

        self.__datasets = {n: [] for n in ['time'] + name}
        self.__file = open(location, "r")

        start_marker = 0
        lines = self.__file.readlines()
        for i, line in enumerate(lines):
            if start_marker == 0:
                if start_line is not None:
                    if start_line in line:
                        start_marker = 1
            elif start_marker == 1:
                if time_marker in line:
                    ## search_text = lines[i].split(delimiter)[6]
                    self.__datasets['time'].append(time_format(lines[i]))
                for j, data_head in enumerate(title):
                    empty_data = len(self.__datasets['time']) - len(self.__datasets[name[j]])
                    if empty_data == 1:
                        if data_head in line:
                            # search_data = lines[i + row[j]].split(delimiter)[column[j]]
                            # self.__datasets[name[j]].append(tranform(search_data, j))
                            self.__datasets[name[j]].append(data_format(lines[i + row[j]], j))
                    elif empty_data == 2:
                        if len(self.__datasets[name[j]]) > 0:
                            self.__datasets[name[j]].append(self.__datasets[name[j]][-1])
                        else:
                            self.__datasets[name[j]].append(initial_value[j])
                    elif empty_data == 0:
                        pass
                    else:
                        raise ValueError("Check codes! There is a bug at %s" % [i, j, data_head, empty_data,
                                                                                len(self.__datasets[name[0]]),
                                                                                len(self.__datasets[name[j]])])
                if i == len(lines) - 1:
                    for j, data_head in enumerate(title):
                        empty_data = len(self.__datasets['time']) - len(self.__datasets[name[j]])
                        if empty_data == 1:
                            self.__datasets[name[j]].append(self.__datasets[name[j]][-1])
                        elif empty_data == 0:
                            pass
                        else:
                            raise ValueError("Check codes! There is a bug at the end: %s, %s" % (empty_data, location))
            else:
                raise ValueError("Start marker should be 0 or 1 (%s)" % start_marker)
        self.__file.close()

    def get(self, data):
        return np.asarray(self.__datasets[data])


class Read(object):
    def __init__(self, data, xlabel=None, ylabel=None, x=None, y=None, floatKey='Yes', show='No',
                 checkoverlap='No', interval=None, new_label=None, in_labels=None, ex_labels=None,
                 xType=None, yType=None):
        """
        :param data:
        :param xlabel: None (default)
        :param ylabel: None (default)
        :param x: None (default)
        :param y: None (default)
        :param floatKey: 'Yes"
        :param show: 'No' (default)
        :param checkoverlap: 'No' (default)
        :param interval: [x1, x2] (cut interval)
        :param new_label: {old1: new1, old2: new2}
        :param in_labels: [l1, l2, ...] Only include labels containing at least one of the in_label.
        :param ex_labels: [l1, l2, ...] Exclude labels containing any ex_label.
        """

        def type_conversion(array, axis):
            if axis == 'x':
                if xType is None:
                    return array
                elif xType == 'float':
                    return np.asarray([float(i) for i in array])
                else:
                    raise BaseException("Please add more xType options: %s" % xType)
            elif axis == 'y':
                if yType is None:
                    return array
                elif yType == 'float':
                    return np.asarray([float(i) for i in array])
                else:
                    raise BaseException("Please add more yType options: %s" % yType)
            else:
                raise BaseException("Please add more axis options: %s" % axis)

        def in_ex_labels(csv_label):
            marker = 1
            if isinstance(in_labels, list):
                for in_l in in_labels:
                    if in_l in csv_label:
                        if isinstance(ex_labels, list):
                            for ex_l in ex_labels:
                                if ex_l in csv_label:
                                    marker *= 0
                            if marker == 1:
                                return True
                            else:
                                return False
                        return True
            else:
                if isinstance(ex_labels, list):
                    for ex_l in ex_labels:
                        if ex_l in csv_label:
                            marker *= 0
                    if marker == 1:
                        return True
                else:
                    return True

        self._labels_bk = None
        self._xy_marker = False
        if data is not None:
            self.__data = data
            if data.split(".")[-1] == 'csv':
                self.__data = data
                self.__XY = utils.parse_csv(data, 1, 0, 0, floatKey, show)
                if xlabel is not None:
                    self._labels = [xlabel]
                    if ylabel == 'same':
                        ylabel = xlabel
                    if type(self.__XY[xlabel]) is dict:
                        self._xy_marker = True
                        self._X = np.asarray(self.__XY[xlabel]['X'])
                        self._Y = np.asarray(self.__XY[ylabel]['Y'])
                    else:
                        self._X = type_conversion(self.__XY[xlabel], 'x')
                        self._Y = type_conversion(self.__XY[ylabel], 'y')
                else:
                    self._labels = utils.parse_csv(data, 2, 0, 0, floatKey, show)
                    self._X = dict()
                    self._Y = dict()
                    for label in self._labels:
                        if in_ex_labels(label):
                            if type(self.__XY[label]) is dict:
                                self._xy_marker = True
                                self._X[label] = np.asarray(self.__XY[label]['X'])
                                self._Y[label] = np.asarray(self.__XY[label]['Y'])
                            else:
                                self._X[label] = self.__XY[label]
                                self._Y[label] = self.__XY[label]
                    if in_labels is not None or ex_labels is not None:
                        self._labels = [label for label in self._X.keys()]
            else:
                raise BaseException("*.csv not found!")
        else:
            self._X = np.asarray(x)
            self._Y = np.asarray(y)

        if checkoverlap == 'Yes':
            if isinstance(self._X, np.ndarray) and isinstance(self._Y, np.ndarray):
                self._X, self._Y = utils.CheckOverlap(self._X, self._Y, show)
            elif isinstance(self._X, dict) and isinstance(self._X, dict):
                for key in self._X.keys():
                    self._X[key], self._Y[key] = utils.CheckOverlap(self._X[key], self._Y[key], show)
            else:
                raise ValueError("Please check the type of self._X(%s) and self._Y(%s)" % (type(self._X), type(self._Y)))

        if interval is not None:
            cut_x = np.array([value for value in self._X if interval[0] <= value <= interval[1]])
            cut_y = np.array([value for j, value in enumerate(self._Y) if interval[0] <= self._X[j] <= interval[1]])
            self._X = cut_x
            self._Y = cut_y

        if new_label is not None and xlabel is None:
            tmp_old = self._X.copy()
            if callable(new_label):
                for old_key in tmp_old.keys():
                    self._X[new_label(old_key)] = self._X.pop(old_key)
                    self._Y[new_label(old_key)] = self._Y.pop(old_key)
                self._labels = [label for label in self._X.keys()]
            else:
                for old_key in tmp_old.keys():
                    for new_key in new_label.keys():
                        if new_key in old_key:
                            self._X[new_label[new_key]] = self._X.pop(old_key)
                            self._Y[new_label[new_key]] = self._Y.pop(old_key)
                self._labels = [label for label in self._X.keys()]

    @property
    def x(self):
        return self._X

    @property
    def y(self):
        return self._Y

    def get_xy(self):
        if self._xy_marker:
            return {'X': self._X, 'Y': self._Y}
        else:
            return self._X

    @property
    def curve(self):
        return self._labels

    def recover_curve(self):
        if self._labels_bk is not None:
            return self._labels_bk
        else:
            return self._labels

    def rename(self, name_list):
        self._labels_bk = self._labels
        if isinstance(self._X, dict) and isinstance(self._Y, dict):
            for i, new in enumerate(name_list):
                self._X[new] = self._X.pop(self._labels[i])
                self._Y[new] = self._Y.pop(self._labels[i])
            self._labels = name_list
        else:
            raise TypeError("No need to rename.")


class CSV(object):
    def __init__(self, data, xlabel, ylabel, x=None, y=None, floatKey='Yes'):
        if data is not None:
            self.__data = data
            self.__XY = utils.parse_csv(data, 1, 0, 0, floatKey)
            if ylabel == 'same':
                ylabel = xlabel
            if type(self.__XY[xlabel]) is dict:
                self.X = np.asarray(self.__XY[xlabel]['X'])
                self.Y = np.asarray(self.__XY[ylabel]['Y'])
            else:
                self.X = np.asarray(self.__XY[xlabel])
                self.Y = np.asarray(self.__XY[ylabel])
        else:
            self.X = np.asarray(x)
            self.Y = np.asarray(y)

    def get_location(self):
        return self.__data

    def set_XY(self, x, y):
        self.X = x
        self.Y = y

    def dydx(self):
        x, y = utils.dydx(self.X, self.Y)
        print('CSV.dydx: Note that the length is %s, rather than the initial length %s' % (len(x), len(self.X)))
        return x, y

    def integrate(self, lowerlimit, upperlimit):
        x = self.X
        y = self.Y
        ans = 0
        if lowerlimit == 'no' and upperlimit != 'no':
            for i in range(len(x) - 1):
                if x[i] < upperlimit:
                    yavg = (y[i + 1] + y[i]) / 2
                    dx = x[i + 1] - x[i]
                    ans += yavg * dx  # ans += y[i] * dx
        elif lowerlimit != 'no' and upperlimit == 'no':
            for i in range(len(x) - 1):
                if lowerlimit <= x[i]:
                    yavg = (y[i + 1] + y[i]) / 2
                    dx = x[i + 1] - x[i]
                    ans += yavg * dx
        elif lowerlimit == 'no' and upperlimit == 'no':
            for i in range(len(x) - 1):
                yavg = (y[i + 1] + y[i]) / 2
                dx = x[i + 1] - x[i]
                ans += yavg * dx
        else:
            for i in range(len(x) - 1):
                if lowerlimit <= x[i] < upperlimit:
                    yavg = (y[i + 1] + y[i]) / 2
                    dx = x[i + 1] - x[i]
                    ans += yavg * dx
        return ans


class ZnProfile(CSV):
    def __init__(self, data, xlabel, ylabel):
        super().__init__(data, xlabel, ylabel)
        self.Xj = None

        warn_count = 0
        for i in range(len(self.X) - 1):
            if i < len(self.X) - 1:
                if warn_count == 0:
                    print('DopingProfile : %s' % data)
                    warn_count = 1
                if self.X[i] == self.X[i + 1]:
                    tempX = self.X[i]
                    tempY = self.Y[i]
                    self.Y[i] = (self.Y[i] + self.Y[i + 1]) / 2
                    self.X = np.delete(self.X, i)
                    self.Y = np.delete(self.Y, i + 1)
                    print('x[%s] = x[%s] = %s  -->  y[%s] is deleted and y[%s]=%.4e becomes y[%s]=%.4e' %
                          (i, i + 1, tempX, i, i, tempY, i, self.Y[i]))
            else:
                break

        self.X_wrt_surface = [element - self.X[0] for element in self.X]

    def find_x(self, concentration):
        """
        find_x 與 utils.find() 的差別在於，前者是由 y 求 x，後者是由 x 求 y。
        :param concentration:
        :return:
        """
        for i in range(len(self.Y) - 1):
            if (self.Y[i] - concentration) * (self.Y[i + 1] - concentration) < 0:
                x_goal = utils.point_interpolation(self.X[i], self.Y[i], self.X[i + 1], self.Y[i + 1], 'y', concentration, 'log')
                x_goal_wrt_surface = x_goal - self.X[0]
                print('Linear interpolation: %.4e  ->' % ((self.X[i] + self.X[i + 1]) / 2 - self.X[0]),
                      '  Log interpolation: %.4e' %
                      (utils.point_interpolation(self.X[i], self.Y[i], self.X[i + 1],
                                                 self.Y[i + 1], 'y', concentration, 'log') - self.X[0]))
                return x_goal_wrt_surface
        raise BaseException("Cannot find %.2e at %s " % (concentration, self.data))


class DopingProfile(CSV):
    def __init__(self, data, xlabel, ylabel, CheckOverlap='No'):
        super().__init__(data, xlabel, ylabel)
        self.Xj = None

        if CheckOverlap == 'Yes':
            warn_count = 0
            for i in range(len(self.X) - 1):
                if i < len(self.X) - 1:
                    if warn_count == 0:
                        print('DopingProfile : %s' % data)
                        warn_count = 1
                    if self.X[i] == self.X[i + 1]:
                        tempX = self.X[i]
                        tempY = self.Y[i]
                        self.Y[i] = (self.Y[i] + self.Y[i + 1]) / 2
                        self.X = np.delete(self.X, i)
                        self.Y = np.delete(self.Y, i + 1)
                        print('There are overlaps.')
                        print('x[%s] = x[%s] = %s -->  y[%s] is deleted and y[%s]=%.4e becomes y[%s]=%.4e' %
                              (i, i + 1, tempX, i, i, tempY, i, self.Y[i]))
                else:
                    break

    def find_xj(self):
        for i, x in enumerate(self.X):
            if self.Y[i] * self.Y[i + 1] < 0:
                x1 = self.X[i]
                y1 = self.Y[i]
                x2 = self.X[i + 1]
                y2 = self.Y[i + 1]
                self.Xj = utils.point_interpolation(x1, y1, x2, y2, 'y', 0, 'linear')
                return self.Xj

    def plot(self, number, color='b', label=None):
        if number != -1:
            figure(num=number, figsize=(20, 9.3), dpi=80, facecolor='w', edgecolor='k')
            plt.figure(number)
            plt.grid(True)
            plt.title('Doping profile')
            plt.xlabel(r'Position ($\mu$m)')
            plt.ylabel(r'Concentration (cm$^{-3}$)')
        if label is None:
            plt.plot(self.X, abs(self.Y), color=color, marker='o')
        else:
            plt.plot(self.X, abs(self.Y), color=color, marker='o', label=label)

    def convert_plx(self, add='No', info=None):
        """
        :param add:
        :param info: [RawZn, xlabel, ylabel, ZnConc]
        其中，
        RawZn 是 Zn diffusion profile 的原始數據
        ZnConc 是擷取 Zn profile 的低限濃度
        ZnThick 是指，從低限濃度開始倒推至表面的厚度，將這段距離內的 profile 取出來。
        :return:
        """
        # 製作 *.plx 檔，以重建 TCAD mesh。
        temp = self.X[0]
        new_x = [self.X[i] - temp for i in range(len(self.X))]
        # 在最後面加上 (20, 6.666e17)，這樣才不會出現濃度為零的情況。
        new_x.append(20)
        new_y = np.append(self.Y, self.Y[-1])


        # 新增 Raw Zn doping profile
        if add != 'No':
            RawZn, xlabel, ylabel, ZnConc = info

            # 1. 設定 ZnThick
            ZnThick = None
            for i, element in enumerate(new_y):
                if abs(element) <= ZnConc:
                    ZnThick = new_x[i]
                    break
            if ZnThick is None:
                raise BaseException("ZnConc(%.2e) not found" % ZnConc)

            # 2. 擷取有效厚度
            Zn = CSV(data=RawZn, xlabel=xlabel, ylabel=ylabel)

            Raw_ZnConc_position = None
            for i, element in enumerate(Zn.Y):
                if element < ZnConc:
                    Raw_ZnConc_position = Zn.X[i]
                    break
            if Raw_ZnConc_position is None:
                raise BaseException("Raw_ZnConc_position not found, ZnConc = %.2e" % ZnConc)

            # 3. 新增修改過後的 doping profile 並擷取數據 Zn.Y > ZnConc 的有效數據
            modified_doping = []
            modified_position = []
            X0 = 0
            for i, element in enumerate(Zn.Y):
                if element >= ZnConc and Zn.X[i] + ZnThick >= Raw_ZnConc_position:
                    if X0 == 0:
                        X0 = Zn.X[i]
                    dx = Raw_ZnConc_position - ZnThick
                    modified_doping.append(-abs(element))
                    modified_position.append(Zn.X[i] - dx)
            if modified_position[0] != 0:
                modified_position.insert(0, 0)
                modified_doping.insert(0, modified_doping[0])

            # 4. 補上 new_x, new_y 在 x >= ZnThick 之後的數據
            for i, element in enumerate(new_y):
                if new_x[i] >= ZnThick:
                    modified_doping.append(element)
                    modified_position.append(new_x[i])

            '''
            plt.plot(new_x, new_y, '-b', label='Before')
            plt.plot(modified_position, modified_doping, '-r', label='After')
            plt.grid(True)
            plt.legend(loc='best')
            plt.show()
            '''
            # 5. 替換 new_x, new_y
            new_x = modified_position
            new_y = modified_doping

        # 儲存檔案
        new_data_format = self.get_location().replace("raw_data", "new_data").replace(".csv", ".plx")
        utils.save(new_data_format, 'X', 'Y', new_x, new_y, form='', title='DopingConcentration')

    def align_xj(self, x1, x2, dopingmin):
        for i in range(len(self.Y)):
            if x1 < self.X[i] < x2:
                if abs(self.Y[i]) < dopingmin:
                    dopingmin = abs(self.Y[i])
                    self.Xj = self.X[i]

        for i in range(len(self.X)):
            self.X[i] -= self.Xj
        return np.asarray(self.X)

    def abrupt(self, junction, Vgoal):
        x = self.X * 1e-4  # [cm]
        doping = self.Y  # [cm-3]
        xj = 0
        j = 0
        xcg, xga, xab = junction
        for i in range(len(x)):
            if doping[i] * doping[i + 1] < 0 and doping[i] < 0:
                xj = x[i + 1]
                j = i + 1
                break

        def eps(xi):
            eps_InP = 12.5 * 8.85e-14  # [F/cm]
            eps_InGaAs = 13.9 * 8.854e-14  # [F/cm]
            eps_InGaAsP = 13.436 * 8.85e-14  # [F/cm] Approximated by In 0.53 Ga 0.47 As 0.65 P 0.35
            if xi < xcg or xi > xab:
                return eps_InP  # [F/cm]
            elif xcg < xi < xga:
                return eps_InGaAsP  # [F/cm]
            else:
                return eps_InGaAs  # [F/cm]

        def voltage(Emax, j, junction, x):  # [V/cm], [-], [um], [cm]
            xcg, xga, xab = np.asarray(junction) * 1e-4  # [cm]
            Vm = 0  # [V]
            Vg = 0  # [V]
            Vabs = 0  # [V]
            Vbs = 0  # [V]
            Vt = 0  # [V]
            q = 1.6e-19  # [C]
            E0 = Emax  # [V/cm]
            E_array = np.zeros(len(x))  # [V/cm]
            while E0 > 0:
                if j + 1 == len(x):
                    raise IndexError("len(x) and len(E) is not enough. E(%.3e cm)=%.3e V/cm" % (x[-1], E0))
                E_array[j] = E0
                temp_E0 = E0
                E0 -= q / eps(x[j]) * doping[j] * (x[j + 1] - x[j])  # [V/cm]
                Vt += (temp_E0 + E0) / 2 * (x[j + 1] - x[j])  # [V]
                if x[j] < xcg:
                    Vm += (temp_E0 + E0) / 2 * (x[j + 1] - x[j])  # [V]
                elif xcg <= x[j] < xga:
                    Vg += (temp_E0 + E0) / 2 * (x[j + 1] - x[j])  # [V]
                elif xga <= x[j] < xab:
                    Vabs += (temp_E0 + E0) / 2 * (x[j + 1] - x[j])  # [V]
                elif xab <= x[j] :
                    Vbs += (temp_E0 + E0) / 2 * (x[j + 1] - x[j])  # [V]
                j += 1
            return Vt, Vm, Vg, Vabs, Vbs, E_array

        Emax_initial = 1e5  # [V/cm]
        step = Emax_initial / 10
        Error = 0.05  # [V]
        count = 1
        Vt, Vm, Vg, Vabs, Vbs, E_array = voltage(Emax_initial, j, junction, x)
        while abs(Vt - Vgoal) > Error:
            if Vt > Vgoal:
                Emax_initial -= step
                Vt, Vm, Vg, Vabs, Vbs, E_array = voltage(Emax_initial, j, junction, x)
                if Vt < Vgoal:
                    step = step / (1 + count)
                    count += 1
            else:
                Emax_initial += step
                Vt, Vm, Vg, Vabs, Vbs, E_array = voltage(Emax_initial, j, junction, x)
                if Vt > Vgoal:
                    step = step / (1 + count)
                    count += 1
        return Vt, Vm, Vg, Vabs, Vbs, Emax_initial, E_array

    def FieldTransform(self, Ej, interface_um):
        """
        :param Ej: pn junction field
        :param interface_um: [xj, x_cg, x_ga, x_ab] in um
        :return: V/cm
        """
        xj, x_cg, x_ga, x_ab = np.asarray(interface_um) * 1e-4  # [cm]
        position_cm = self.X * 1e-4
        Field = []
        k = None
        for i in range(len(position_cm) - 1):
            if position_cm[i] == xj:
                k = i
                break
            elif (position_cm[i] - xj) * (position_cm[i + 1] - xj) < 0:
                k = i
                break
        for i in range(len(position_cm)):
            if i < k:
                if i == k - 1:
                    Field.append(Ej)
                else:
                    Field.append(0)
            else:
                if position_cm[i] <= x_cg:  # InP
                    dx = position_cm[i] - position_cm[i - 1]
                    dE = phys.e * self.Y[i] / phys.eps_InP * dx
                elif x_cg < position_cm[i] <= x_ga:  # InGaAsP
                    dx = position_cm[i] - position_cm[i - 1]
                    dE = phys.e * self.Y[i] / phys.eps_InGaAsP * dx
                elif x_ga < position_cm[i] <= x_ab:  # InGaAs
                    dx = position_cm[i] - position_cm[i - 1]
                    dE = phys.e * self.Y[i] / phys.eps_InGaAs * dx
                elif x_ab < position_cm[i]:  # InP
                    dx = position_cm[i] - position_cm[i - 1]
                    dE = phys.e * self.Y[i] / phys.eps_InP * dx
                else:
                    raise BaseException("The position unit or data might have some problems")
                E = Field[-1] - abs(dE)
                if E < 0:
                    Field.append(0)
                    break
                Field.append(E)
        while len(Field) < len(position_cm):
            Field.append(0)
        return np.asarray(Field)  # [V/cm]


class WidthProfile(CSV):
    def __init__(self, data, xlabel, ylabel):
        super().__init__(data, xlabel, ylabel)

    def plot(self, number, label=None):
        if number != -1:
            figure(num=number, figsize=(20, 9.3), dpi=80, facecolor='w', edgecolor='k')
            plt.figure(number)
        if label is None:
            plt.plot(self.X, abs(self.Y), '-b')
        else:
            plt.plot(self.X, abs(self.Y), '-b', label=label)
            plt.legend(loc='best')
        plt.grid(True)
        plt.title('Depletion width vs. voltage')
        plt.xlabel(r'Voltage (V)')
        plt.ylabel(r'Width ($\mu$m)')


class IV(CSV):
    def __init__(self, data, xlabel, ylabel, area):
        super().__init__(data, xlabel, ylabel)
        self.__data = data
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.J = self.Y / area

    def find_E_at_J(self, J):
        # 此 CurveList 的長度應該為 1，也就是只有一個元素。
        CurveList = [curve for curve in utils.parse_csv(self.__data, 2, 0, 0, 'Yes') if
                     curve.split("(", 2)[1].split(")", 2)[0] == self.xlabel.split("(", 2)[1].split(")", 2)[0] and
                     'ElectricField' in curve]
        return utils.find(self.J, CSV(self.__data, CurveList[0], CurveList[0]).Y, J, 'linear')

    def find_V(self, J, x1=-499, x2=-0.1):
        def func1(x):
            return J
        def func2(x):
            return utils.find(self.X, self.J, x, 'log')
        return utils.find_intersection(func1, func2, x1, x2, 1e-13)

    def find_j(self, V):
        J = utils.find(self.X, self.J, V, 'log')
        if J is None:
            raise BaseException("Can't find j at V=%s" % V)
        else:
            return J

    def find_I(self, V):
        I = utils.find(self.X, self.Y, V, 'log')
        if I is None:
            raise BaseException("Can't find I at V=%s" % V)
        else:
            return I

    def plot(self, number, color='b', label=None):
        if number != -1:
            figure(num=number, figsize=(20, 9.3), dpi=80, facecolor='w', edgecolor='k')
            plt.figure(number)
            plt.grid(True)
            plt.yscale('log')
            plt.title('IV curve')
            plt.xlabel(r'Voltage (V)')
            plt.ylabel(r'Current (A)')
        if label is None:
            plt.plot(self.X, abs(self.Y), color)
        else:
            plt.plot(self.X, abs(self.Y), color, label=label)
            if number != -1:
                plt.legend(loc='best')

    def plotJV(self, number, color='b', label=None):
        if number != -1:
            figure(num=number, figsize=(20, 9.3), dpi=80, facecolor='w', edgecolor='k')
            plt.figure(number)
            plt.grid(True)
            plt.yscale('log')
            plt.title('JV curve')
            plt.xlabel(r'Voltage (V)')
            plt.ylabel(r'Current density (A/cm$^2$)')
        if label is None:
            plt.plot(self.X, abs(self.J), color)
        else:
            plt.plot(self.X, abs(self.J), color, label=label)
            plt.legend(loc='best')

    def plotJVEmax(self, number, EmaxProfile, location_Vabs, color='-b', fillstyle='none', label=None):
        V_nonzero, Emax, JV = self.DataJVEmax(EmaxProfile, location_Vabs)
        if number != -1:
            figure(num=number, figsize=(20, 9.3), dpi=80, facecolor='w', edgecolor='k')
            plt.figure(number)
            plt.grid(True)
            plt.title(r'$\ln(J/V)$ vs. $1/E_{max}$')
            plt.xlabel(r'$1/E_m$ (cm/V) $\times 10^6$')
            plt.ylabel(r'$\ln$(J / V)')
        if label is None:
            plt.plot(1 / Emax * 1e6, np.log(JV), color, fillstyle=fillstyle)
        else:
            plt.plot(1 / Emax * 1e6, np.log(JV), color, fillstyle=fillstyle, label=label)  # [cm/V * 1e6, ln(J/V)]
            if number != -1:
                plt.legend(loc='best')

    def DataJVEmax(self, EmaxProfile, location_Vabs):
        if len(EmaxProfile) != len(self.X):
            raise BaseException("The size of IV and Emax are not equal.")
        else:
            Vabs_data = utils.parse_csv(location_Vabs, 1, 0, 0, 'Yes')
            Vabs = utils.interpolation(Vabs_data['Vt'], self.X, Vabs_data['Vabs'])
            V = self.X
            J = self.J
            V_nonzero = np.empty([0])
            JV = np.empty([0])
            Emax = np.empty([0])
            for i in range(len(V)):
                if V[i] != 0 and J[i] != 0:
                    V_nonzero = np.append(V_nonzero, abs(V[i]))
                    JV = np.append(JV, abs(J[i] / V[i]))
                    Emax = np.append(Emax, EmaxProfile[i])
        return V_nonzero, Emax, JV  # [V], [V/cm], [A/cm2-V]


class ElectricField(CSV):
    def __init__(self, data, xlabel, ylabel, interfaces, position_unit='um', x=None, y=None):
        """
        interfaces 是指材料的界面位置。
        :param data:
        :param xlabel:
        :param ylabel:
        :param interfaces: [Xj, x_InP_InGaAsP, x_InGaAsP_InGaAs, x_InGaAs_InP]
        :param position_unit: 通常電場單位都是 V/cm，所以位置物理量的單位必須轉換為 cm。
        """
        super().__init__(data, xlabel, ylabel, x, y)
        self._Ef_X = None
        self._Ef_Y = None
        self._files = None
        self.__interfaces = interfaces
        self.__Xj = interfaces[0]
        self.__Xcg = interfaces[1]
        self.__Xga = interfaces[2]
        self.__Xab = interfaces[3]
        self.__unit = position_unit
        if position_unit == 'um':
            c = 1e-4
        elif position_unit == 'cm':
            c = 1
        else:
            raise BaseException("Check unit.")
        self.V1 = self.integrate('no', self.__Xj) * c
        self.V2 = self.integrate(self.__Xj, self.__Xcg) * c
        self.V3 = self.integrate(self.__Xcg, self.__Xga) * c
        self.V4 = self.integrate(self.__Xga, self.__Xab) * c
        self.V5 = self.integrate(self.__Xab, 'no') * c
        self.Vapp = self.V1 + self.V2 + self.V3 + self.V4 + self.V5

    def impact_ionization(self, model='vanOverst', temperature=300):
        if self.__unit == 'um':
            c = 1
        elif self.__unit == 'cm':
            c = 1e4
        else:
            raise BaseException("Check unit.")
        x = self.X * c
        interface = np.asarray([self.__Xcg, self.__Xga, self.__Xab]) * c
        E = self.Y
        coef = {'alpha': np.empty([0]), 'beta': np.empty([0])}
        for i in range(len(x)):
            coef['alpha'] = np.append(coef['alpha'], II.alpha(E[i], x[i], interface, model, temperature))
            coef['beta'] = np.append(coef['beta'], II.beta(E[i], x[i], interface, model, temperature))
        # Unit of input x in multiplication_factor is [um]
        # interface = [x_InP_InGaAsP, x_InGaAsP_InGaAs, x_InGaAs_InP]
        start_time = time.time()
        Mn, Mp = phys.multiplication_factor(x, E, interface, model=model)
        #print('Multiplication factor calculation time: %.5s seconds' % (time.time() - start_time))
        return coef, Mn, Mp

    @staticmethod
    def load_ef_ponts(input='EFCutlines/'):
        # [1] /Users/Ethan/PycharmProjects/Python/Avalanche Photodiode/Planar/1A2F/22/
        # [2] /Users/Ethan/PycharmProjects/Python/Avalanche Photodiode/Planar/1A2F/22/EFCutlines/
        # [3]/Users/Ethan/PycharmProjects/Python/Avalanche Photodiode/Planar/1A2F/22/EFCutlines/*.csv
        directory = os.getcwd() + '/'  # [1]
        Cutline_dir = directory + input  # [2]
        files = glob.glob(Cutline_dir + '*.csv')  # [3]
        Ylist = []
        for i in files:
            Ylist.append(float(i.replace(Cutline_dir, "").replace(".csv", "")))
        X, files = utils.orderfunction(Ylist, files, 'increasing')
        Y = [- np.asarray(utils.parse_csv(i.replace(directory, ""), 1, 0, 0, 'Yes')['X']) for i in files]
        return X, Y, files

    @staticmethod
    def find_Ef(x, y, X, Y, files):
        x = abs(x)

        def distance(p1, p2):
            return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

        def find_index(x0, x):
            temp = abs(x0 - x[0])
            ans = 0
            ans2 = 0
            for i in range(len(x)):
                if abs(x0 - x[i]) < temp:
                    temp = abs(x0 - x[i])
                    ans2 = ans
                    ans = i
            return ans, ans2

        if type(x) is not np.ndarray:
            i_x1, i_x2 = find_index(x, X)
            i_y11, i_y12 = find_index(y, Y[i_x1])
            i_y21, i_y22 = find_index(y, Y[i_x2])
            points = [[X[i_x1], Y[i_x1][i_y11]], [X[i_x1], Y[i_x1][i_y12]], [X[i_x2], Y[i_x2][i_y21]],
                      [X[i_x2], Y[i_x2][i_y22]]]
            Efs = [utils.parse_csv(files[find_index(i[0], X)[0]], 1, 0, 0, 'Yes')['ElectricField'][
                       find_index(i[1], Y[find_index(i[0], X)[0]])[0]] for i in points]
            d_points = np.asarray([distance([x, y], i) for i in points])
            return sum(Efs[i] * np.exp(- d_points[i]) for i in range(len(Efs))) / sum(np.exp(- d_points))
        else:
            Ef_output = np.zeros((len(y), len(x)))
            for s in range(len(x)):
                for t in range(len(y)):
                    i_x1, i_x2 = find_index(x[s], X)
                    i_y11, i_y12 = find_index(y[t], Y[i_x1])
                    i_y21, i_y22 = find_index(y[t], Y[i_x2])
                    points = [[X[i_x1], Y[i_x1][i_y11]], [X[i_x1], Y[i_x1][i_y12]], [X[i_x2], Y[i_x2][i_y21]],
                              [X[i_x2], Y[i_x2][i_y22]]]
                    Efs = [utils.parse_csv(files[find_index(i[0], X)[0]], 1, 0, 0, 'Yes')['ElectricField'][
                               find_index(i[1], Y[find_index(i[0], X)[0]])[0]] for i in points]
                    d_points = np.asarray([distance([x[s], y[t]], i) for i in points])
                    Ef_output[t][s] = sum(Efs[i] * np.exp(- d_points[i]) for i in range(len(Efs))) / sum(np.exp(- d_points))
            return Ef_output

    def joint_breakdown_probability(self, plot, form, material_interface, figure_number=1):
        alpha = np.asarray([II.alpha(self.Y[i], self.X[i], material_interface, form) for i in range(len(self.X))])
        beta = np.asarray([II.beta(self.Y[i], self.X[i], material_interface, form) for i in range(len(self.X))])
        f = [np.exp(utils.ydx(self.X * 1e-4, alpha - beta, 0, i)) for i in range(len(self.X))]

        def f1(pe0):
            return -np.log(1 - pe0)

        def f2(pe0):
            func = [pe0 * beta[i] * f[i] / (pe0 * f[i] + 1 - pe0) for i in range(len(self.X))]
            return utils.ydx(self.X * 1e-4, func, 0, len(self.X) - 1)

        pe0 = utils.find_intersection(f1, f2, 1e-7, 1, 1e-7)
        prob_pair = [pe0 * f[i] / (pe0 * f[i] + 1 - pe0) for i in range(len(f))]
        prob_electron = [1 - (1 - pe0) * np.exp(utils.ydx(self.X * 1e-4, beta * prob_pair, 0, i)) for i in
                     range(len(self.X))]

        prob_hole = [(prob_pair[i] - prob_electron[i]) / (1 - prob_electron[i]) for i in range(len(self.X))]
        if plot == 'Yes':
            figure(num=figure_number, figsize=(20, 9.2), dpi=80, facecolor='w', edgecolor='k')
            plt.figure(figure_number)
            x = np.arange(-1, 1, 0.01)
            plt.plot(x, f1(x), '-bo', label=r'f$_1$[$P_e(0)$]=-ln[1-$P_e(0)$]')
            plt.plot(x, f2(x), '-ro',
                     label=r'f$_2$[$P_e(0)$]=$\int_0^w\frac{P_e(0)\alpha_h(x)f(x)}{P_e(0)f(x)+1-P_e(0)}dx$')
            plt.vlines(x=pe0, ymin=0, ymax=4, color='k', linestyles='-.', label=r'$P_h(0)=%s$' % pe0)
            plt.xlabel(r'$P_e(0)$')
            plt.ylabel(r'f[$P_e(0)$]')
            plt.legend(loc='best')
            plt.grid(True)

        return np.asarray(prob_pair), np.asarray(prob_hole), np.asarray(prob_electron)

    def ionization_rate(self, form, material_interface):
        alpha = np.asarray([II.alpha(self.Y[i], self.X[i], material_interface, form) for i in range(len(self.X))])
        beta = np.asarray([II.beta(self.Y[i], self.X[i], material_interface, form) for i in range(len(self.X))])
        return alpha, beta


class EdgeJunction(CSV):
    def __init__(self, data, xlabel, ylabel):
        super().__init__(data, xlabel, ylabel)
        self.X, self.Y = utils.orderfunction(self.Y, - self.X, 'increasing')
        self.__Xj = None
        self.__Yj = None
        self.__xj_p = None
        self.__xj_pp = None
        self.__yj_p = None
        self.__yj_pp = None
        self.__Kappa_j = None

    @staticmethod
    def JunctionGenerator(input='Cutlines/', output='data/generated_edge_profile.csv'):
        # [1] /Users/Ethan/PycharmProjects/Python/Avalanche Photodiode/Planar/1A2F/22/
        # [2] /Users/Ethan/PycharmProjects/Python/Avalanche Photodiode/Planar/1A2F/22/Cutlines/
        # [3] /Users/Ethan/PycharmProjects/Python/Avalanche Photodiode/Planar/1A2F/22/Cutlines/*.csv
        directory = os.getcwd() + '/'  # [1]
        Cutline_dir = directory + input  # [2]
        files = glob.glob(Cutline_dir + '*.csv')  # [3]
        Ylist = []
        for i in files:
            Ylist.append(float(i.replace(Cutline_dir, "").replace(".csv", "")))
        Ylist, files = utils.orderfunction(Ylist, files, 'increasing')
        xj = []
        Ylist_which_has_xj = []
        for i in files:
            cutline = utils.parse_csv(i.replace(directory, ""), 1, 0, 0, 'Yes')
            conc = cutline['DopingConcentration']
            position = cutline['X']
            for j in range(len(position)):
                if position[j] < -3.7:
                    if conc[j] * conc[j + 1] < 0:
                        x0 = position[j]
                        dx = position[j + 1] - position[j]
                        ratio = abs(conc[j]) / (abs(conc[j]) + abs(conc[j + 1]))
                        xj.append(x0 + dx * ratio)
                        Ylist_which_has_xj.append(float(i.replace(Cutline_dir, "").replace(".csv", "")))
                        break
        utils.save(output, 'y(TCAD)', 'x(TCAD)', Ylist_which_has_xj, xj)

    @staticmethod
    def weight_function(dl, beta=0):
        if type(dl) is int:
            if dl != 0:
                alpha = 1
                k = 0
                return alpha * np.exp(- beta * dl ** 2) / dl ** k
            else:
                return 0
        else:
            alpha = 10
            k = 2
            return alpha * np.exp(- beta * dl ** 2)

    def _Lk(self, k):
        dX_k = self.X[k + 1] - self.X[k]
        dY_k = self.Y[k + 1] - self.Y[k]
        ans = (dX_k ** 2 + dY_k ** 2) ** 0.5
        return ans

    def _dLij(self, i, j):
        if i > j:
            return sum(self._Lk(k) for k in range(j, i))
        elif i < j:
            return sum(- self._Lk(k) for k in range(i, j))
        else:
            return 0

    def curvature(self, position, q, method='Independent'):
        j = utils.value_convert_index(position, self.X)
        j1 = j - q
        j2 = j + q + 1
        if j < q:
            j1 = 0
        if j2 > len(self.X):
            j2 = len(self.X)

        a = sum(self.weight_function(self._dLij(i, j)) ** 2 * self._dLij(i, j) ** 2 for i in range(j1, j2))
        b = 0.5 * sum(
            self.weight_function(self._dLij(i, j)) ** 2 * self._dLij(i, j) ** 3 for i in range(j1, j2))
        c = 0.25 * sum(
            self.weight_function(self._dLij(i, j)) ** 2 * self._dLij(i, j) ** 4 for i in range(j1, j2))
        e = sum(self.weight_function(self._dLij(i, j)) ** 2 * self._dLij(i, j) * (self.X[i] - self.X[j]) for i in
                range(j1, j2))
        f = 0.5 * sum(self.weight_function(self._dLij(i, j)) ** 2
                      * self._dLij(i, j) ** 2 * (self.X[i] - self.X[j])
                      for i in range(j1, j2))
        g = sum(self.weight_function(self._dLij(i, j)) ** 2 * self._dLij(i, j) * (self.Y[i] - self.Y[j]) for i in
                range(j1, j2))
        h = 0.5 * sum(
            self.weight_function(self._dLij(i, j)) ** 2 * self._dLij(i, j) ** 2 * (self.Y[i] - self.Y[j]) for i in
            range(j1, j2))

        if a * c - b ** 2 == 0:
            return ('none', 'none', 100), (self.__Xj, self.__Yj), (self.__xj_p, self.__xj_pp, self.__yj_p, self.__yj_pp, self.__Kappa_j)

        self.__Xj = self.X[j]
        self.__Yj = self.Y[j]
        if method == 'Independent':
            self.__xj_p = (c * e - b * f) / (a * c - b ** 2)
            self.__xj_pp = (a * f - b * e) / (a * c - b ** 2)
            self.__yj_p = (c * g - b * h) / (a * c - b ** 2)
            self.__yj_pp = (a * h - b * g) / (a * c - b ** 2)
            self.__Kappa_j = (e * h - f * g) / (a * c - b ** 2)
        elif method == 'Dependent':
            self.__xj_p = (c * e - b * f) / (a * c - b ** 2)
            self.__yj_p = (c * g - b * h) / (a * c - b ** 2)
            if abs(self.__xj_p) < abs(self.__yj_p):
                self.__yj_p = np.sign(abs(self.__yj_p)) * (1 - abs(self.__xj_p) ** 2) ** 0.5
                self.__xj_pp = (a * f - b * e) / (a * c - b ** 2)
                self.__yj_pp = - (self.__xj_p * self.__xj_pp) / self.__yj_p
            else:
                self.__xj_p = np.sign(abs(self.__xj_p)) * (1 - abs(self.__yj_p) ** 2) ** 0.5
                self.__yj_pp = (a * h - b * g) / (a * c - b ** 2)
                self.__xj_pp = - (self.__yj_p * self.__yj_pp) / self.__xj_p
            self.__Kappa_j = self.__xj_p * self.__yj_pp - self.__yj_p * self.__xj_pp
            self.__xj_pp = - self.__yj_p
            self.__yj_pp = self.__xj_p
        else:
            raise BaseException("Wrong method.")

        r = 1 / self.__Kappa_j
        if self.__xj_p != 0:
            self.__m = self.__yj_p / self.__xj_p
            if self.__xj_p > 0:
                theta = np.arctan(self.__m)
            else:
                theta = np.arctan(self.__m) + np.pi
        else:
            self.__m = float('inf')
            theta = np.pi / 2
        xc = r * np.cos(np.pi / 2 + theta) + self.__Xj
        yc = r * np.sin(np.pi / 2 + theta) + self.__Yj

        return (xc, yc, r), (self.__Xj, self.__Yj), (self.__xj_p, self.__xj_pp, self.__yj_p, self.__yj_pp, self.__Kappa_j)

    def plotTangentLine(self):
        def line(m, x, x0, y0):
            if m != float('inf'):
                return m * (x - x0) + y0
            else:
                raise BaseException("The slope is infinite.")

        plt.plot(self.X, line(self.__m, self.X, self.__Xj, self.__Yj), '-.k', label='slope: %.3f' % self.__m)

    def plotWi(self, j, q):
        j1 = j - q
        j2 = j + q + 1
        if j < q:
            j1 = 0
        if j2 > len(self.X):
            j2 = len(self.X)
        wi = [self.weight_function(self._dLij(i, j)) for i in range(j1, j2) if i != j]
        Lij = [self._dLij(i, j) for i in range(j1, j2) if i != j]
        Lij, wi = utils.orderfunction(Lij, wi, 'increasing')
        plt.plot(Lij, wi, '-bo')


class SurfacePlot(CSV):
    def __init__(self, data, xlabel, ylabel):
        super().__init__(data, xlabel, ylabel)

    @staticmethod
    def load_ponts(input):
        # [1] /Users/Ethan/PycharmProjects/Python/Avalanche Photodiode/Planar/1A2F/22/
        # [2] /Users/Ethan/PycharmProjects/Python/Avalanche Photodiode/Planar/1A2F/22/EFCutlines/
        # [3]/Users/Ethan/PycharmProjects/Python/Avalanche Photodiode/Planar/1A2F/22/EFCutlines/*.csv
        directory = os.getcwd() + '/'  # [1]
        Cutline_dir = directory + input  # [2]
        files = glob.glob(Cutline_dir + '*.csv')  # [3]
        Ylist = []
        for i in files:
            Ylist.append(float(i.replace(Cutline_dir, "").replace(".csv", "")))
        X, files = utils.orderfunction(Ylist, files, 'increasing')
        Y = [- np.asarray(utils.parse_csv(i.replace(directory, ""), 1, 0, 0, 'Yes')['X']) for i in files]
        return X, Y, files

    @staticmethod
    def surface_function(x, y, X, Y, files, yaxis):
        x = abs(x)

        def distance(p1, p2):
            return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

        def find_index(x0, x):
            temp = abs(x0 - x[0])
            ans = 0
            ans2 = 0
            for i in range(len(x)):
                if abs(x0 - x[i]) < temp:
                    temp = abs(x0 - x[i])
                    ans2 = ans
                    ans = i
            return ans, ans2

        if type(x) is not np.ndarray:
            i_x1, i_x2 = find_index(x, X)
            i_y11, i_y12 = find_index(y, Y[i_x1])
            i_y21, i_y22 = find_index(y, Y[i_x2])
            points = [[X[i_x1], Y[i_x1][i_y11]], [X[i_x1], Y[i_x1][i_y12]], [X[i_x2], Y[i_x2][i_y21]],
                      [X[i_x2], Y[i_x2][i_y22]]]
            Efs = [utils.parse_csv(files[find_index(i[0], X)[0]], 1, 0, 0, 'Yes')['ElectricField'][
                       find_index(i[1], Y[find_index(i[0], X)[0]])[0]] for i in points]
            d_points = np.asarray([distance([x, y], i) for i in points])
            return sum(Efs[i] * np.exp(-d_points[i]) for i in range(len(Efs))) / sum(np.exp(-d_points))
        else:
            Ef_output = np.zeros((len(y), len(x)))
            for s in range(len(x)):
                for t in range(len(y)):
                    i_x1, i_x2 = find_index(x[s], X)
                    i_y11, i_y12 = find_index(y[t], Y[i_x1])
                    i_y21, i_y22 = find_index(y[t], Y[i_x2])
                    points = [[X[i_x1], Y[i_x1][i_y11]], [X[i_x1], Y[i_x1][i_y12]], [X[i_x2], Y[i_x2][i_y21]],
                              [X[i_x2], Y[i_x2][i_y22]]]
                    Efs = [utils.parse_csv(files[find_index(i[0], X)[0]], 1, 0, 0, 'Yes')[yaxis][
                               find_index(i[1], Y[find_index(i[0], X)[0]])[0]] for i in points]
                    d_points = np.asarray([distance([x[s], y[t]], i) for i in points])
                    Ef_output[t][s] = sum(Efs[i] * np.exp(-d_points[i]) for i in range(len(Efs))) / sum(np.exp(-d_points))
            return Ef_output


class CreateMesh(object):
    def __init__(self):
        self.__line = np.empty([0])
        self.__lastnumber = 0

    def add_line(self, a, b, density):
        """
        起點是 a，終點是 b，在這區間中，每條線的間距為 density
        :param a: 0
        :param b: 2
        :param density:0.2
        :return: np.linspace(a, b, int((b - a) / density + 1))
                 = np.linspace(0, 2, 11) = [0, 0.2, 0.4, 0.6, ... , 1.6, 1.8, 2.0]
        """
        if len(self.__line) == 0:
            self.__lastnumber = b
        else:
            if self.__lastnumber != a:
                raise BaseException("Two mesh intervals are not matched, [..,%s] and [%s,...] " % (self.__lastnumber, a))
            else:
                self.__lastnumber = b
        input_line = np.linspace(a, b, int((b - a) / density + 1))
        self.__line = np.append(self.__line, input_line)
        self.__line.sort()

    def sort_line(self):
        temp = np.empty([0])
        for element in self.__line:
            if element not in temp:
                temp = np.append(temp, element)
        self.__line = temp
        print('Grid number: %s' % len(self.__line))

    def reflection(self):
        self.__line = np.append(self.__line, - self.__line)
        self.__line.sort()

    def get_mesh(self):
        return self.__line

    def get_lastnumber(self):
        return self.__lastnumber


class CreateMesh2(object):
    def __init__(self, X, default_interval):
        self.line = np.linspace(min(X), max(X), (max(X) - min(X)) / default_interval)
        self.lastnumber = 0
        self.position = []
        self.interval = []

    def set_meshline(self, position, interval):
        self.position.append(position)
        self.interval.append(interval)

    def generate(self):
        self.position, self.interval = utils.orderfunction(self.position, self.interval, type='increasing')

        # 檢查是否重疊
        for i in range(len(self.position)):
            if i == 0:
                if self.position[i] + self.interval[i] >= self.position[i + 1]:
                    raise BaseException("Mesh line overlap: %s and %s" % (self.position[i], self.position[i + 1]))
            elif i == len(self.position) - 1:
                if self.position[i] - self.interval[i] <= self.position[i - 1]:
                    raise BaseException("Mesh line overlap: %s and %s" % (self.position[i], self.position[i - 1]))
            else:
                if self.position[i] + self.interval[i] >= self.position[i + 1]:
                    raise BaseException("Mesh line overlap: %s and %s" % (self.position[i], self.position[i + 1]))
                elif self.position[i] - self.interval[i] <= self.position[i - 1]:
                    raise BaseException("Mesh line overlap: %s and %s" % (self.position[i], self.position[i - 1]))


class CV(object):
    def __init__(self):
        self.__V = []
        self.__C = []
        self.Vnet = None
        self.Cnet = None
        self.cp = None
        self.cl = None
        self.r12 = None

    @staticmethod
    def align_xj(position, concentration, x1, x2, dopingmin):
        Xj = 0
        for i in range(len(concentration)):
            if x1 < position[i] < x2:
                if abs(concentration[i]) < dopingmin:
                    dopingmin = abs(concentration[i])
                    Xj = position[i]

        for i in range(len(position)):
            position[i] -= Xj
        return np.asarray(position)

    def add_CV(self, V, C):
        self.__V.append(V)
        self.__C.append(C)

    def combine(self):
        self.V = self.__V[0]
        for i in range(1, len(self.__V)):
            self.V = utils.merge(self.V, self.__V[i])
        for i in range(len(self.__C)):
            self.__C[i] = utils.interpolation(self.__V[i], self.V, self.__C[i])
        self.Vnet = self.V
        self.Cnet = self.__C
        return self.Vnet, self.Cnet

    @staticmethod
    def find_doping(v, c, area):
        c = c / area
        q = 1.6e-19
        eps = 13.9 * 8.85e-14
        position = np.empty([0])
        voltage = np.empty([0])
        capacitance = np.empty([0])
        concentration = np.empty([0])
        for i in range(len(c)):
            if - c[i] > 0 and v[i] < 1.1:
                position = np.append(position, - eps / c[i])
                voltage = np.append(voltage, v[i])
                capacitance = np.append(capacitance, c[i])
                dCpdV = (c[i + 1] - c[i]) / (v[i + 1] - v[i])
                concentration = np.append(concentration, abs(c[i] ** 3 / (q * eps) / dCpdV))
        return position, voltage, capacitance, concentration

    def find_cpl(self, mode, r, dr_um):
        q = 1.6e-19
        eps = 13.9 * 8.85e-14
        if self.Cnet is None:
            raise BaseException("You haven't used CV.combine() method.")

        self.cp = []
        self.cl = []
        self.r12 = []
        for i in range(len(self.Cnet)):
            for j in range(i + 1, len(self.Cnet)):
                c1, c2 = self.Cnet[i], self.Cnet[j]
                r1, r2 = r[i], r[j]
                if mode == 'cylinder':
                    R1 = r1 + 3.5  # [um]
                    R2 = r2 + 3.5  # [um]
                    r1_cm = R1 * 1e-4  # [cm]
                    r2_cm = R2 * 1e-4  # [cm]
                    self.r12.append((r1, r2))
                    self.cp.append((r2_cm * c1 - r1_cm * c2) / (np.pi * r1_cm * r2_cm * (r1_cm - r2_cm)))  # [F/cm^2]
                    self.cl.append((r2_cm ** 2 * c1 - r1_cm ** 2 * c2) / (2 * np.pi * r1_cm * r2_cm * (r2_cm - r1_cm)))
                elif mode == 'cone':
                    dr_cm = dr_um * 1e-4
                    r1a = r1 * 1e-4
                    r1b = r1 * 1e-4 + dr_cm
                    r2a = r2 * 1e-4
                    r2b = r2 * 1e-4 + dr_cm
                    cp_temp = ((r2a + r2b) * c1 - (r1a + r1b) * c2) / (
                            np.pi * r1a ** 2 * (r2a + r2b) - np.pi * r2a ** 2 * (r1a + r1b))
                    H_temp = eps / cp_temp
                    self.r12.append((r1, r2))
                    self.cp.append(cp_temp)  # [F/cm^2]
                    self.cl.append((r2a ** 2 * c1 - r1a ** 2 * c2) / (np.pi * (H_temp ** 2 + dr_cm ** 2) ** 0.5
                                                                      * ((r1a + r1b) * r2a ** 2 - (r2a + r2b) * r1a ** 2)))  # [F/cm^2]
        return self.r12, self.cp, self.cl


class MultipleCSV(object):
    def __init__(self, location, parameter_dict):
        """
        讀取整排資料的 *.csv
        :param location: 檔案位址
        :param parameter_dict: {'Nc': [1e17, 2e17, 3e17], 'Thickness': [0.2, 0.3, 0.6, 0.8, 0.9, 1.2]}
        """
        self.__location = location
        self.__curve = utils.parse_csv(self.__location, 2, 0, 0, 'Yes')
        self.__parameter_dict = parameter_dict
        self.__parameter_name = list(self.__parameter_dict)


class SuperXY(object):
    def __init__(self, numpy_array_x, numpy_array_y):
        self.__x, self.__y = utils.CheckOverlap(numpy_array_x, numpy_array_y)

    def __abs__(self):
        return utils.ydx(self.__x, self.__y ** 2, 0, len(self.__x)-1)

    def __add__(self, other):
        x = utils.merge(self.__x, other.__x)
        y1 = utils.interpolation(self.__x, x, self.__y, extra='No')
        y2 = utils.interpolation(other.__x, x, other.__y, extra='No')
        return x, y1 + y2

    def __sub__(self, other):
        x = utils.merge(self.__x, other.__x)
        y1 = utils.interpolation(self.__x, x, self.__y)
        y2 = utils.interpolation(other.__x, x, other.__y)
        return x, y1 - y2

    def __mul__(self, other):
        x = utils.merge(self.__x, other.__x)
        y1 = utils.interpolation(self.__x, x, self.__y)
        y2 = utils.interpolation(other.__x, x, other.__y)
        return x, y1 * y2

    def __truediv__(self, other):
        x = utils.merge(self.__x, other.__x)
        y1 = utils.interpolation(self.__x, x, self.__y)
        y2 = utils.interpolation(other.__x, x, other.__y)
        return x, y1 / y2