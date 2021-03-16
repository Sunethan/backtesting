import numpy as np
import utils

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
