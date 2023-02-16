import math
import numpy as np
import bisect
import csv
from scipy import interpolate
class CubicSpline1D:
    """
    1D Cubic Spline class

    Parameters
    ----------
    x : list
        x coordinates for data points. This x coordinates must be
        sorted
        in ascending order.
    y : list
        y coordinates for data points

    Examples
    --------
    You can interpolate 1D data points.

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> x = np.arange(5)
    >>> y = [1.7, -6, 5, 6.5, 0.0]
    >>> sp = CubicSpline1D(x, y)
    >>> xi = np.linspace(0.0, 5.0)
    >>> yi = [sp.calc_position(x) for x in xi]
    >>> plt.plot(x, y, "xb", label="Data points")
    >>> plt.plot(xi, yi , "r", label="Cubic spline interpolation")
    >>> plt.grid(True)
    >>> plt.legend()
    >>> plt.show()

    .. image:: cubic_spline_1d.png

    """

    def __init__(self, x, y):

        h = np.diff(x)
        if np.any(h < 0):
            raise ValueError("x coordinates must be sorted in ascending order")

        self.a, self.b, self.c, self.d = [], [], [], []
        self.x = x
        self.y = y
        self.nx = len(x)  # dimension of x

        # calc coefficient a
        self.a = [iy for iy in y]

        # calc coefficient c
        A = self.__calc_A(h)
        B = self.__calc_B(h, self.a)
        self.c = np.linalg.solve(A, B)

        # calc spline coefficient b and d
        for i in range(self.nx - 1):
            d = (self.c[i + 1] - self.c[i]) / (3.0 * h[i])
            b = 1.0 / h[i] * (self.a[i + 1] - self.a[i]) \
                - h[i] / 3.0 * (2.0 * self.c[i] + self.c[i + 1])
            self.d.append(d)
            self.b.append(b)

    def calc_position(self, x):
        """
        Calc `y` position for given `x`.

        if `x` is outside the data point's `x` range, return None.

        Returns
        -------
        y : float
            y position for given x.
        """
        if x < self.x[0]:
            return None
        elif x > self.x[-1]:
            return None

        i = self.__search_index(x)
        dx = x - self.x[i]
        position = self.a[i] + self.b[i] * dx + \
                   self.c[i] * dx ** 2.0 + self.d[i] * dx ** 3.0

        return position

    def calc_first_derivative(self, x):
        """
        Calc first derivative at given x.

        if x is outside the input x, return None

        Returns
        -------
        dy : float
            first derivative for given x.
        """

        if x < self.x[0]:
            return None
        elif x > self.x[-1]:
            return None

        i = self.__search_index(x)
        dx = x - self.x[i]
        dy = self.b[i] + 2.0 * self.c[i] * dx + 3.0 * self.d[i] * dx ** 2.0
        return dy

    def calc_second_derivative(self, x):
        """
        Calc second derivative at given x.

        if x is outside the input x, return None

        Returns
        -------
        ddy : float
            second derivative for given x.
        """

        if x < self.x[0]:
            return None
        elif x > self.x[-1]:
            return None

        i = self.__search_index(x)
        dx = x - self.x[i]
        ddy = 2.0 * self.c[i] + 6.0 * self.d[i] * dx
        return ddy

    def __search_index(self, x):
        """
        search data segment index
        """
        return bisect.bisect(self.x, x) - 1

    def __calc_A(self, h):
        """
        calc matrix A for spline coefficient c
        """
        A = np.zeros((self.nx, self.nx))
        A[0, 0] = 1.0
        for i in range(self.nx - 1):
            if i != (self.nx - 2):
                A[i + 1, i + 1] = 2.0 * (h[i] + h[i + 1])
            A[i + 1, i] = h[i]
            A[i, i + 1] = h[i]

        A[0, 1] = 0.0
        A[self.nx - 1, self.nx - 2] = 0.0
        A[self.nx - 1, self.nx - 1] = 1.0
        return A

    def __calc_B(self, h, a):
        """
        calc matrix B for spline coefficient c
        """
        B = np.zeros(self.nx)
        for i in range(self.nx - 2):
            B[i + 1] = 3.0 * (a[i + 2] - a[i + 1]) / h[i + 1] \
                       - 3.0 * (a[i + 1] - a[i]) / h[i]
        return B


class CubicSpline2D:
    """
    Cubic CubicSpline2D class

    Parameters
    ----------
    x : list
        x coordinates for data points.
    y : list
        y coordinates for data points.

    Examples
    --------
    You can interpolate a 2D data points.

    >>> import matplotlib.pyplot as plt
    >>> x = [-2.5, 0.0, 2.5, 5.0, 7.5, 3.0, -1.0]
    >>> y = [0.7, -6, 5, 6.5, 0.0, 5.0, -2.0]
    >>> ds = 0.1  # [m] distance of each interpolated points
    >>> sp = CubicSpline2D(x, y)
    >>> s = np.arange(0, sp.s[-1], ds)
    >>> rx, ry, ryaw, rk = [], [], [], []
    >>> for i_s in s:
    ...     ix, iy = sp.calc_position(i_s)
    ...     rx.append(ix)
    ...     ry.append(iy)
    ...     ryaw.append(sp.calc_yaw(i_s))
    ...     rk.append(sp.calc_curvature(i_s))
    >>> plt.subplots(1)
    >>> plt.plot(x, y, "xb", label="Data points")
    >>> plt.plot(rx, ry, "-r", label="Cubic spline path")
    >>> plt.grid(True)
    >>> plt.axis("equal")
    >>> plt.xlabel("x[m]")
    >>> plt.ylabel("y[m]")
    >>> plt.legend()
    >>> plt.show()

    .. image:: cubic_spline_2d_path.png

    >>> plt.subplots(1)
    >>> plt.plot(s, [np.rad2deg(iyaw) for iyaw in ryaw], "-r", label="yaw")
    >>> plt.grid(True)
    >>> plt.legend()
    >>> plt.xlabel("line length[m]")
    >>> plt.ylabel("yaw angle[deg]")

    .. image:: cubic_spline_2d_yaw.png

    >>> plt.subplots(1)
    >>> plt.plot(s, rk, "-r", label="curvature")
    >>> plt.grid(True)
    >>> plt.legend()
    >>> plt.xlabel("line length[m]")
    >>> plt.ylabel("curvature [1/m]")

    .. image:: cubic_spline_2d_curvature.png
    """

    def __init__(self, x, y):
        self.s = self.__calc_s(x, y)
        self.sx = CubicSpline1D(self.s, x)
        self.sy = CubicSpline1D(self.s, y)

    def __calc_s(self, x, y):
        # 各个点的s坐标
        dx = np.diff(x)
        dy = np.diff(y)
        self.ds = np.hypot(dx, dy)
        s = [0]
        s.extend(np.cumsum(self.ds))
        return s

    def calc_position(self, s):
        """
        calc position

        Parameters
        ----------
        s : float
            distance from the start point. if `s` is outside the data point's
            range, return None.

        Returns
        -------
        x : float
            x position for given s.
        y : float
            y position for given s.
        """
        x = self.sx.calc_position(s)
        y = self.sy.calc_position(s)

        return x, y

    def calc_curvature(self, s):
        """
        calc curvature

        Parameters
        ----------
        s : float
            distance from the start point. if `s` is outside the data point's
            range, return None.

        Returns
        -------
        k : float
            curvature for given s.
        """
        dx = self.sx.calc_first_derivative(s)
        ddx = self.sx.calc_second_derivative(s)
        dy = self.sy.calc_first_derivative(s)
        ddy = self.sy.calc_second_derivative(s)
        k = (ddy * dx - ddx * dy) / ((dx ** 2 + dy ** 2) ** (3 / 2))
        return k

    def calc_yaw(self, s):
        """
        calc yaw

        Parameters
        ----------
        s : float
            distance from the start point. if `s` is outside the data point's
            range, return None.

        Returns
        -------
        yaw : float
            yaw angle (tangent vector) for given s.
        """
        dx = self.sx.calc_first_derivative(s)
        dy = self.sy.calc_first_derivative(s)
        yaw = math.atan2(dy, dx)
        return yaw

        # return yaw * 180 / math.pi

"""
change

"""
def calc_spline_course(x, y, ds= 0.5):
    sp = CubicSpline2D(x, y)
    s = list(np.arange(0, sp.s[-1], ds))

    rx, ry, ryaw, rk = [], [], [], []
    for i_s in s:
        ix, iy = sp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        ryaw.append(sp.calc_yaw(i_s))
        rk.append(sp.calc_curvature(i_s))

    return rx, ry, ryaw, rk, s


import numpy as np
import matplotlib.pyplot as plt
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))



L = 2.347  # [m] Wheel base of vehicle
max_steer = np.radians(30.0)  # [rad] max steering angle

show_animation = True
proportional_part = 0
integral_part = 0


class State(object):
    """
    Class representing the state of a vehicle.

    :param x: (float) x-coordinate
    :param y: (float) y-coordinate
    :param yaw: (float) yaw angle
    :param v: (float) speed
    """

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        """Instantiate the object."""
        super(State, self).__init__()
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v

    def update(self, acceleration, delta):
        """
        Update the state of the vehicle.

        Stanley Control uses bicycle model.

        :param acceleration: (float) Acceleration
        :param delta: (float) Steering
        """
        delta = np.clip(delta, -max_steer, max_steer)

        self.x += self.v * np.cos(self.yaw) * dt
        self.y += self.v * np.sin(self.yaw) * dt
        self.yaw += self.v / L * np.tan(delta) * dt
        self.yaw = normalize_angle(self.yaw)
        self.v += acceleration * dt


# def pid_control(target, current):
#     """
#     Proportional control for the speed.
#
#     :param target: (float)
#     :param current: (float)
#     :return: (float)
#     """
#     error = target - current
#     error_sum = error * Kp + error * dt * Ki
#
#     return error_sum

"""
yao-yuan更新


"""
Kp = 5.0  # speed proportional gain
Ki = 0.8
Kd = 2.5
dT = 0.25
dt = 0.01  # [s] time difference
def pid_control(target, current, error_last):
    """
    Proportional control for the speed.

    :param target: (float)
    :param current: (float)
    :return: (float)
    """
    error = target - current
    error_sum = error * Kp + error * dt * Ki + Kd * (error - error_last) / dT

    return error_sum

def get_map_yaw(ref_x, ref_y):
    diff_x = np.diff(ref_x)
    diff_y = np.diff(ref_y)
    diff_x.append(diff_x[-1])
    diff_y.append(diff_y[-1])
    yaw = np.arctan2(diff_y, diff_x)


def get_error_yaw(state, cx, cy,cyaw):
    current_target_idx, error_front_axle = calc_target_index(state, cx, cy)
    theta_e = normalize_angle(cyaw[current_target_idx] - state.yaw)
    error_yaw = state.yaw - cyaw[current_target_idx]
    return  error_yaw

"""
stanly
"""
k = 1.6  # control gain
k_theta = 1.13
k_d = 3
k_curvature = 40
def deg_to_rad(yaw_deg):
    return  yaw_deg / 180 * math.pi
def stanley_control(state, cx, cy, cyaw, last_target_idx, lane_current_curvature, is_at_solid_line, error_d):
    """

    :param state:
    :param cx:
    :param cy:
    :param cyaw:
    :param last_target_idx:
    :param lane_current_curvature:
    :param is_at_solid_line:
    :param error_d:
    :return:
    """
    current_target_idx, error_front_axle = calc_target_index(state, cx, cy)

    # print("current_target_idx", current_target_idx)
    # if last_target_idx >= current_target_idx:
    #     current_target_idx = last_target_idx

    #1.  theta_e corrects the heading error
    theta_e = normalize_angle(cyaw[current_target_idx] - state.yaw)
    
    theta_d = np.arctan2(k * error_front_axle, state.v)
    if error_d > 1.2:
        theta_d *= 4
    elif error_d >0.8:
        theta_d *= 3


    if theta_d > math.pi:
        theta_d -= 2 * math.pi
    elif theta_d < -math.pi:
        theta_d += 2 * math.pi
    k_theta_ = k_theta
    if theta_e > 0.2:
        k_theta_ += 0.3
    elif theta_e > 0.3:
        k_theta_ += 0.4
    elif theta_e > 0.4:
        k_theta_ += 0.5

    average_curvature = 0
    for i in range(len(lane_current_curvature)):
        average_curvature += lane_current_curvature[i]
    average_curvature /= 4
    if average_curvature < 0.004:
        k_curvature = 40
    elif average_curvature < 0.045:
        k_curvature = 50
    else:
        k_curvature = 65
    delta = k_theta_ * theta_e + k_d * theta_d + k_curvature * lane_current_curvature[0]
    # max--0.005


    if delta > math.pi:
        delta -= 2 * math.pi
    elif delta < -math.pi:
        delta += 2 * math.pi

    if not is_at_solid_line:
        if delta >= deg_to_rad(30):
            delta = deg_to_rad(30)
        elif delta <= -deg_to_rad(30):
            delta = -deg_to_rad(30)

    # delta = theta_e + theta_d
    # delat_out = pid_control(delta, steer)
    return delta, current_target_idx, theta_e


def normalize_angle(angle):
    """
    Normalize an angle to [-pi, pi].

    :param angle: (float)
    :return: (float) Angle in radian in [-pi, pi]
    """
    while angle > np.pi:
        angle -= 2.0 * np.pi

    while angle < -np.pi:
        angle += 2.0 * np.pi

    return angle
def calc_target_index(state, cx, cy):
    """
    Compute index in the trajectory list of the target.在目标的轨迹列表中计算索引。

    :param state: (State object)
    :param cx: [float]
    :param cy: [float]
    :return: (int, float)
    """
    # Calc front axle position
    fx = state.x + L * np.cos(state.yaw)
    fy = state.y + L * np.sin(state.yaw)

    # Search nearest point index
    dx = [fx - icx for icx in cx]
    dy = [fy - icy for icy in cy]
    d = np.hypot(dx, dy)
    target_idx = np.argmin(d)

    # Project RMS error onto front axle vector将RMS误差投影到前轴矢量上
    front_axle_vec = [-np.cos(state.yaw + np.pi / 2),
                      -np.sin(state.yaw + np.pi / 2)]
    error_front_axle = np.dot([dx[target_idx], dy[target_idx]], front_axle_vec)

    return target_idx, error_front_axle


def distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))


def mapxy2frenet(maps_x, maps_y):
    """

    :param maps_x:
    :param maps_y:
    :return: maps_d, maps_s
    """
    maps_s = []
    maps_d = []
    dist_s = 0  # 相邻点距离
    dist_s_sum = 0
    pred_index = 0
    for i in range(len(maps_x)):
        maps_d.append(0)
        dist_s = math.sqrt((maps_x[i] - maps_x[pred_index]) * (maps_x[i] - maps_x[pred_index])
                           + (maps_y[i] - maps_y[pred_index]) * (maps_y[i] - maps_y[pred_index]))
        dist_s_sum += dist_s
        maps_s.append(dist_s_sum)
        pred_index = i
    return maps_s, maps_d


def getXY(s, d, maps_s, maps_x, maps_y):

    # # 比自车s小的最近点
    prev_wp = -1
    for i in range(len(maps_s)):
        if maps_s[i] > s:
            prev_wp = i - 1
            break
    # print("in",maps_s[prev_wp], maps_s[prev_wp+1])


    # print("-------1")
    # 比自车s大的最近点
    wp2 = prev_wp + 1
    # print("getXY", prev_wp, "wp2", wp2)

    heading = math.atan2(maps_y[wp2] - maps_y[prev_wp], maps_x[wp2] - maps_x[prev_wp])
    seg_s = (s - maps_s[prev_wp])
    seg_x = maps_x[prev_wp] + seg_s * math.cos(heading)
    seg_y = maps_y[prev_wp] + seg_s * math.sin(heading)

    if d < 0:
        d = -d
        perp_heading = heading - math.pi / 2
        x = seg_x + d * math.cos(perp_heading)
        y = seg_y + d * math.sin(perp_heading)
    else:
        d = d
        perp_heading = heading + math.pi / 2
        x = seg_x + d * math.cos(perp_heading)
        y = seg_y + d * math.sin(perp_heading)

    return x, y


def get_closed_point(state, maps_x, maps_y):
    # Calc front axle positio
    fx = state.x
    fy = state.y

    # Search nearest point index
    dx = [fx - icx for icx in maps_x]
    dy = [fy - icy for icy in maps_y]
    d = np.hypot(dx, dy)
    target_idx = np.argmin(d)
    return target_idx


def getFrenet(state, maps_x, maps_y, maps_s):
    """

    :param state:   class---自车状态
    :param maps_x:
    :param maps_y:
    :param maps_s:
    :return:
    """
    closest_wp_index = get_closed_point(state, maps_x, maps_y)
    # print("closest_wp_index", closest_wp_index)
    x = state.x
    y = state.y
    prev_wp_index = closest_wp_index - 1
    if closest_wp_index == 0:
        prev_wp_index = len(maps_x) - 1

    n_x = maps_x[closest_wp_index] - maps_x[prev_wp_index]
    n_y = maps_y[closest_wp_index] - maps_y[prev_wp_index]
    x_x = x - maps_x[prev_wp_index]
    x_y = y - maps_y[prev_wp_index]

    proj_norm = (x_x * n_x + x_y * n_y) / (n_x * n_x + n_y * n_y)
    # print("proj_norm!!!!!!!!!!!!!!!!!" , proj_norm)
    proj_x = proj_norm * n_x
    proj_y = proj_norm * n_y
    frenet_d = distance(x_x, x_y, proj_x, proj_y)
    """
    定义：平面上的三点P1(x1, y1), P2(x2, y2), P3(x3, y3)
    的面积量：
    S(P1, P2, P3) = | y1
    y2
    y3 |= (x1 - x3) * (y2 - y3) - (y1 - y3) * (x2 - x3)

    当P1P2P3逆时针时S为正的，当P1P2P3顺时针时S为负的。

    令矢量的起点为A，终点为B，判断的点为C，
    如果S（A，B，C）为正数，则C在矢量AB的左侧；
    如果S（A，B，C）为负数，则C在矢量AB的右侧；
    如果S（A，B，C）为0，则C在直线AB上。
    """
    judge_rl = (maps_x[prev_wp_index] - x) * (maps_y[closest_wp_index] - y) - (maps_y[prev_wp_index] - y) * (
                maps_x[closest_wp_index] - x)
    # center_x = 100000000 - maps_x[prev_wp_index]
    # center_y = 200000000 - maps_y[prev_wp_index]
    # centerToPos = distance(center_x, center_y, x_x, x_y)
    # centerToRef = distance(center_x, center_y, proj_x, proj_y)
    # if centerToPos <= centerToRef:
    #     frenet_d = -1 * frenet_d
    if judge_rl < 0:
        frenet_d = -frenet_d
    # print("prev_wp_index---------------------------------------------", prev_wp_index)
    frenet_s = maps_s[0]
    for i in range(prev_wp_index):
        frenet_s += distance(maps_x[i], maps_y[i], maps_x[i + 1], maps_y[i + 1])
    frenet_s += distance(0, 0, proj_x, proj_y)
    return frenet_s, frenet_d


def get_ref_yaw(ref_x, ref_y):
    _diff_x = list(np.diff(ref_x))
    _diff_y = list(np.diff(ref_y))
    _diff_x.append(_diff_x[-1])
    _diff_y.append(_diff_y[-1])

    _ref_yaw = np.arctan2(_diff_y, _diff_x)
    return _ref_yaw


def vehilce_to_world(state, xs_in_vehicle, ys_in_vehicle):
    theta = state.yaw
    xs_in_world = []
    ys_in_world = []
    if theta < -math.pi:
        theta += 2 * math.pi
    if theta > math.pi:
        theta -= 2 * math.pi
    for i in range(len(xs_in_vehicle)):
        xs_in_world.append(state.x + xs_in_vehicle[i] * math.cos(theta) - ys_in_vehicle[i] * math.sin(theta))
        ys_in_world.append(state.y + xs_in_vehicle[i] * math.sin(theta) + ys_in_vehicle[i] * math.cos(theta))
    return xs_in_world, ys_in_world
def Map_Process(map_csv):
    maps_x_ = []
    maps_y_ = []
    maps_yaw_ = []
    with open(map_csv) as f:
        reader = csv.reader(f)
        header_row = next(reader)
        print(header_row)
        for row in reader:
            maps_x_.append(float(row[1]))
            maps_y_.append(float(row[2]))
            maps_yaw_.append(float(row[3]))
    return maps_x_, maps_y_, maps_yaw_


def linear_process(current, target, x, T = 30):
    """

    :param current: 当前参量
    :param target: 目标参量
    :param x: 插值【0，T】内
    :param T: 跨度
    :return out : x处的输出值----T时为target
    """

    k = (target - current) / T

    out = k * x + current
    return out




x_uni = np.linspace(20, 120, 6)
y_uni = [0.1240, 0.1145, 0.1205, 0.1475, 0.1745, 0.2195]
x_th, y_th = np.mgrid[0:1:11j, 0:120:7j]
z_th = [[1.9730, -0.7466, -0.3934, -0.3420, -0.4822, -0.6413, -0.8234],
        [2.1454, -0.5067, -0.2101, -0.2111, -0.3834, -0.5642, -0.7601],
        [3.6572,  1.8343,  1.4560,  0.9574,  0.5727,  0.2392, -0.0820],
        [3.7796,  2.9935,  2.1136,  1.3881,  1.0263,  0.6868,  0.3288],
        [3.8177,  3.8622,  2.7008,  1.7739,  1.4233,  1.0673,  0.7387],
        [3.8177,  4.2992,  3.1524,  2.3269,  1.7783,  1.4070,  1.1328],
        [3.8177,  4.5076,  3.3241,  2.4839,  1.9103,  1.5453,  1.3129],
        [3.8177,  4.6816,  4.1868,  3.0012,  2.2289,  1.6911,  1.5128],
        [3.8177,  4.7295,  4.2352,  3.0626,  2.8261,  2.1037,  1.5675],
        [3.8177,  4.7696,  4.2537,  3.1145,  2.9018,  2.1636,  1.6130],
        [3.8177,  4.7892,  4.2542,  3.1507,  2.9570,  2.2059,  1.6462],
        ] # 节气门标定表
x_br, y_br = np.mgrid[0:15:16j, 0:120:7j]
z_br = [[ -0.0503,  -0.7466,  -0.3934,  -0.3420,  -0.4822,  -0.6413,  -0.8234],
        [ -1.5165,  -2.1053,  -2.3524,  -2.4023,  -2.5396,  -2.6935,  -2.8686],
        [ -2.9879,  -3.4409,  -4.3118,  -4.4455,  -4.5778,  -4.7261,  -4.8948],
        [ -4.1143,  -4.3511,  -5.7410,  -6.2246,  -6.0613,  -6.1647,  -6.3235],
        [ -5.2906,  -5.2770,  -7.1214,  -7.4565,  -7.5685,  -7.6996,  -7.8637],
        [ -6.6704,  -6.2952,  -8.6877,  -8.9886,  -9.1028,  -9.2361,  -9.3857],
        [ -8.0940,  -7.3411, -10.1128, -10.4433, -10.5372, -10.6693, -10.8235],
        [ -9.4772,  -8.1753, -10.1739, -10.4433, -10.5372, -10.6693, -10.8235],
        [-10.6571,  -8.9192, -10.1764, -10.4433, -10.5372, -10.6693, -10.8235],
        [-10.6571,  -9.2036, -10.1764, -10.4433, -10.5372, -10.6693, -10.8235],
        [-10.6571,  -9.6282, -10.1764, -10.4433, -10.5372, -10.6693, -10.8235],
        [-10.6571,  -9.8832, -10.1764, -10.4433, -10.5372, -10.6693, -10.8235],
        [-10.6571, -10.0302, -10.1764, -10.4433, -10.5372, -10.6693, -10.8235],
        [-10.6571, -10.0302, -10.1764, -10.4433, -10.5372, -10.6693, -10.8235],
        [-10.6571, -10.0302, -10.1764, -10.4433, -10.5372, -10.6693, -10.8235],
        [-10.6571, -10.0302, -10.1764, -10.4433, -10.5372, -10.6693, -10.8235],
        ] # 刹车主缸压力标定表
# 车速保持
f_uni = interpolate.UnivariateSpline(x_uni, y_uni, s=0) # s=0强制通过所有点
# 节气门开度
f_th = interpolate.interp2d(x_th, y_th, z_th, kind='linear') # 高次多项式效果不理想，线性拟合kind=linear
# 刹车主缸压力
f_br = interpolate.interp2d(x_br, y_br, z_br, kind='linear')