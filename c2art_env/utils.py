import numpy as np
from numba import jit, njit, prange


@njit(nogil=True)
def clip_min_max(x: float, a_min: float, a_max: float) -> np.ndarray:
    # return np.clip(x, a_min, a_max)
    return min(a_max, max(a_min, x))


@njit(nogil=True)
def wrap_to_pi(x):
    # Wrap [rad] angle to [-PI..PI), https://stackoverflow.com/questions/4633177/c-how-to-wrap-a-float-to-the-interval-pi-pi
    return ((x + np.pi) % (2 * np.pi)) - np.pi


@njit(nogil=True)
def not_zero(x: float, eps: float = 1e-3) -> float:
    if abs(x) > eps:
        return x
    elif x > 0:
        return eps
    else:
        return -eps


@njit(nogil=True)
def not_zero_array(x: np.array, eps: float = 1e-3) -> np.array:
    for i in range(len(x)):
        if abs(x[i]) > eps:
            pass
        elif x[i] >= 0:
            x[i] = eps
        else:
            x[i] = -eps
    return x


@njit(nogil=True)
def binary_search(a, x, lo=0):

    hi = len(a)
    while lo < hi:
        mid = (lo+hi)//2
        midval = a[mid]
        if midval < x:
            lo = mid+1
        elif midval > x:
            hi = mid
        else:
            return mid
    return mid


@njit(nogil=True)
def interp_binary(X, Y, x):
    ind = binary_search(X, x)
    if ind >= len(X) - 1:
        return Y[len(Y) - 1]
    x1 = X[ind]
    x2 = X[ind+1]
    y1 = Y[ind]
    y2 = Y[ind+1]
    y = y1 + ((y2 - y1) / (x2 - x1)) * (x - x1)

    return y


@njit(nogil=True)
def interp_grid_slope(X, Y, x):
    if x < X[1]:
        y = Y[1]
    elif x > X[-2]:
        y = Y[-2]
    else:
        i = int((x + 2.5) / 5)
        y = Y[i] + (x - X[i]) * ((Y[i+1] - Y[i]) / (X[i+1] - X[i]))

    return y


@njit(nogil=True)
def motion_integ(
    pos: float,
    speed: float,
    accel_next: float,
    dt: float) -> np.ndarray:

    speed_next = max(0, speed + accel_next * dt)
    accel_next = (speed_next - speed) / dt
    pos_next = pos + (speed + speed_next)/2 * dt

    return np.array([pos_next, speed_next, accel_next])


@njit(nogil=True)
def my_round(num: float, dec: int):
    p = float(10**dec)
    num_r = int(num * p + 0.5)/p
    return num_r


@njit(nogil=True)
def numbadiff(x):
    return x[1:] - x[:-1]


@njit(nogil=True)
def xy2yaw(x, y):
    yaw = np.zeros_like(x)

    for i in range(len(x)):
        if i == 0:
            pass
        elif i == len(x)-1 or x[i+1] - x[i-1] == 0:
            yaw[i] = yaw[i-1]
        else:
            yaw[i] = np.arctan2(y[i+1] - y[i-1], x[i+1] - x[i-1])

    yaw[0] = yaw[1]

    return yaw


@njit(nogil=True)
def xy2lon(x, y, lon_init=0):
    lon = np.zeros_like(x)
    lon[0] = lon_init
    for i in range(1, len(x)):
        lon[i] = lon[i-1] + np.sqrt((x[i]-x[i-1])**2 + (y[i]-y[i-1])**2)

    return lon


@njit(nogil=True)
def xy2curvature(x, y):
    xy = np.vstack((x, y)).transpose()
    cur = np.zeros_like(x)
    for i in range(1, len(x)-1):
        p1_ = xy[i-1]
        p2_ = xy[i]
        p3_ = xy[i+1]
        A_ = ((p2_[0]-p1_[0]) * (p3_[1]-p1_[1]) - (p2_[1]-p1_[1]) * (p3_[0]-p1_[0])) / 2
        cur[i] = 4 * A_ / (numba_norm2(p1_-p2_) * numba_norm2(p2_-p3_) * numba_norm2(p3_-p1_))

    cur[0] = cur[1]
    cur[-1] = cur[-2]

    return cur


@njit
def numba_norm2(x):
    sq = 0
    for i in range(len(x)):
        sq = sq + x[i]**2
    root = np.sqrt(sq)

    return root


@njit
def numba_norm2_min(x):
    norm2 = np.zeros(shape=(len(x),1))
    for i in range(len(x)):
        norm2[i] = np.sqrt(x[i][0] ** 2 + x[i][1] ** 2)
    return norm2[np.argmin(norm2)], np.argmin(norm2)
 

@njit
def numba_wrap_to_pi(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi


@njit
def min_max_normalize(x):
    return (x-min(x))/(max(x)-min(x))

# @jit(parallel=True)
# def mean_numba(a):
#     if a.shape[0] == 1:
#         res = a.mean()
#     elif a.shape[0] > 1:
#         res = np.empty(a.shape[0])
#         for i in prange(a.shape[0]):
#             res[i] = a[i, :].mean()

#     return res

# '''
# if __name__ == '__main__':

    # X = np.array([1,2,3,4,5])
    # Y = np.array([2,3,6,8,12])
    # x = np.nan
    # print(interp_grid_slope(X,Y, x))
    # print(clip_min_max(10, 1, 8))
    # print(motion_integ(1, 10, 5, 0.1))
    # print(my_round(1.123456415, 5))
    # import timeit
    # start = timeit.default_timer()
    # for i in range(1000000):
        # a = interp_binary(X,Y, x)
        # a = clip_min_max(0.5, 1, 8)
        # a = motion_integ(1, 10, 5, 0.1)

    # stop = timeit.default_timer()    
    # print('Time: ', stop - start) 
# '''
