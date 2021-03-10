import numpy as np

# Modified from https://github.com/diegoroyo/flowvid/blob/master/flowvid/core/util/color_flow.py


def _make_color_wheel():
    """ :returns: [n_cols, 3] ndarray color_wheel """
    # how many hues ("cols") separate each color
    # (for this color wheel)
    RY = 15  # red-yellow
    YG = 6   # yellow-green
    GC = 4   # green-cyan
    CB = 11  # cyan-blue
    BM = 13  # blue-magenta
    MR = 6   # magenta-red
    n_cols = RY + YG + GC + CB + BM + MR
    color_wheel = np.zeros((n_cols, 3), dtype=np.uint8)  # r g b

    col = 0
    # RY
    color_wheel[col:col+RY, 0] = 255
    color_wheel[col:col+RY, 1] = np.floor(255*np.arange(RY)/RY)
    col = col + RY
    # YG
    color_wheel[col:col+YG, 0] = np.ceil(255*np.arange(YG, 0, -1)/YG)
    color_wheel[col:col+YG, 1] = 255
    col = col + YG
    # GC
    color_wheel[col:col+GC, 1] = 255
    color_wheel[col:col+GC, 2] = np.floor(255*np.arange(GC)/GC)
    col = col + GC
    # CB
    color_wheel[col:col+CB, 1] = np.ceil(255*np.arange(CB, 0, -1)/CB)
    color_wheel[col:col+CB, 2] = 255
    col = col + CB
    # BM
    color_wheel[col:col+BM, 2] = 255
    color_wheel[col:col+BM, 0] = np.floor(255*np.arange(BM)/BM)
    col = col + BM
    # MR
    color_wheel[col:col+MR, 2] = np.ceil(255*np.arange(MR, 0, -1)/MR)
    color_wheel[col:col+MR, 0] = 255

    return color_wheel


_color_wheel = _make_color_wheel()


def flow_to_rgb(flo_data, rad_normaliser=None):
    """
        :param flo_data: [2, h, w] ndarray (flow data)
        :returns: [h, w, 3] ndarray (rgb data) using color wheel
    """
    n_cols = len(_color_wheel)

    flo_data = flo_data.numpy()

    fu = flo_data[0, :, :]
    fv = flo_data[1, :, :]

    [h, w] = fu.shape
    rgb_data = np.empty([h, w, 3], dtype=np.uint8)

    rad = np.sqrt(fu ** 2 + fv ** 2)
    if rad_normaliser is None:
        rad_normaliser = np.max(rad)
        rad /= rad_normaliser
        fv = fv / rad_normaliser
        fu = fu / rad_normaliser
    else:
        rad /= rad_normaliser
        rad = np.clip(rad, 0.0, 1.0)
        fv = fv / rad_normaliser
        rad = np.clip(rad, -1.0, 1.0)
        fu = fu / rad_normaliser
        rad = np.clip(rad, -1.0, 1.0)

    a = np.arctan2(-fv, -fu) / np.pi
    fk = (a + 1) / 2 * (n_cols - 1)  # -1~1 mapped to 1~n_cols
    k0 = fk.astype(np.uint8)
    k1 = (k0 + 1) % n_cols
    f = fk - k0
    for i in range(3):  # r g b
        col0 = _color_wheel[k0, i]/255.0
        col1 = _color_wheel[k1, i]/255.0
        col = np.multiply(1.0-f, col0) + np.multiply(f, col1)

        # increase saturation with radius
        col = 1.0 - np.multiply(rad, 1.0 - col)

        # save to data channel i
        rgb_data[:, :, i] = np.floor(col * 255).astype(np.uint8)

    return rgb_data
