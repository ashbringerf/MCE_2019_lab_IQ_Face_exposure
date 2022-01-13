""" frequent color space conversions  """
import numpy as np

def sRGB2RGB(sRGB):
    """
    convert sRGB 2 linear RGB
    :param sRGB: array-like
    :return: array-like
    """
    RGB = np.array([0.0, 0.0, 0.0])
    th = 0.04045
    for c in range(3):
        if sRGB[c] <= th:
            RGB[c] = sRGB[c] / 12.92
        else:
            RGB[c] = ((sRGB[c] + 0.055) / 1.055) ** 2.4

    return RGB

def RGB2sRGB(RGB):
    """
    convert linear RGB 2 sRGB
    :param sRGB: array-like
    :return: array-like
    """
    sRGB = np.array([0.0, 0.0, 0.0])
    th = 0.0031308
    for c in range(2):
        if RGB[c] <= th:
            sRGB[c] = RGB[c] * 12.92
        else:
            sRGB[c] = 1.055 * (RGB[c]) ** (1 / 2.4) - 0.055
    return sRGB


def xy2CCT(x, y):
    """
    convert xy color coordinates to correlated color temperature
    :param x, y: array-like
    :return: a value
    """
    n = (x - 0.332) / (0.1858 - y)
    CCT = 449 * n ** 3 + 3525 * n ** 2 + 6823.3 * n + 5520.33
    return CCT


def xyY2XYZ(x, y, Y):
    """
    convert xy color coordinates to XYZ
    :param x, y, Y: array-like
    :return: array like
    """
    X = x * Y / y
    Y = Y
    Z = (1 - x - y) * Y / y
    return XYZ


def CCT2xy_(CCT):
    """ http://www.brucelindbloom.com/ """
    """
    convert correlated color temperature to xy color coordinates only accurate when CCT < 4000
    :param CCT: array-like
    :return: array like
    """
    if CCT <= 7000:
        x = (-4.607 * 10 ** 9) / (CCT ** 3) + (2.9678 * 10 ** 6) / (CCT ** 2) + (0.09911 * 10 ** 3) / (CCT) + 0.244063
    else:
        x = (-2.00064 * 10 ** 9) / (CCT ** 3) + (1.9018 * 10 ** 6) / (CCT ** 2) + (0.24748 * 10 ** 3) / (CCT) + 0.237040
    y = -3 * x ** 2 + 2.870 * x - 0.275
    return [x, y, 1.0]

def CCT2xy(CCT):
    """ https://en.wikipedia.org/wiki/Planckian_locus#Approximation/ """
    """ the CCT should be larger than 1667K
    :param CCT: array-like
    :return: array like
    """
    if CCT <= 4000:
        x = (-0.2661239 * (10 ** 9)) / (CCT ** 3) - (0.2343589 * (10 ** 6)) / (CCT ** 2) + (0.8776956 * (10 ** 3)) / (CCT) + 0.17991
    else:
        x = (-3.0258469 * (10 ** 9)) / (CCT ** 3) + (2.1070379* (10 ** 6)) / (CCT ** 2) + (0.2226347 * (10 ** 3)) / (CCT) + 0.24039
        
    if CCT <= 2222:
        y = -1.1063814 * (x ** 3) - 1.34811020 * (x ** 2) + 2.18555832 * x - 0.20219683 
    elif CCT<=4000 and CCT>2222:
        y = -0.9549476 * (x ** 3) - 1.37418593 * (x ** 2) + 2.09137015 * x - 0.16748867 
    else:
        y = 3.0817580 * (x ** 3) - 5.87338670 * (x ** 2) + 3.75112997 * x - 0.37001483 

    return [x, y, 1.0]


def xyY2XYZ(xyY):
    """
    convert xyY to XYZ
    :param xyY: array-like
    :return: array like
    """
    X = xyY[0] * xyY[2] / xyY[1]
    Y = xyY[2]
    Z = ((1 - xyY[0] - xyY[1]) * xyY[2]) / xyY[1]

    return [X, Y, Z]


def XYZ2RGB(XYZ):
    """
    convert XYZ 2 RGB
    :param XYZ: array-like
    :return: array like
    """
    XYZ_to_RGB_matrix = np.array(
        [[3.2404542, -1.5371385, -0.4985314],
         [-0.9692660, 1.8760108, 0.0415560],
         [0.0556434, -0.2040259, 1.0572252]])
    RGB = np.matmul(XYZ_to_RGB_matrix, XYZ)
    
    """ RGB <=1 & RGB >=0 """
    
    RGB = np.where(RGB > 0, RGB, 0)
    RGB = np.where(RGB < 1, RGB, 1)

    return RGB


def RGB2XYZ(RGB):
    """
    convert RGB 2 XYZ
    :param RGB: array-like
    :return: array like
    """
    RGB_to_XYZ_matrix = np.array(
        [[0.4124564, 0.3575761, 0.1804375],
         [0.2126729, 0.7151522, 0.0721750],
         [0.0193339, 0.1191920, 0.9503041]])
    XYZ = np.matmul(RGB_to_XYZ_matrix, RGB)
    return XYZ


def RGB_normalization(RGB):
    """
    0-255 RGB value normalize to 0-1
    :param RGB: array-like
    :return: array like
    """
    RGB = RGB / 255.0
    return RGB


def XYZ2Lab(XYZ):
    """
    convert XYZ 2 Lab
    :param XYZ: array-like
    :return: array like
    """
    """ precision need to be np.float """
    XYZ = np.array(XYZ).astype(np.float64) * 100
    """ D65 white """
    white = np.array([95.047, 100.00, 108.883])
    Lab = np.array([0.0, 0.0, 0.0])
    fx = np.array([0.0, 0.0, 0.0])
    for c in range(3):
        r_white = 0
        if (XYZ[c] / white[c]) > ((6.0 / 29.0) ** 3.0):
            r_white = 1
        fx[c] = fx[c] + r_white * (XYZ[c] / white[c]) ** (1 / 3.0)
        fx[c] = fx[c] + (1 - r_white) * ((841.0 / 108.0) * XYZ[c] / white[c] + 4.0 / 29.0)
    Lab[0] = 116 * fx[1] - 16
    Lab[1] = 500 * (fx[0] - fx[1])
    Lab[2] = 200 * (fx[1] - fx[2])
    return Lab


def sRGB2Lab(sRGB):
    """
    convert sRGB 2 Lab
    :param sRGB: array-like
    :return: array like
    """
    sRGB_normalized = RGB_normalization(sRGB)
    RGB = sRGB2RGB(sRGB_normalized)
    #RGB=sRGB_normalized
    XYZ = RGB2XYZ(RGB)
    Lab = XYZ2Lab(XYZ)
    return Lab


def tsplit(a):
    """
    Splits arrays in sequence along the last axis (tail).

    :param a: array-like
    :return: ndarray
    """
    a = np.array(a)
    return np.array([a[..., x] for x in range(a.shape[-1])])


def tstack(a):
    """
    Stacks arrays in squence along the last axis (tail).

    :param a: array-like
    :return: ndarray
    """
    a = np.array(a)
    return np.concatenate([x[..., np.newaxis] for x in a], axis=-1)


def Lab2LCh(Lab):
    """
    Transform colorspace from Lab to LCh.

    :param Lab: array-like
    :return:  ndarray
    """
    [L, a, b] = tsplit(Lab)
    C = np.hypot(a, b)
    h = np.rad2deg(np.arctan2(b, a))
    LCh = tstack([L, C, h])
    return LCh


def LCh2Lab(LCh):
    """
    Transform colorspace from LCh to Lab.

    :param LCh: array-like
    :return: ndarray
    """
    [L, C, h] = tsplit(LCh)
    a = C * np.cos(np.deg2rad(h))
    b = C * np.sin(np.deg2rad(h))
    Lab = tstack([L, a, b])
    return Lab


def cal_chroma(Lab_1, Lab_2):
    """
    Calculate the :math:`\Delta C` between Lab_1 and Lab_2.

    :math:`\\Delta C = \sqrt{a_1^2+b_1^2} - \sqrt{a_2^2+b_2^2}`.

    :param Lab_1: array-like
    :param Lab_2: array-like
    :return: numeric or ndarray
    """
    [_, a1, b1] = tsplit(Lab_1)
    [_, a2, b2] = tsplit(Lab_2)
    C1 = np.hypot(a1, b1)
    C2 = np.hypot(a2, b2)
    chroma = C1 - C2
    return chroma


def cal_deltaAB(Lab_1, Lab_2):
    """
    Calculate the :math:`\Delta ab` between Lab_1 and Lab_2.

    :math:`\Delta ab = \sqrt{(a_1-a_2)^2+(b_1-b_2)^2}`.

    :param Lab_1: array-like
    :param Lab_2: array-like
    :return: numeric or ndarray
    """
    [_, a1, b1] = tsplit(Lab_1)
    [_, a2, b2] = tsplit(Lab_2)
    deltaAB = np.sqrt((a1 - a2) ** 2 + (b1 - b2) ** 2)
    return deltaAB


def delta_E_CIE2000(Lab_1, Lab_2, textiles=False):
    """
    Calculate the color difference :math:`\Delta E_{00}` between Lab_1 and Lab_2.

    :param Lab_1: array-like
    :param Lab_2: array-like
    :param textiles: bool
    :return: numeric or ndarray
    """
    L_1, a_1, b_1 = tsplit(Lab_1)
    L_2, a_2, b_2 = tsplit(Lab_2)

    k_L = 2 if textiles else 1
    k_C = 1
    k_H = 1

    l_bar_prime = 0.5 * (L_1 + L_2)

    c_1 = np.hypot(a_1, b_1)
    c_2 = np.hypot(a_2, b_2)

    c_bar = 0.5 * (c_1 + c_2)
    c_bar7 = c_bar ** 7

    g = 0.5 * (1 - np.sqrt(c_bar7 / (c_bar7 + 25 ** 7)))

    a_1_prime = a_1 * (1 + g)
    a_2_prime = a_2 * (1 + g)
    c_1_prime = np.hypot(a_1_prime, b_1)
    c_2_prime = np.hypot(a_2_prime, b_2)
    c_bar_prime = 0.5 * (c_1_prime + c_2_prime)

    h_1_prime = np.degrees(np.arctan2(b_1, a_1_prime)) % 360
    h_2_prime = np.degrees(np.arctan2(b_2, a_2_prime)) % 360

    h_bar_prime = np.where(
        np.fabs(h_1_prime - h_2_prime) <= 180,
        0.5 * (h_1_prime + h_2_prime),
        (0.5 * (h_1_prime + h_2_prime + 360)),
        )

    t = (1 - 0.17 * np.cos(np.deg2rad(h_bar_prime - 30)) +
         0.24 * np.cos(np.deg2rad(2 * h_bar_prime)) +
         0.32 * np.cos(np.deg2rad(3 * h_bar_prime + 6)) -
         0.20 * np.cos(np.deg2rad(4 * h_bar_prime - 63)))

    h = h_2_prime - h_1_prime
    delta_h_prime = np.where(h_2_prime <= h_1_prime, h - 360, h + 360)
    delta_h_prime = np.where(np.fabs(h) <= 180, h, delta_h_prime)

    delta_L_prime = L_2 - L_1
    delta_C_prime = c_2_prime - c_1_prime
    delta_H_prime = (2 * np.sqrt(c_1_prime * c_2_prime) * np.sin(
        np.deg2rad(0.5 * delta_h_prime)))

    s_L = 1 + ((0.015 * (l_bar_prime - 50) * (l_bar_prime - 50)) /
               np.sqrt(20 + (l_bar_prime - 50) * (l_bar_prime - 50)))
    s_C = 1 + 0.045 * c_bar_prime
    s_H = 1 + 0.015 * c_bar_prime * t

    delta_theta = (
            30 * np.exp(-((h_bar_prime - 275) / 25) * ((h_bar_prime - 275) / 25)))

    c_bar_prime7 = c_bar_prime ** 7

    r_C = np.sqrt(c_bar_prime7 / (c_bar_prime7 + 25 ** 7))
    r_T = -2 * r_C * np.sin(np.deg2rad(2 * delta_theta))

    d_E = np.sqrt((delta_L_prime / (k_L * s_L)) ** 2 +
                  (delta_C_prime / (k_C * s_C)) ** 2 +
                  (delta_H_prime / (k_H * s_H)) ** 2 +
                  (delta_C_prime / (k_C * s_C)) * (delta_H_prime /
                                                   (k_H * s_H)) * r_T)

    return d_E



def tsplit(a):
    a = np.array(a)
    return np.array([a[..., x] for x in range(a.shape[-1])])


def deltaE00(Lab_1, Lab_2, textiles = False):
    L_1, a_1, b_1 = tsplit(Lab_1)
    L_2, a_2, b_2 = tsplit(Lab_2)

    k_L = 2 if textiles else 1
    k_C = 1
    k_H = 1

    l_bar_prime = 0.5 * (L_1 + L_2)

    c_1 = np.hypot(a_1, b_1)
    c_2 = np.hypot(a_2, b_2)

    c_bar = 0.5 * (c_1 + c_2)
    c_bar7 = c_bar ** 7

    g = 0.5 * (1 - np.sqrt(c_bar7 / (c_bar7 + 25 ** 7)))

    a_1_prime = a_1 * (1 + g)
    a_2_prime = a_2 * (1 + g)
    c_1_prime = np.hypot(a_1_prime, b_1)
    c_2_prime = np.hypot(a_2_prime, b_2)
    c_bar_prime = 0.5 * (c_1_prime + c_2_prime)

    h_1_prime = np.degrees(np.arctan2(b_1, a_1_prime)) % 360
    h_2_prime = np.degrees(np.arctan2(b_2, a_2_prime)) % 360

    h_bar_prime = np.where(
        np.fabs(h_1_prime - h_2_prime) <= 180,
        0.5 * (h_1_prime + h_2_prime),
        (0.5 * (h_1_prime + h_2_prime + 360)),
    )

    t = (1 - 0.17 * np.cos(np.deg2rad(h_bar_prime - 30)) +
         0.24 * np.cos(np.deg2rad(2 * h_bar_prime)) +
         0.32 * np.cos(np.deg2rad(3 * h_bar_prime + 6)) -
         0.20 * np.cos(np.deg2rad(4 * h_bar_prime - 63)))

    h = h_2_prime - h_1_prime
    delta_h_prime = np.where(h_2_prime <= h_1_prime, h - 360, h + 360)
    delta_h_prime = np.where(np.fabs(h) <= 180, h, delta_h_prime)

    delta_L_prime = L_2 - L_1
    delta_C_prime = c_2_prime - c_1_prime
    delta_H_prime = (2 * np.sqrt(c_1_prime * c_2_prime) * np.sin(
        np.deg2rad(0.5 * delta_h_prime)))

    s_L = 1 + ((0.015 * (l_bar_prime - 50) * (l_bar_prime - 50)) /
               np.sqrt(20 + (l_bar_prime - 50) * (l_bar_prime - 50)))
    s_C = 1 + 0.045 * c_bar_prime
    s_H = 1 + 0.015 * c_bar_prime * t

    delta_theta = (
            30 * np.exp(-((h_bar_prime - 275) / 25) * ((h_bar_prime - 275) / 25)))

    c_bar_prime7 = c_bar_prime ** 7

    r_C = np.sqrt(c_bar_prime7 / (c_bar_prime7 + 25 ** 7))
    r_T = -2 * r_C * np.sin(np.deg2rad(2 * delta_theta))

    d_E = np.sqrt((delta_L_prime / (k_L * s_L)) ** 2 +
                  (delta_C_prime / (k_C * s_C)) ** 2 +
                  (delta_H_prime / (k_H * s_H)) ** 2 +
                  (delta_C_prime / (k_C * s_C)) * (delta_H_prime /
                                                   (k_H * s_H)) * r_T)

    return d_E


def spectrum2XYZ(spectrum):
    ## spectrum 2 xyz
    Light_Increment = 5
    ## CL500A measured wavelength from 360nm to 780nm; row 30 to 450
    ## load from 380 to 780; row 50 to 450
    #xlsx_name = r"Template_Spectral.xlsx"
    #relative_path = os.path.dirname(os.path.abspath(__file__))
    #excel_workbook = load_workbook(os.path.join(relative_path, xlsx_name))
    #start_row = 50
    #start_col = 'C'
    #excel_worksheet = excel_workbook['光谱值']
    #measured_lspd = []
    #for row in range(401):
    #    print('row index ' , start_row + row, excel_worksheet[start_col+str(start_row+row)].value)
    #    measured_lspd.append(excel_worksheet[start_col+str(start_row+row)].value)
    
    ## CIE color matching function
    ## 1931 2 degrees
    cmfs_wavelengths = []
    for cmfs_wavelength in range(380, 785, Light_Increment):
        cmfs_wavelengths.append(cmfs_wavelength)
    
    cmfs_X = [0.001368,0.002236,0.004243,0.007650,0.014310,0.023190,0.043510,0.077630,0.134380,0.214770,0.283900,0.328500,0.348280,0.348060,0.336200,0.318700,0.290800,0.251100,0.195360,0.142100,0.095640,0.057950,0.032010,0.014700,0.004900,0.002400,0.009300,0.029100,0.063270,0.109600,0.165500,0.225750,0.290400,0.359700,0.433450,0.512050,0.594500,0.678400,0.762100,0.842500,0.916300,0.978600,1.026300,1.056700,1.062200,1.045600,1.002600,0.938400,0.854450,0.751400,0.642400,0.541900,0.447900,0.360800,0.283500,0.218700,0.164900,0.121200,0.087400,0.063600,0.046770,0.032900,0.022700,0.015840,0.011359,0.008111,0.005790,0.004109,0.002899,0.002049,0.001440,0.001000,0.000690,0.000476,0.000332,0.000235,0.000166,0.000117,0.000083,0.000059,0.000042]
    cmfs_Y = [0.000039,0.000064,0.000120,0.000217,0.000396,0.000640,0.001210,0.002180,0.004000,0.007300,0.011600,0.016840,0.023000,0.029800,0.038000,0.048000,0.060000,0.073900,0.090980,0.112600,0.139020,0.169300,0.208020,0.258600,0.323000,0.407300,0.503000,0.608200,0.710000,0.793200,0.862000,0.914850,0.954000,0.980300,0.994950,1.000000,0.995000,0.978600,0.952000,0.915400,0.870000,0.816300,0.757000,0.694900,0.631000,0.566800,0.503000,0.441200,0.381000,0.321000,0.265000,0.217000,0.175000,0.138200,0.107000,0.081600,0.061000,0.044580,0.032000,0.023200,0.017000,0.011920,0.008210,0.005723,0.004102,0.002929,0.002091,0.001484,0.001047,0.000740,0.000520,0.000361,0.000249,0.000172,0.000120,0.000085,0.000060,0.000042,0.000030,0.000021,0.000015]
    cmfs_Z = [0.006450,0.010550,0.020050,0.036210,0.067850,0.110200,0.207400,0.371300,0.645600,1.039050,1.385600,1.622960,1.747060,1.782600,1.772110,1.744100,1.669200,1.528100,1.287640,1.041900,0.812950,0.616200,0.465180,0.353300,0.272000,0.212300,0.158200,0.111700,0.078250,0.057250,0.042160,0.029840,0.020300,0.013400,0.008750,0.005750,0.003900,0.002750,0.002100,0.001800,0.001650,0.001400,0.001100,0.001000,0.000800,0.000600,0.000340,0.000240,0.000190,0.000100,0.000050,0.000030,0.000020,0.000010,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    
    ## ignore accuracy so skip interpolation
    # pchip_obj = scipy.interpolate.PchipInterpolator(cmfs_wavelengths, cmfs_X)
    # print(pchip_obj(400))
    
    ## get CL500A measured light spectrum density as 
    light_spectrum_list = []
    for light_spectrum in spectrum[::Light_Increment]:
        light_spectrum_list.append(light_spectrum)
    
    ## Y = cmfs_Y * lightsource_spectrum
    Xn, Yn, Zn = 0.0, 0.0, 0.0
    for index in range(len(light_spectrum_list)):
        Xn = Xn+ cmfs_X[index] * light_spectrum_list[index] * 5
        Yn = Yn+ cmfs_Y[index] * light_spectrum_list[index] * 5
        Zn = Zn+ cmfs_Z[index] * light_spectrum_list[index] * 5
    
    normalized_ratio = 5 / Yn
    X, Y, Z = 0.0, 0.0, 0.0
    for index in range(len(light_spectrum_list)):
        X = X+ cmfs_X[index] * light_spectrum_list[index] * normalized_ratio
        Y = Y+ cmfs_Y[index] * light_spectrum_list[index] * normalized_ratio
        Z = Z+ cmfs_Z[index] * light_spectrum_list[index] * normalized_ratio
        
    return [X, Y, Z]
