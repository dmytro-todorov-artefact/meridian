# Copyright 2024 The Meridian Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helper functions and constants for Analysis module unit tests.

These constants are meant to be used in the end-to-end-like unit tests. They
were generated by running the model in a colab. The model was initialized with
random data (using seed=0), using same settings as the test Colab notebook.
"""

from collections.abc import Mapping, Sequence
from xml.etree import ElementTree as ET

from meridian import constants as c
import numpy as np
import pandas as pd
import xarray as xr


INC_IMPACT_MEDIA_AND_RF_USE_PRIOR = np.array([[
    [292.280, 103.246, 918.444, 231.445, 145.350],
    [284.166, 226.080, 313.855, 90.850, 76.162],
    [1318.729, 192.352, 85.081, 200.551, 218.050],
    [82.772, 259.622, 827.061, 434.968, 2084.384],
    [1238.606, 74.437, 369.152, 316.955, 829.716],
    [447.766, 196.569, 332.160, 206.566, 277.591],
    [284.244, 143.807, 979.908, 648.551, 231.348],
    [623.868, 366.245, 251.629, 136.273, 572.884],
    [1261.644, 176.535, 197.908, 337.837, 18.785],
    [208.106, 276.397, 103.565, 176.486, 445.057],
]])
INC_IMPACT_MEDIA_AND_RF_USE_POSTERIOR = np.array([
    [
        [319.356, 1224.602, 618.867, 498.894, 338.580],
        [322.174, 1359.807, 175.861, 145.931, 284.189],
        [155.755, 2911.697, 105.527, 198.284, 272.755],
        [745.873, 1673.716, 186.126, 251.668, 419.398],
        [756.005, 1630.19, 322.240, 104.903, 66.826],
        [125.727, 3350.011, 421.733, 1254.148, 137.125],
        [152.831, 2435.974, 398.307, 1717.595, 652.695],
        [1915.283, 336.376, 203.890, 1527.78, 283.467],
        [299.659, 1092.829, 71.807, 352.951, 210.160],
        [197.516, 963.677, 79.662, 383.824, 1453.590],
    ],
    [
        [485.850, 584.374, 554.454, 100.817, 329.212],
        [349.412, 196.939, 1246.548, 123.349, 272.997],
        [418.929, 54.107, 360.211, 265.047, 187.335],
        [567.911, 416.580, 888.026, 72.789, 82.438],
        [139.148, 108.430, 1028.455, 1372.795, 899.357],
        [1764.261, 209.111, 770.428, 108.084, 82.818],
        [1664.264, 172.911, 1050.239, 133.470, 90.200],
        [45.440, 868.115, 203.526, 177.862, 108.080],
        [771.144, 133.354, 160.595, 54.706, 947.392],
        [604.693, 313.808, 129.073, 144.443, 131.478],
    ],
])
INC_IMPACT_MEDIA_ONLY_USE_PRIOR = np.array([[
    [72.294, 124.757, 219.998],
    [111.245, 340.476, 1322.180],
    [243.282, 537.606, 551.248],
    [750.162, 1900.137, 519.069],
    [341.818, 415.011, 531.249],
    [439.134, 608.732, 121.669],
    [2313.321, 323.136, 692.972],
    [474.022, 312.865, 37.624],
    [1248.375, 215.313, 272.291],
    [558.978, 742.605, 555.243],
]])
INC_IMPACT_MEDIA_ONLY_USE_POSTERIOR = np.array([
    [
        [181.507, 4674.248, 149.593],
        [204.109, 6558.650, 1252.532],
        [195.388, 2999.373, 2631.340],
        [204.401, 6744.849, 59.055],
        [261.820, 4309.667, 248.466],
        [642.292, 3979.963, 2825.300],
        [881.553, 3978.233, 183.301],
        [1373.777, 3225.317, 279.114],
        [1348.765, 5123.532, 390.503],
        [892.066, 6685.637, 515.108],
    ],
    [
        [1725.798, 8974.518, 63.000],
        [6806.624, 3235.591, 3324.75],
        [567.261, 7380.798, 743.117],
        [310.246, 9488.625, 3889.356],
        [492.566, 303.134, 692.320],
        [166.423, 1639.318, 333.353],
        [597.806, 6173.561, 253.579],
        [1005.579, 2365.839, 482.680],
        [454.964, 6894.853, 1338.915],
        [705.025, 809.158, 345.726],
    ],
])
INC_IMPACT_RF_ONLY_USE_PRIOR = np.array([[
    [475.681, 899.632],
    [219.095, 402.044],
    [1006.009, 690.104],
    [382.913, 2338.101],
    [198.065, 167.123],
    [1015.466, 386.239],
    [403.642, 482.581],
    [415.762, 594.456],
    [239.836, 187.258],
    [315.780, 190.132],
]])
INC_IMPACT_RF_ONLY_USE_POSTERIOR = np.array([
    [
        [44.915, 83.615],
        [1230.312, 2874.910],
        [403.201, 528.229],
        [642.859, 220.397],
        [524.696, 117.616],
        [90.330, 792.652],
        [74.400, 140.923],
        [440.459, 687.958],
        [724.000, 225.569],
        [171.506, 323.671],
    ],
    [
        [37.575, 1869.470],
        [37.506, 1863.426],
        [37.519, 1858.543],
        [37.516, 1856.388],
        [37.540, 1858.885],
        [37.570, 1856.769],
        [37.437, 1860.816],
        [37.681, 1869.188],
        [37.580, 1870.828],
        [37.413, 1884.181],
    ],
])

MROI_MEDIA_AND_RF_USE_PRIOR = np.array([[
    [0.517, 0.225, 1.976, 0.444, 0.205],
    [0.456, 0.267, 0.602, 0.175, 0.076],
    [2.144, 0.331, 0.126, 0.307, 0.324],
    [0.172, 0.647, 1.372, 0.696, 2.750],
    [2.587, 0.178, 0.672, 0.384, 1.290],
    [0.866, 0.378, 0.277, 0.212, 0.322],
    [0.140, 0.229, 1.565, 1.276, 0.247],
    [0.790, 0.487, 0.535, 0.117, 0.628],
    [0.699, 0.264, 0.357, 0.548, 0.019],
    [0.386, 0.298, 0.165, 0.262, 0.952],
]])
MROI_MEDIA_AND_RF_USE_PRIOR_BY_REACH = np.array([[
    [0.517, 0.225, 1.976, 0.850, 0.504],
    [0.456, 0.267, 0.602, 0.333, 0.264],
    [2.144, 0.331, 0.126, 0.736, 0.757],
    [0.172, 0.647, 1.372, 1.598, 7.240],
    [2.587, 0.178, 0.672, 1.164, 2.882],
    [0.866, 0.378, 0.277, 0.759, 0.964],
    [0.140, 0.229, 1.565, 2.383, 0.803],
    [0.790, 0.487, 0.535, 0.500, 1.989],
    [0.699, 0.264, 0.357, 1.241, 0.065],
    [0.386, 0.298, 0.165, 0.648, 1.546],
]])
MROI_MEDIA_AND_RF_USE_POSTERIOR = np.array([
    [
        [0.598, 1.958, 1.359, 0.819, 0.226],
        [0.574, 2.657, 0.358, 0.142, 0.496],
        [0.238, 5.807, 0.199, 0.230, 0.391],
        [1.134, 3.416, 0.400, 0.347, 0.473],
        [0.954, 3.253, 0.708, 0.121, 0.088],
        [0.121, 8.351, 0.769, 1.704, 0.188],
        [0.243, 4.900, 0.815, 2.103, 1.216],
        [3.908, 0.324, 0.370, 2.087, 0.551],
        [0.564, 1.233, 0.130, 0.547, 0.189],
        [0.215, 1.108, 0.114, 1.076, 2.731],
    ],
    [
        [0.789, 1.232, 1.218, 0.163, 0.233],
        [0.609, 0.482, 2.431, 0.192, 0.156],
        [0.687, 0.053, 0.706, 0.454, 0.361],
        [1.116, 0.310, 0.879, 0.101, 0.059],
        [0.218, 0.214, 2.436, 1.133, 0.436],
        [3.203, 0.408, 1.420, 0.107, 0.057],
        [2.391, 0.252, 1.910, 0.149, 0.051],
        [0.095, 1.375, 0.466, 0.202, 0.296],
        [0.366, 0.281, 0.328, 0.081, 0.602],
        [1.351, 0.715, 0.283, 0.239, 0.070],
    ],
])
MROI_MEDIA_AND_RF_USE_POSTERIOR_BY_REACH = np.array([
    [
        [0.598, 1.958, 1.359, 1.832, 1.176],
        [0.574, 2.657, 0.358, 0.536, 0.987],
        [0.238, 5.807, 0.199, 0.728, 0.947],
        [1.134, 3.416, 0.400, 0.924, 1.456],
        [0.954, 3.253, 0.708, 0.385, 0.232],
        [0.121, 8.351, 0.769, 4.608, 0.476],
        [0.243, 4.900, 0.815, 6.310, 2.267],
        [3.908, 0.324, 0.370, 5.613, 0.984],
        [0.564, 1.233, 0.130, 1.296, 0.729],
        [0.215, 1.108, 0.114, 1.410, 5.049],
    ],
    [
        [0.789, 1.232, 1.218, 0.370, 1.143],
        [0.609, 0.482, 2.431, 0.453, 0.948],
        [0.687, 0.053, 0.706, 0.973, 0.650],
        [1.116, 0.310, 0.879, 0.267, 0.286],
        [0.218, 0.214, 2.436, 5.044, 3.124],
        [3.203, 0.408, 1.420, 0.397, 0.287],
        [2.391, 0.252, 1.910, 0.490, 0.313],
        [0.095, 1.375, 0.466, 0.653, 0.375],
        [0.366, 0.281, 0.328, 0.200, 3.290],
        [1.351, 0.715, 0.283, 0.530, 0.456],
    ],
])
MROI_MEDIA_ONLY_USE_PRIOR = np.array([[
    [0.063, 0.163, 0.456],
    [0.238, 0.718, 2.009],
    [0.425, 1.178, 0.949],
    [1.499, 2.992, 1.245],
    [0.467, 1.010, 0.460],
    [0.596, 1.011, 0.214],
    [4.385, 0.743, 1.122],
    [0.791, 0.421, 0.091],
    [2.701, 0.451, 0.769],
    [0.915, 0.720, 1.339],
]])
MROI_MEDIA_ONLY_USE_POSTERIOR = np.array([
    [
        [0.434, 11.111, 0.336],
        [0.474, 12.774, 2.424],
        [0.118, 6.825, 6.218],
        [0.434, 12.541, 0.106],
        [0.507, 7.852, 0.559],
        [1.104, 8.963, 6.693],
        [1.360, 7.941, 0.464],
        [1.560, 5.297, 0.378],
        [1.783, 11.469, 1.016],
        [1.240, 15.824, 1.232],
    ],
    [
        [3.656, 15.143, 0.111],
        [8.248, 7.154, 8.247],
        [0.637, 14.318, 1.519],
        [0.370, 20.936, 6.977],
        [0.902, 0.277, 1.367],
        [0.261, 3.416, 0.785],
        [1.106, 13.360, 0.491],
        [1.896, 4.443, 0.890],
        [0.915, 13.588, 3.155],
        [1.297, 1.886, 0.779],
    ],
])
MROI_RF_ONLY_USE_PRIOR = np.array([[
    [1.032, 1.225],
    [0.220, 0.693],
    [1.355, 0.416],
    [0.413, 2.789],
    [0.186, 0.125],
    [1.219, 0.586],
    [0.792, 0.774],
    [1.251, 0.722],
    [0.256, 0.392],
    [0.622, 1.059],
]])
MROI_RF_ONLY_USE_PRIOR_BY_REACH = np.array([[
    [1.744, 3.125],
    [0.813, 1.397],
    [3.697, 2.392],
    [1.396, 8.113],
    [0.728, 0.582],
    [3.734, 1.346],
    [1.491, 1.679],
    [1.532, 2.062],
    [0.884, 0.642],
    [1.161, 0.664],
]])
MROI_RF_ONLY_USE_POSTERIOR = np.array([
    [
        [0.079, 0.132],
        [2.615, 2.905],
        [0.852, 0.787],
        [0.564, 0.336],
        [0.480, 0.138],
        [0.110, 1.183],
        [0.089, 0.164],
        [0.524, 0.779],
        [1.436, 0.125],
        [0.183, 0.542],
    ],
    [
        [0.030, 0.745],
        [0.026, 0.734],
        [0.029, 0.743],
        [0.029, 0.729],
        [0.032, 0.735],
        [0.029, 0.734],
        [0.031, 0.728],
        [0.032, 0.735],
        [0.026, 0.745],
        [0.028, 0.754],
    ],
])
MROI_RF_ONLY_USE_POSTERIOR_BY_REACH = np.array([
    [
        [0.164, 0.284],
        [4.514, 9.987],
        [1.483, 1.839],
        [2.370, 0.762],
        [1.930, 0.411],
        [0.332, 2.749],
        [0.275, 0.490],
        [1.609, 2.386],
        [2.657, 0.786],
        [0.629, 1.122],
    ],
    [
        [0.137, 6.494],
        [0.140, 6.472],
        [0.135, 6.450],
        [0.132, 6.446],
        [0.144, 6.448],
        [0.144, 6.449],
        [0.135, 6.456],
        [0.144, 6.497],
        [0.135, 6.507],
        [0.137, 6.548],
    ],
])

SAMPLE_ROI = np.array([
    [
        [2.056, 2.008],
        [0.473, 0.414],
        [4.400, 6.030],
    ],
    [
        [0.722, 3.592],
        [0.313, 0.379],
        [1.168, 10.520],
    ],
    [
        [1.712, 1.754],
        [0.365, 0.309],
        [3.723, 4.144],
    ],
    [
        [1.021, 1.651],
        [0.408, 0.264],
        [2.029, 5.648],
    ],
    [
        [1.701, 1.259],
        [0.154, 0.283],
        [5.279, 3.378],
    ],
    [
        [1.448, 2.054],
        [0.784, 0.951],
        [2.378, 3.811],
    ],
])
SAMPLE_MROI = np.array([
    [
        [0.876, 0.969],
        [0.154, 0.120],
        [2.388, 3.239],
    ],
    [
        [0.330, 1.916],
        [0.199, 0.206],
        [0.575, 5.934],
    ],
    [
        [0.765, 0.865],
        [0.144, 0.129],
        [1.791, 2.431],
    ],
    [
        [0.442, 0.600],
        [0.143, 0.100],
        [1.015, 2.088],
    ],
    [
        [0.681, 0.443],
        [0.044, 0.057],
        [2.093, 1.292],
    ],
    [
        [0.621, 0.959],
        [0.341, 0.444],
        [1.093, 1.868],
    ],
])
SAMPLE_CPIK = np.array([
    [
        [3.08717942, 3.08717942],
        [0.71385074, 0.71385074],
        [8.1156775, 8.1156775],
    ],
    [
        [5.25037432, 5.25037432],
        [2.74044206, 2.74044206],
        [10.07572379, 10.07572379],
    ],
    [
        [3.42448807, 3.42448807],
        [0.84418333, 0.84418333],
        [8.68120933, 8.68120933],
    ],
    [
        [4.02361917, 4.02361917],
        [1.59949402, 1.59949402],
        [7.73949795, 7.73949795],
    ],
    [
        [8.16407204, 8.16407204],
        [0.71428683, 0.71428683],
        [31.43474998, 31.43474998],
    ],
    [
        [2.46112823, 2.46112823],
        [1.31645199, 1.31645199],
        [4.02072748, 4.02072748],
    ],
])
SAMPLE_EFFECTIVENESS = np.array([
    [
        [4.161e-01, 4.063e-01],
        [9.584e-02, 8.382e-02],
        [8.905e-01, 1.220e00],
    ],
    [
        [1.464e-01, 7.280e-01],
        [6.351e-02, 7.682e-02],
        [2.367e-01, 2.131e00],
    ],
    [
        [3.439e-01, 3.525e-01],
        [7.336e-02, 6.227e-02],
        [7.480e-01, 8.327e-01],
    ],
    [
        [8.498e-05, 1.373e-04],
        [3.401e-05, 2.197e-05],
        [1.688e-04, 4.698e-04],
    ],
    [
        [1.400e-04, 1.036e-04],
        [1.275e-05, 2.335e-05],
        [4.345e-04, 2.781e-04],
    ],
    [
        [2.970e-04, 4.211e-04],
        [1.608e-04, 1.951e-04],
        [4.875e-04, 7.813e-04],
    ],
])
SAMPLE_SPEND = np.array([293.807, 278.854, 255.744, 272.165, 287.876, 1388.448])
SAMPLE_PCT_OF_SPEND = np.array([21.160, 20.083, 18.419, 19.602, 20.733, 100.0])
SAMPLE_INCREMENTAL_IMPACT = np.array([
    [
        [604.218, 590.062],
        [139.173, 121.713],
        [1293.041, 1771.812],
    ],
    [
        [201.529, 1001.830],
        [87.401, 105.714],
        [325.814, 2933.613],
    ],
    [
        [437.876, 448.779],
        [93.398, 79.269],
        [952.249, 1060.055],
    ],
    [
        [278.048, 449.467],
        [111.291, 71.885],
        [552.439, 1537.270],
    ],
    [
        [489.933, 362.505],
        [44.605, 81.657],
        [1519.783, 972.702],
    ],
    [
        [2011.605, 2852.645],
        [1089.450, 1321.615],
        [3301.822, 5292.183],
    ],
])
SAMPLE_PCT_OF_CONTRIBUTION = np.array([
    [
        [1.579, 2.984],
        [0.363, 0.615],
        [3.379, 8.961],
    ],
    [
        [0.526, 5.066],
        [0.228, 0.534],
        [0.851, 14.837],
    ],
    [
        [1.144, 2.269],
        [0.244, 0.400],
        [2.488, 5.361],
    ],
    [
        [0.726, 2.273],
        [0.290, 0.363],
        [1.443, 7.774],
    ],
    [
        [1.280, 1.833],
        [0.116, 0.412],
        [3.972, 4.919],
    ],
    [
        [5.257, 14.427],
        [2.847, 6.684],
        [8.630, 26.765],
    ],
])
ADSTOCK_DECAY_CI_HI = np.array([1.0, 1.0, 0.949, 0.985, 0.902])
ADSTOCK_DECAY_CI_LO = np.array([1.0, 1.0, 0.625, 0.838, 0.391])
ADSTOCK_DECAY_MEAN = np.array([1.0, 1.0, 0.815, 0.929, 0.676])
HILL_CURVES_CI_HI = np.array([0.0, 0.0, 0.01570, 0.03354, 0.03086])
HILL_CURVES_CI_LO = np.array([0.0, 0.0, 0.00340, 0.00350, 0.00677])
HILL_CURVES_MEAN = np.array([0.0, 0.0, 0.00811, 0.01113, 0.01597])
HILL_CURVES_COUNT_HISTOGRAM = np.array(
    [34.55127961, 34.55127961, 51.82691941, 51.82691941, 17.2756398]
)
HILL_CURVES_SCALED_COUNT_HISTOGRAM = np.array(
    [0.06667, 0.06667, 0.09999999, 0.09999999, 0.03333]
)
HILL_CURVES_START_INTERVAL_HISTOGRAM = np.array(
    [0.00445, 0.00468, 0.004898, 0.00512, 0.00534]
)
HILL_CURVES_END_INTERVAL_HISTOGRAM = np.array(
    [0.00468, 0.004898, 0.00512, 0.00534, 0.00557]
)

PREDICTIVE_ACCURACY_NO_HOLDOUT_ID_NO_GEOS_OR_TIMES = np.array(
    [0.492, 0.960, 14.385, 0.050, 0.390, 0.044]
)
PREDICTIVE_ACCURACY_NO_HOLDOUT_ID_GEOS_NO_TIMES = np.array(
    [0.356, 0.480, 0.933, 0.357, 0.387, 0.258]
)
PREDICTIVE_ACCURACY_NO_HOLDOUT_ID_TIMES_NO_GEOS = np.array(
    [0.766, 0.998, 1.599, 0.027, 0.408, 0.014]
)
PREDICTIVE_ACCURACY_NO_HOLDOUT_ID_TIMES_AND_GEOS = np.array(
    [-0.836, -0.996, 0.609, 0.354, 0.444, 0.364]
)
PREDICTIVE_ACCURACY_HOLDOUT_ID = np.array([
    5.00223070e-01,
    4.26935375e-01,
    4.92482483e-01,
    9.06991921e-01,
    6.31574780e-01,
    0.960301068,
    1.18091524e00,
    7.16603165e01,
    1.44137821e01,
    0.113417998,
    1.01811195e02,
    0.0506119095,
    0.38362843,
    0.43501151,
    3.90862823e-01,
    8.96737352e-02,
    0.33451566,
    0.04509374,
])
SAMPLE_IMPRESSIONS = np.array([
    1.4520000e03,
    1.3760000e03,
    1.2730000e03,
    3.2716225e06,
    3.4970928e06,
    6.7728160e06,
])
SAMPLE_PCT_OF_IMPRESSIONS = np.array([
    2.143865e-02,
    2.031651e-02,
    1.879573e-02,
    4.830520e01,
    5.163425e01,
    1.000000e02,
])
SAMPLE_CPM = np.array([
    2.023468e02,
    2.026558e02,
    2.008992e02,
    8.318969e-02,
    8.231886e-02,
    2.050032e-01,
])


def generate_model_fit_data(
    geo: Sequence[str] | None = None,
    time: Sequence[str] | None = None,
    actual: Sequence[Sequence[int]] | None = None,
) -> xr.Dataset:
  """Helper method to generate simulated model fit analyzed data."""
  metric = [c.MEAN, c.CI_LO, c.CI_HI]
  if geo:
    n_geos = len(geo)
  else:
    n_geos = 5
    geo = [f"geo {i}" for i in range(n_geos)]
  if time:
    n_time = len(time)
  else:
    n_time = 52
    time = pd.date_range("2023-01-01", freq="W-SUN", periods=n_time).format(
        formatter=lambda x: x.strftime("%Y-%m-%d")
    )

  np.random.seed(0)
  expected = abs(np.random.lognormal(10, 1, size=(n_geos, n_time, 3)))
  baseline = abs(np.random.lognormal(10, 1, size=(n_geos, n_time, 3)))
  if not actual:
    actual = abs(np.random.lognormal(10, 1, size=(n_geos, n_time)))

  return xr.Dataset(
      data_vars={
          c.EXPECTED: (
              [c.GEO, c.TIME, c.METRIC],
              expected,
          ),
          c.BASELINE: (
              [c.GEO, c.TIME, c.METRIC],
              baseline,
          ),
          c.ACTUAL: ([c.GEO, c.TIME], actual),
      },
      coords={
          c.GEO: geo,
          c.TIME: time,
          c.METRIC: metric,
      },
      attrs={c.CONFIDENCE_LEVEL: 0.9},
  )


def generate_predictive_accuracy_table(
    with_holdout: bool = False, column_var: str | None = None
) -> pd.DataFrame:
  """Helper method to simulate predictive accuracy DataFrame for Summarizer."""
  metric = [c.R_SQUARED, c.MAPE, c.WMAPE]
  geo_granularity = [c.GEO, c.NATIONAL]
  evaluation_set = [c.TRAIN, c.TEST, c.ALL_DATA]

  shape = [len(metric), len(geo_granularity)]
  dims = [c.METRIC, c.GEO_GRANULARITY]
  coords = {
      c.METRIC: ([c.METRIC], metric),
      c.GEO_GRANULARITY: ([c.GEO_GRANULARITY], geo_granularity),
  }
  if with_holdout:
    shape.append(len(evaluation_set))
    dims.append(c.EVALUATION_SET_VAR)
    coords[c.EVALUATION_SET_VAR] = (
        [c.EVALUATION_SET_VAR],
        evaluation_set,
    )
  np.random.seed(0)
  value = np.random.lognormal(0, 1, size=shape)
  ds = xr.Dataset(
      data_vars={
          c.VALUE: (dims, value),
      },
      coords=coords,
  )
  df = ds.to_dataframe().reset_index()
  if not column_var:
    return df
  coords = list(ds.coords)
  if column_var not in coords:
    raise ValueError(
        f"The DataFrame cannot be pivoted by {column_var} as it does not"
        " exist in the DataFrame."
    )
  indices = coords.copy()
  indices.remove(column_var)
  pivoted_df = (
      df.pivot(
          index=indices,
          columns=column_var,
          values=c.VALUE,
      )
      .reset_index()
      .rename_axis(None, axis=1)
  )
  # The 2-Pager displays the national predictive accuracy metric data only.
  national_table = pivoted_df[pivoted_df[c.GEO_GRANULARITY] == c.NATIONAL]
  return national_table


def generate_media_summary_metrics() -> xr.Dataset:
  """Helper method to generate simulated media summary metrics data."""
  channel = [f"channel {i}" for i in range(5)]
  channel.append(c.ALL_CHANNELS)
  metric = [c.MEAN, c.CI_LO, c.CI_HI]
  distribution = [c.PRIOR, c.POSTERIOR]

  np.random.seed(0)
  shape = (len(channel), len(metric), len(distribution))
  pct_of_spend = np.random.randint(low=0, high=100, size=len(channel))
  spend = np.random.randint(low=10, high=1000, size=len(channel))
  impressions = np.random.randint(low=10, high=1000, size=len(channel))
  cpm = np.random.random(size=len(channel))
  roi = np.random.lognormal(1, 1, size=shape)
  mroi = np.random.lognormal(0, 1, size=shape)
  cpik = np.random.lognormal(0, 1, size=shape)
  incremental_impact = np.random.lognormal(10, 1, size=shape)
  effectiveness = np.random.lognormal(1, 1, size=shape)
  pct_of_contribution = np.random.randint(low=0, high=50, size=shape)
  pct_of_impressions = np.random.randint(low=0, high=100, size=len(channel))

  return xr.Dataset(
      data_vars={
          c.IMPRESSIONS: ([c.CHANNEL], impressions),
          c.PCT_OF_IMPRESSIONS: ([c.CHANNEL], pct_of_impressions),
          c.SPEND: ([c.CHANNEL], spend),
          c.PCT_OF_SPEND: ([c.CHANNEL], pct_of_spend),
          c.CPM: ([c.CHANNEL], cpm),
          c.INCREMENTAL_IMPACT: (
              [c.CHANNEL, c.METRIC, c.DISTRIBUTION],
              incremental_impact,
          ),
          c.PCT_OF_CONTRIBUTION: (
              [c.CHANNEL, c.METRIC, c.DISTRIBUTION],
              pct_of_contribution,
          ),
          c.ROI: (
              [c.CHANNEL, c.METRIC, c.DISTRIBUTION],
              roi,
          ),
          c.EFFECTIVENESS: (
              [c.CHANNEL, c.METRIC, c.DISTRIBUTION],
              effectiveness,
          ),
          c.MROI: (
              [c.CHANNEL, c.METRIC, c.DISTRIBUTION],
              mroi,
          ),
          c.CPIK: (
              [c.CHANNEL, c.METRIC, c.DISTRIBUTION],
              cpik,
          ),
      },
      coords={
          c.CHANNEL: channel,
          c.METRIC: metric,
          c.DISTRIBUTION: distribution,
      },
      attrs={c.CONFIDENCE_LEVEL: 0.9},
  )


def generate_response_curve_data(
    n_channels: int = 5, spend_multiplier: Sequence[int] | None = None
) -> xr.Dataset:
  """Helper method to generate simulated response curve data."""
  channels = [f"channel {i}" for i in range(n_channels)]
  metric = [c.MEAN, c.CI_LO, c.CI_HI]
  spend_multiplier = (
      list(np.arange(0, 2, 0.05))
      if spend_multiplier is None
      else spend_multiplier
  )

  np.random.seed(0)
  shape = (
      len(spend_multiplier),
      len(channels),
      len(metric),
  )
  spend = np.random.lognormal(
      25, 1, size=(len(spend_multiplier), len(channels))
  )
  incremental_impact = np.random.lognormal(10, 1, size=shape)

  xarray = xr.Dataset(
      data_vars={
          c.SPEND: (
              [c.SPEND_MULTIPLIER, c.CHANNEL],
              spend,
          ),
          c.INCREMENTAL_IMPACT: (
              [c.SPEND_MULTIPLIER, c.CHANNEL, c.METRIC],
              incremental_impact,
          ),
      },
      coords={
          c.CHANNEL: channels,
          c.METRIC: metric,
          c.SPEND_MULTIPLIER: spend_multiplier,
      },
      attrs={c.CONFIDENCE_LEVEL: 0.9},
  )
  return xarray


def generate_predictive_accuracy_data(holdout_id: bool = False) -> xr.Dataset:
  """Helper method to generate simulated predictive accuracy data."""

  np.random.seed(0)

  xr_dims = [c.METRIC, c.GEO_GRANULARITY]
  xr_coords = {
      c.METRIC: (
          [c.METRIC],
          [c.R_SQUARED, c.MAPE, c.WMAPE],
      ),
      c.GEO_GRANULARITY: (
          [c.GEO_GRANULARITY],
          [c.GEO, c.NATIONAL],
      ),
  }
  rsquared_arr = [np.random.uniform(0.0, 1.0) for _ in range(2)]
  mape_arr = [np.random.uniform(0.0, 1.0) for _ in range(2)]
  wmape_arr = [np.random.uniform(0.0, 1.0) for _ in range(2)]

  if not holdout_id:
    stacked_metric_values = np.stack([rsquared_arr, mape_arr, wmape_arr])
    xr_data = {c.VALUE: (xr_dims, stacked_metric_values)}
  else:
    geo_train = [np.random.uniform(0.0, 1.0) for _ in range(3)]
    national_train = [np.random.uniform(0.0, 1.0) for _ in range(3)]
    geo_test = [np.random.uniform(0.0, 1.0) for _ in range(3)]
    national_test = [np.random.uniform(0.0, 1.0) for _ in range(3)]
    geo_all_data = [np.random.uniform(0.0, 1.0) for _ in range(3)]
    national_all_data = [np.random.uniform(0.0, 1.0) for _ in range(3)]

    stacked_train = np.stack([geo_train, national_train], axis=-1)
    stacked_test = np.stack([geo_test, national_test], axis=-1)
    stacked_all_data = np.stack([geo_all_data, national_all_data], axis=-1)
    stacked_total = np.stack(
        [stacked_train, stacked_test, stacked_all_data], axis=-1
    )

    xr_dims.append(c.EVALUATION_SET_VAR)
    xr_coords[c.EVALUATION_SET_VAR] = (
        [c.EVALUATION_SET_VAR],
        list(c.EVALUATION_SET),
    )
    xr_data = {c.VALUE: (xr_dims, stacked_total)}

  return xr.Dataset(data_vars=xr_data, coords=xr_coords)


def generate_optimal_frequency_data(
    channel_prefix: str = c.RF_CHANNEL,
    num_channels: int = 5,
    use_roi: bool = True,
) -> xr.Dataset:
  """Helper method to generate simulated optimal frequency data."""
  frequency = list(np.arange(1, 7.05, 0.1))
  rf_channel = [f"{channel_prefix} {i}" for i in range(num_channels)]
  metric = [c.MEAN, c.CI_LO, c.CI_HI]

  np.random.seed(0)
  metric_by_frequency = np.random.lognormal(
      1, 1, size=(len(frequency), len(rf_channel), len(metric))
  )
  optimal_frequency = np.random.lognormal(1, 1, size=(len(rf_channel)))

  metric_name = c.ROI if use_roi else c.CPIK
  return xr.Dataset(
      data_vars={
          metric_name: (
              [c.FREQUENCY, c.RF_CHANNEL, c.METRIC],
              metric_by_frequency,
          ),
          c.OPTIMAL_FREQUENCY: (
              [c.RF_CHANNEL],
              optimal_frequency,
          ),
      },
      coords={
          c.FREQUENCY: frequency,
          c.RF_CHANNEL: rf_channel,
          c.METRIC: metric,
      },
  )


def generate_hill_curves_dataframe() -> pd.DataFrame:
  """Helper method to generate simulated hill curve data."""
  channel_names = [f"channel {i}" for i in range(5)]
  channel_array = []
  channel_type_array = []
  for i, channel in enumerate(channel_names):
    for _ in range(100):
      channel_array.append(channel)
      if i <= 3:
        channel_type_array.append(c.MEDIA)
      else:
        channel_type_array.append(c.RF)

  np.random.seed(0)
  media_units_array = [
      np.random.uniform(0, 1000) for _ in range(len(channel_array))
  ]
  distribution_array = [
      c.POSTERIOR if i % 2 == 0 else c.PRIOR for i in range(len(channel_array))
  ]
  ci_hi_array = [np.random.rand() for _ in range(len(channel_array))]
  ci_lo_array = [np.random.rand() for _ in range(len(channel_array))]
  mean_array = [np.random.rand() for _ in range(len(channel_array))]

  one_channel_impressions_hist = [
      np.random.rand() if i == 0 else None for i in range(100)
  ]
  one_channel_start_interval_hist = [
      np.random.uniform(0, 1000) if i == 0 else None for i in range(100)
  ]
  one_channel_end_interval_hist = [
      np.random.uniform(0, 1000) if i == 0 else None for i in range(100)
  ]

  scaled_count, start_interval, end_interval = [], [], []

  for _ in range(len(channel_names)):
    scaled_count.extend(one_channel_impressions_hist)
    start_interval.extend(one_channel_start_interval_hist)
    end_interval.extend(one_channel_end_interval_hist)

  data_hill = {
      c.CHANNEL: channel_array,
      c.MEDIA_UNITS: media_units_array,
      c.DISTRIBUTION: distribution_array,
      c.CI_HI: ci_hi_array,
      c.CI_LO: ci_lo_array,
      c.MEAN: mean_array,
      c.CHANNEL_TYPE: channel_type_array,
      c.SCALED_COUNT_HISTOGRAM: scaled_count,
      c.START_INTERVAL_HISTOGRAM: start_interval,
      c.END_INTERVAL_HISTOGRAM: end_interval,
  }

  return pd.DataFrame(data_hill).reset_index(drop=True)


def generate_adstock_decay_data() -> pd.DataFrame:
  """Helper method to generate simulated adstock decay data."""
  channel_names = [f"channel {i}" for i in range(5)]

  channel_array, time_units_array = [], []
  np.random.seed(0)

  for channel in range(len(channel_names)):
    for _ in range(100):
      channel_array.append(channel)
    time_units_array.extend(np.linspace(0, 20, 100))

  distribution_array = [
      c.PRIOR if i % 2 == 0 else c.POSTERIOR for i in range(len(channel_array))
  ]
  mean_array = [np.random.rand() for _ in range(len(channel_array))]
  ci_lo_array = [np.random.rand() for _ in range(len(channel_array))]
  ci_hi_array = [np.random.rand() for _ in range(len(channel_array))]

  data_hill = {
      c.TIME_UNITS: time_units_array,
      c.CHANNEL: channel_array,
      c.DISTRIBUTION: distribution_array,
      c.MEAN: mean_array,
      c.CI_LO: ci_lo_array,
      c.CI_HI: ci_hi_array,
  }

  return pd.DataFrame(data_hill).reset_index(drop=True)


def get_child_element(
    root: ET.Element,
    path: str,
    attribs: Mapping[str, str] | None = None,
) -> ET.Element:
  """Searches for a descendant element under `root` with the given path.

  Args:
    root: The top-level element to search from.
    path: Path fom the root to search.
    attribs: Optional attribute match to search for.

  Returns:
    ElementTree Element found from the path and attribute match.
  Raises:
    AssertionError if not found.
  """
  for div in root.findall(path):
    if attribs:
      if not (attribs.items() <= div.attrib.items()):
        continue
    return div
  raise AssertionError(
      f"Cannot find child element {path} under {root} "
      + (f"with {attribs}" if attribs else "")
  )


def get_table_row_values(tr: ET.Element, row_element="td") -> Sequence[str]:
  row_values = []
  for row in tr.findall(row_element):
    row_values.append(row.text or "")
  return row_values


def generate_selected_times(start: str, periods: int) -> Sequence[str]:
  return pd.date_range(start, freq="W-SUN", periods=periods).format(
      formatter=lambda x: x.strftime("%Y-%m-%d")
  )
