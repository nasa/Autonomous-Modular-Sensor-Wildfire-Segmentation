import json
import warnings

from pyhdf.SD import *
import scipy.interpolate
from osgeo.gdalconst import *
from shapely.geometry import Polygon
from osgeo import gdal
import glob, os
import ntpath
import numpy as np
import rasterio
from rasterio.mask import mask
import cv2
import dataset_utils

os.environ['GDAL_PAM_ENABLED'] = 'NO'
import sklearn.feature_extraction
import tifffile

FINAL_RESOLUTION = 0.0001801801802 / 2  # 10 meters
CORRUPTED_AMS_FILES = """FIRE_EMPHASIZING_ENHANCED_CLEAN_TIF_AMSL1B_0700103_03_20061028_0118_0122_V02
FIRE_EMPHASIZING_ENHANCED_CLEAN_TIF_AMSL1B_0700103_02_20061028_0105_0112_V02
FIRE_EMPHASIZING_ENHANCED_CLEAN_TIF_AMSL1B_0700103_08_20061028_0735_0745_V02
FIRE_EMPHASIZING_ENHANCED_CLEAN_TIF_AMSL1B_0700103_09_20061028_0751_0756_V02
FIRE_EMPHASIZING_ENHANCED_CLEAN_TIF_AMSL1B_0700103_10_20061028_0812_0824_V02
FIRE_EMPHASIZING_ENHANCED_CLEAN_TIF_AMSL1B_0700103_11_20061028_0829_0833_V02
FIRE_EMPHASIZING_ENHANCED_CLEAN_TIF_AMSL1B_0700103_12_20061028_0849_0858_V02
FIRE_EMPHASIZING_ENHANCED_CLEAN_TIF_AMSL1B_0700103_13_20061028_0907_0912_V02
FIRE_EMPHASIZING_ENHANCED_CLEAN_TIF_AMSL1B_0700103_15_20061028_0955_0959_V02
FIRE_EMPHASIZING_ENHANCED_CLEAN_TIF_AMSL1B_0700103_16_20061028_1003_1016_V02
FIRE_EMPHASIZING_ENHANCED_CLEAN_TIF_AMSL1B_0700103_17_20061028_1041_1055_V02
FIRE_EMPHASIZING_ENHANCED_CLEAN_TIF_AMSL1B_0700103_18_20061028_1100_1104_V02
FIRE_EMPHASIZING_ENHANCED_CLEAN_TIF_AMSL1B_0700103_19_20061028_1108_1122_V02
FIRE_EMPHASIZING_ENHANCED_CLEAN_TIF_AMSL1B_0700103_20_20061028_1126_1130_V02
FIRE_EMPHASIZING_ENHANCED_CLEAN_TIF_AMSL1B_0700103_21_20061028_1138_1149_V02
FIRE_EMPHASIZING_ENHANCED_CLEAN_TIF_AMSL1B_0700103_22_20061028_1157_1202_V02
FIRE_EMPHASIZING_ENHANCED_CLEAN_TIF_AMSL1B_0700103_23_20061028_1209_1221_V02
FIRE_EMPHASIZING_ENHANCED_CLEAN_TIF_AMSL1B_0700103_24_20061028_1227_1232_V02
FIRE_EMPHASIZING_ENHANCED_CLEAN_TIF_AMSL1B_0700103_25_20061028_1239_1251_V02
FIRE_EMPHASIZING_ENHANCED_CLEAN_TIF_AMSL1B_0700103_26_20061028_1259_1303_V02
FIRE_EMPHASIZING_ENHANCED_CLEAN_TIF_AMSL1B_0700103_27_20061028_1316_1324_V02
FIRE_EMPHASIZING_ENHANCED_CLEAN_TIF_AMSL1B_0704300_04_20070829_0235_0244_V02
FIRE_EMPHASIZING_ENHANCED_CLEAN_TIF_AMSL1B_0704300_05_20070829_0246_0254_V02
FIRE_EMPHASIZING_ENHANCED_CLEAN_TIF_AMSL1B_0704300_06_20070829_0400_0408_V02
FIRE_EMPHASIZING_ENHANCED_CLEAN_TIF_AMSL1B_0704300_07_20070829_0413_0423_V02
FIRE_EMPHASIZING_ENHANCED_CLEAN_TIF_AMSL1B_0704300_08_20070829_0426_0434_V02
FIRE_EMPHASIZING_ENHANCED_CLEAN_TIF_AMSL1B_0704300_09_20070829_0454_0501_V02
FIRE_EMPHASIZING_ENHANCED_CLEAN_TIF_AMSL1B_0704300_10_20070829_0503_0512_V02
FIRE_EMPHASIZING_ENHANCED_CLEAN_TIF_AMSL1B_0704300_11_20070829_0514_0521_V02
FIRE_EMPHASIZING_ENHANCED_CLEAN_TIF_AMSL1B_0704300_12_20070829_0544_0554_V02
FIRE_EMPHASIZING_ENHANCED_CLEAN_TIF_AMSL1B_0704300_13_20070829_0556_0605_V02
FIRE_EMPHASIZING_ENHANCED_CLEAN_TIF_AMSL1B_0704300_14_20070829_0607_0616_V02
FIRE_EMPHASIZING_ENHANCED_CLEAN_TIF_AMSL1B_0704300_15_20070829_0742_0751_V02
FIRE_EMPHASIZING_ENHANCED_CLEAN_TIF_AMSL1B_0704300_16_20070829_0753_0801_V02
FIRE_EMPHASIZING_ENHANCED_CLEAN_TIF_AMSL1B_0704300_17_20070829_0804_0813_V02
FIRE_EMPHASIZING_ENHANCED_CLEAN_TIF_AMSL1B_0704300_18_20070829_0816_0824_V02
FIRE_EMPHASIZING_ENHANCED_CLEAN_TIF_AMSL1B_0704300_19_20070829_0829_0839_V02
FIRE_EMPHASIZING_ENHANCED_CLEAN_TIF_AMSL1B_0704300_20_20070829_0841_0850_V02
FIRE_EMPHASIZING_ENHANCED_CLEAN_TIF_AMSL1B_0704300_21_20070829_0853_0902_V02
FIRE_EMPHASIZING_ENHANCED_CLEAN_TIF_AMSL1B_0704300_22_20070829_0904_0913_V02
FIRE_EMPHASIZING_ENHANCED_CLEAN_TIF_AMSL1B_0704400_04_20070908_0425_0430_V02
FIRE_EMPHASIZING_ENHANCED_CLEAN_TIF_AMSL1B_0704400_05_20070908_0436_0444_V02
FIRE_EMPHASIZING_ENHANCED_CLEAN_TIF_AMSL1B_0704400_06_20070908_0447_0453_V02
FIRE_EMPHASIZING_ENHANCED_CLEAN_TIF_AMSL1B_0704400_07_20070908_0457_0503_V02
FIRE_EMPHASIZING_ENHANCED_CLEAN_TIF_AMSL1B_0704400_08_20070908_0537_0542_V02
FIRE_EMPHASIZING_ENHANCED_CLEAN_TIF_AMSL1B_0704400_09_20070908_0631_0636_V02
FIRE_EMPHASIZING_ENHANCED_CLEAN_TIF_AMSL1B_0704400_10_20070908_0638_0644_V02
FIRE_EMPHASIZING_ENHANCED_CLEAN_TIF_AMSL1B_0704400_11_20070908_0647_0654_V02
FIRE_EMPHASIZING_ENHANCED_CLEAN_TIF_AMSL1B_0704400_12_20070908_0823_0828_V02
FIRE_EMPHASIZING_ENHANCED_CLEAN_TIF_AMSL1B_0704400_15_20070908_1129_1134_V02
FIRE_EMPHASIZING_ENHANCED_CLEAN_TIF_AMSL1B_0704400_16_20070908_1137_1146_V02
FIRE_EMPHASIZING_ENHANCED_CLEAN_TIF_AMSL1B_0704400_17_20070908_1307_1315_V02"""
CORRUPTED_AMS_FILES = CORRUPTED_AMS_FILES.split('\n')
CORRUPTED_AMS_FILES = [file + '.tif' for file in CORRUPTED_AMS_FILES]
CORRUPTED_AMS_FILES = [file[17:] for file in CORRUPTED_AMS_FILES]



radiance_scaling_factors = {
    "AMSL1B_0700103_01_20061028_0048_0056_V02.hdf": np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01]),
    "AMSL1B_0700202_07_20070602_1850_1854_V02.hdf": np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01]),
    "AMSL1B_0704200_01_20070816_1450_1456_V02.hdf": np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01]),
    "AMSL1B_0704300_01_20070829_0152_0157_V02.hdf": np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01]),
    "AMSL1B_0704400_19_20070908_1452_1458_V02.hdf": np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01]),
    "AMSL1B_0805000_10_20071024_1840_1846_V02.hdf": np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01]),
    "AMSL1B_0805100_13_20071025_1935_1939_V02.hdf": np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01]),
    "AMSL1B_0805200_30_20071026_2228_2235_V02.hdf": np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01]),
    "AMSL1B_0805300_11_20071028_1816_1825_V02.hdf": np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01]),
    "AMSL1B_0805300_18_20071028_1914_1918_V02.hdf": np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01]),
    "AMSL1B_0806600_13_20080708_1757_1807_V02.hdf": np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01]),
    "AMSL1B_0806600_32_20080708_2244_2247_V02.hdf": np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01]),
    "AMSL1B_0806700_07_20080719_1858_1903_V02.hdf": np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01]),
    "AMSL1B_0807000_01_20080919_1552_1555_V01.hdf": np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01]),
    "AMSL1B_1180104_05_20110722_2000_2003_V01.hdf": np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]),
    "AMSL1B_1180107_02_20110726_2234_2244_V01.hdf": np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]),
    "AMSL1B_1280104_02_20111018_1738_1741_V01.hdf": np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]),
    "AMSL1B_1280105_05_20111019_2128_2132_V01.hdf": np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]),
    "MASTERL1B_1662900_12_20160617_2204_2219_V01.hdf": np.array([0.150, 0.150, 0.150, 0.150, 0.150, 0.150, 0.150, 0.150, 0.150, 0.150,
                                                                0.150, 0.010, 0.010, 0.010, 0.010, 0.010, 0.010, 0.010, 0.010, 0.010,
                                                                0.010, 0.010, 0.010, 0.010, 0.010, 0.100, 0.005, 0.006, 0.006, 0.006,
                                                                0.006, 0.005, 0.009, 0.009, 0.006, 0.006, 0.005, 0.005, 0.008, 0.006,
                                                                0.010, 0.010, 0.010, 0.010, 0.010, 0.010, 0.010, 0.010, 0.010, 0.010
                                                            ]),
    "MASTERL1B_1981718_09_20190804_0020_0023_V01.hdf": np.array([0.100, 0.100, 0.100, 0.100, 0.100, 0.100, 0.100, 0.100, 0.100, 0.100,
                                                                0.100, 0.018, 0.015, 0.015, 0.015, 0.010, 0.010, 0.010, 0.010, 0.010,
                                                                0.010, 0.010, 0.010, 0.010, 0.010, 0.100, 0.010, 0.010, 0.010, 0.010,
                                                                0.010, 0.010, 0.010, 0.010, 0.010, 0.010, 0.010, 0.010, 0.010, 0.010,
                                                                0.010, 0.010, 0.010, 0.010, 0.010, 0.010, 0.010, 0.010, 0.010, 0.010
                                                            ])
}

band_mappings = {
    12: np.array(list(range(1,13))),
    16: np.array(list(range(1, 13))),
    50: np.array([1, 2, 3, 4, 5, 6, 7, 8, 15, 22, 30, 47]),

}

K_to_C = 273.15
MIN_TEMP = 250
MAX_TEMP = 500

def get_radiance(ams_file, filename):
    scaled_radiance = np.asarray(ams_file.select('CalibratedData'), dtype=np.float64)
    scaling_factors = radiance_scaling_factors[filename]
    radiance = scaled_radiance * scaling_factors[None, :, None]
    radiance = np.clip(radiance, 0, 10000)
    return radiance

def get_image_basename(img):
    return '_'.join(os.path.basename(img).split('_')[4:])


def get_image_band(img):
    return int(os.path.basename(img).split('_')[3])

def get_geotransform(flat_lat, flat_long, res):
    ymin, ymax = min(flat_lat), max(flat_lat)
    xmin, xmax = min(flat_long), max(flat_long)
    geotransform = (xmin, res, 0, ymax, 0, -res)
    return geotransform


def get_new_grid(flat_lat, flat_long, res):
    ymin, ymax = min(flat_lat), max(flat_lat)
    xmin, xmax = min(flat_long), max(flat_long)
    xi = np.arange(xmin, xmax, res)
    yi = np.arange(ymin, ymax, res)
    xi, yi = np.meshgrid(xi, yi)
    return (xi, yi)


def interpolate_image(image, old_long_lat_grid, new_long_lat_grid):
    interp_data = []
    for channel in image:
        channel = channel.flatten()
        zi = scipy.interpolate.griddata(old_long_lat_grid, channel, new_long_lat_grid, method='nearest', fill_value=-1)
        zi = zi[::-1, :]
        interp_data.append(zi)

    return np.array(interp_data)


def write_data_out(interp_data, out_data):
    for i in range(len(interp_data)):
        out_band = out_data.GetRasterBand(i + 1)
        out_band.WriteArray(interp_data[i], 0, 0)
        out_band.FlushCache()
        out_band.SetNoDataValue(-1)


def get_outside_points(lats, longs):
    new_long = np.array([])
    new_lat = np.array([])

    new_long = np.append(new_long, longs[0])
    new_lat = np.append(new_lat, lats[0])

    lats = lats.T
    longs = longs.T
    new_long = np.append(new_long, longs[0])
    new_lat = np.append(new_lat, lats[0])

    lats = lats.T
    longs = longs.T
    new_long = np.append(new_long, np.flip(longs[len(longs) - 1]))
    new_lat = np.append(new_lat, np.flip(lats[len(lats) - 1]))

    lats = lats.T
    longs = longs.T
    new_long = np.append(new_long, np.flip(longs[len(longs) - 1]))
    new_lat = np.append(new_lat, np.flip(lats[len(lats) - 1]))

    new_latlong = np.array(list(zip(new_long, new_lat)))

    return new_latlong


def crop_to_hdf(lats, longs, tif_file):
    shell_points = get_outside_points(lats, longs)
    hull = Polygon(shell_points)
    hull_pts = list(hull.exterior.coords)
    hull_geom = [{'type': 'Polygon', 'coordinates': [hull_pts]}]

    with rasterio.open(tif_file) as src:
        out_image, out_transform = mask(src, hull_geom, crop=True)
    out_meta = src.meta.copy()

    out_image[out_image == -1] = 0

    # save the resulting raster
    out_meta.update({"driver": "GTiff",
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "transform": out_transform})

    cleaned_out_file = f'{os.path.dirname(tif_file)}/CLEANTIF_{os.path.basename(tif_file)}'

    with rasterio.open(cleaned_out_file, 'w', **out_meta) as dest:
        dest.write(out_image)

    os.remove(tif_file)

    return cleaned_out_file

def process_ams(ams_image, out_path, overwrite=False):
    np.set_printoptions(precision=2)
    out_file = out_path + "TIF_" + ntpath.basename(ams_image).split(".")[0] + ".tif"
    if os.path.exists(out_file) and not overwrite:
        print('found')
        return
    if os.path.exists(out_file) and overwrite:
        os.remove(out_file)

    normalization_constants = {}

    file = SD(ams_image, SDC.READ)
    radiance = get_radiance(file, os.path.basename(ams_image))
    radiance = np.where(radiance < 0, 0.001, radiance)
    # print('radiance range')
    # print(np.min(radiance, axis = (0,2)), )
    # print(np.max(radiance, axis = (0,2)), )

    solar_irradiance = np.asarray(file.select('SolarSpectralIrradiance'), dtype=np.float64)
    solar_zenith = np.radians(np.asarray(file.select('SolarZenithAngle'), dtype=np.float64))

    apparent_reflectance = (radiance * np.pi) / (solar_irradiance[None, :, None] * np.cos(solar_zenith[:, None, :]))
    if np.max(apparent_reflectance) > 1:
        warnings.warn('Reflectance in ' + ams_image + ' is greater than 1:' )
        print("min: ", np.min(apparent_reflectance, axis=(0,2)))
        print("max: ", np.max(apparent_reflectance, axis=(0,2)))
        print("mean: ", np.mean(apparent_reflectance, axis=(0,2)))

    c = 3e8
    h = 6.626e-34
    k = 1.38e-23
    lam = np.asarray(file.select('EffectiveCentralWavelength_IR_bands'), dtype=np.float64) * 1e-6
    IR_bands = lam > 0
    if np.sum(IR_bands) not in [2,4,25]:
        print('Issue with identifying IR', ams_image)
    temp_correction_slope = np.asarray(file.select('TemperatureCorrectionSlope'), dtype=np.float64)
    temp_correction_intercept = np.asarray(file.select('TemperatureCorrectionIntercept'), dtype=np.float64)

    brightness_temp = (h * c / k) / (lam[None, :, None] * np.log(2 * h * c ** 2 / (1e6 * radiance * np.power(lam[None, :, None], 5)) + 1))
    brightness_temp = (brightness_temp - temp_correction_intercept[None, :, None]) / temp_correction_slope[None, :, None]


    normalized_radiance = radiance / solar_irradiance[None, :, None] * 2 # arbitrary scaling factor
    print('Radiance Clipping Percentage by channel')
    print(np.sum(normalized_radiance > 1, axis = (0,2)) / np.sum(normalized_radiance > 0, axis = (0,2)) * 100)
    normalized_radiance = np.clip(normalized_radiance, 0, 1)

    brightness_temp = np.clip(brightness_temp, MIN_TEMP, MAX_TEMP)
    normalized_brightness_temp = (brightness_temp - MIN_TEMP) / (MAX_TEMP - MIN_TEMP)

    normalized_output_image = np.where(IR_bands[None, :, None], normalized_brightness_temp, normalized_radiance)
    if np.max(normalized_output_image) > 1:
        print('Normalization Issue With', ams_image)
        return
    else:
        print('Normalized Image Range')
        print("min: ", np.min(normalized_output_image, axis=(0,2)))
        print("max: ", np.max(normalized_output_image, axis=(0,2)))
        print("mean: ", np.mean(normalized_output_image, axis=(0,2)))


    output_image = np.where(IR_bands[None, :, None], brightness_temp, radiance)
    if np.isnan(output_image).any():
        print('NaNs in output image', ams_image)
        return


    normalized_output_image = np.swapaxes(normalized_output_image, 0, 1) # move the channel to first dimension
    output_image = np.swapaxes(output_image, 0, 1) # move the channel to first dimension

    channels = output_image.shape[0]

    import copy
    bands = copy.deepcopy(band_mappings[channels])
    bands -= 1

    normalized_output_image = normalized_output_image[bands]
    output_image = output_image[bands]

    normalization_constants['radiance'] = (solar_irradiance[bands]).tolist()
    normalization_constants['temperature'] = {'slope': 1/(MAX_TEMP - MIN_TEMP), 'intercept': -MIN_TEMP/(MAX_TEMP - MIN_TEMP)}
    normalization_constants['IR_bands'] = IR_bands.tolist()

    lats = np.asarray(file.select('PixelLatitude'))
    longs = np.asarray(file.select('PixelLongitude'))
    flat_lat = lats.flatten()
    flat_long = longs.flatten()

    geotransform = get_geotransform(flat_lat, flat_long, FINAL_RESOLUTION)
    new_long_lat_grid = get_new_grid(flat_lat, flat_long, FINAL_RESOLUTION)

    normalized_output_image = interpolate_image(normalized_output_image, (flat_long, flat_lat), new_long_lat_grid)
    output_image = interpolate_image(output_image, (flat_long, flat_lat), new_long_lat_grid)

    # define output tiff
    template = gdal.Open("./data/ams_data/template.tif")
    driver = template.GetDriver()

    out_data = driver.Create(out_file, normalized_output_image[0].shape[1], normalized_output_image[0].shape[0], len(bands), GDT_Byte)

    write_data_out(normalized_output_image*255, out_data)

    out_data.SetGeoTransform(geotransform)
    out_data.SetProjection(template.GetProjection())

    del template
    del out_data

    cleaned_out_file = crop_to_hdf(lats, longs, out_file)

    return normalization_constants

def extract_all_bands(hdf_files, out_folder, overwrite=False):
    out_path = f'{out_folder}/'
    normalization_constants = {}

    for i, ams_image in enumerate(hdf_files):
        print("Processing image " + str(i + 1) + "/" + str(len(hdf_files)))
        print("Creating ", out_path)
        constants = process_ams(ams_image, out_path, overwrite=overwrite)
        normalization_constants[ams_image] = constants

    with open(out_path + 'normalization_constants.json', 'w') as f:
        json.dump(normalization_constants, f)


##################################################################################################################
# This code is used to generate patches of the AMS data. This was for testing and development.
##################################################################################################################

def check_corruption(filename):
    for f in CORRUPTED_AMS_FILES:
        if f in filename:
            return True
    return False


def extract_patches(filename):
    img = dataset_utils.load_tif_image(filename, [], all_bands=True)
    patches = sklearn.feature_extraction.image.extract_patches_2d(img, (256, 256), max_patches=300)
    return patches


def determine_patch_validity(patch):
    if np.sum(patch <= 10) > 0.1 * np.sum(patch >= 0):
        return False
    return True


def split_image(filename, patch_folder):
    patches = extract_patches(filename)
    patch_num = 0
    for i in range(patches.shape[0]):
        patch = patches[i]
        if determine_patch_validity(patch):
            new_name = f'{patch_folder}/PATCH_{patch_num}_{os.path.basename(filename)}'
            tifffile.imwrite(new_name, patch)
            patch_num += 1


def generate_ams_patches(orig_folder, patch_folder):
    complete_images = glob.glob(f'{orig_folder}/*.tif')
    for img in complete_images:
        if not check_corruption(img):
            split_image(img, patch_folder)




