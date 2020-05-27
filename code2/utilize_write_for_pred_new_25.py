# -*- coding:utf-8 -*-
import os
import re
import json
import pandas as pd
import numpy as np
from numpy.core.multiarray import ndarray
from scipy import ndimage as ndi
import pydicom as dicom
import scipy.misc
from dicompylercore import dicomparser
from matplotlib import path
from tensorflow.keras import models
from tensorflow.keras import losses
import tensorflow as tf
from tensorflow.keras import callbacks


#-----------------------------------------------------------------------------
#                          The shape of every organ
#
#-----------------------------------------------------------------------------
#  ORGAN           |     stage1      |      stage2      |     center
#-----------------------------------------------------------------------------
# Mandible:            [320,320,192] -> [120, 136, 96]    [250,250,119]
# Spinal cord:         [160,160,256] -> [80,80,256]       [The center of array]
# Right Tm-joint       [320,320,192] -> [48, 48, 24]      [250,250,119]
# Right eye            [320,320,192] -> [32, 32, 48]      [250,250,119]
# Right lens           [320,320,192] -> [16, 16, 16]      [250,250,119]
# Right optic nerve    [320,320,192] -> [48, 32, 16]      [250,250,119]
# R-parotid            [320,320,192] -> [56, 56, 80]      [250,250,119]
# R-temporal lobe      [320,320,192] -> [120, 64, 56]     [250,250,119]
# Optic chiasm         [320,320,192] -> [48, 48, 16]      [250,250,119]
# Brain stem           [320,320,192] -> [48, 48, 72]      [250,250,119]
# Left lens            [320,320,192] -> [16, 16, 16]      [250,250,119]
# Left optic nerve     [320,320,192] -> [48, 32, 16]      [250,250,119]
# Left Tm-joint        [320,320,192] -> [48, 48, 24]      [250,250,119]
# L-parotid            [320,320,192] -> [56, 56, 80]      [250,250,119]
# L-temporal lobe      [320,320,192] -> [120, 64, 56]     [250,250,119]
# Left eye             [320,320,192] -> [32, 32, 48]      [250,250,119]
# new
# Eyes_nerves_chiasm   [320,320,192] -> [120,120,16]      [250,250,119]
# Eyes_nerves          [320,320,192] -> [120,120,16]      [250,250,119]



Organ_config_stage1 = {
    'default':           [320,320,192],
    'Mandible':          [320,320,192],
    'Spinal cord':       [160,160,256],
    'Right TM-joint':    [320,320,192],
    'Right eye':         [320,320,192],
    'Right lens':        [320,320,192],
    'Right optic nerve': [320,320,192],
    'R-parotid':         [320,320,192],
    'R-temporal lobe':   [320,320,192],
    'Optic chiasm':      [320,320,192],
    'Brain stem':        [320,320,192],
    'Left lens':         [320,320,192],
    'Left optic nerve':  [320,320,192],
    'Left TM-joint':     [320,320,192],
    'L-parotid':         [320,320,192],
    'L-temporal lobe':   [320,320,192],
    'Left eye':          [320,320,192]

}





#将坐标确定下来

'''
Organ_config_stage2 = {
    'Mandible':          [120, 160, 112],
    'Spinal cord':          [88,88,256],
    'Right TM-joint':      [72, 72, 48],
    'Right eye':           [56, 56, 72],
    'Right lens':          [40, 40, 40],
    'Right optic nerve':   [72, 56, 40],
    'R-parotid':           [80, 80, 104],
    'R-temporal lobe':    [144, 88, 80],
    'Optic chiasm':        [72, 72, 40],
    'Brain stem':          [72, 72, 96],
    'Left lens':           [40, 40, 40],
    'Left optic nerve':    [72, 56, 40],
    'Left TM-joint':       [72, 72, 48],
    'L-parotid':           [80, 80, 104],
    'L-temporal lobe':    [144, 88, 80],
    'Left eye':            [56, 56, 72],
    'Eyes_nerves_chiasm':  [144,144,40],
    'Eyes_nerves':         [144,144,40]
}

Organ_config_stage2 = {
    'Mandible':          [104, 144, 96],
    'Spinal cord':          [72,72,240],
    'Right TM-joint':      [56, 56, 32],
    'Right eye':           [40, 40, 56],
    'Right lens':          [24, 24, 24],
    'Right optic nerve':   [56, 40, 24],
    'R-parotid':           [48, 48, 88],
    'R-temporal lobe':    [128, 72, 64],
    'Optic chiasm':        [56, 56, 24],
    'Brain stem':          [56, 56, 80],
    'Left lens':           [24, 24, 24],
    'Left optic nerve':    [56, 40, 24],
    'Left TM-joint':       [56, 56, 32],
    'L-parotid':           [64, 64, 88],
    'L-temporal lobe':    [128, 72, 64],
    'Left eye':            [40, 40, 56]
}
'''
Organ_config_stage2 = {
    'Brain stem':          [48, 48, 72],
    'L-parotid':           [56, 56, 80],
    'L-temporal lobe':    [120, 64, 56],
    'Left TM-joint':       [48, 48, 24],
    'Left eye':            [32, 32, 48],
    'Left lens':           [16, 16, 16],
    'Left optic nerve':    [48, 32, 16],
    'Mandible':          [120, 136, 96],
    'R-parotid':           [56, 56, 80],
    'R-temporal lobe':    [120, 64, 56],
    'Right TM-joint':      [48, 48, 24],
    'Right eye':           [32, 32, 48],
    'Right lens':          [16, 16, 16],
    'Right optic nerve':   [48, 32, 16],
    'Spinal cord':          [80,80,256],
    'Optic chiasm':        [48, 48, 16]
}
'''
Organ_config_stage2={#这里的参数是x,y,z加上第一次裁剪的初始x，y，z，三个数字
    "Brain stem":[120, 168, 160, 208, 56, 128, 90, 90, 23],
    "L-parotid":[172, 228, 225, 281, 95, 175, 90, 90, 23],
    #这块只能按照模型的大小来，模型的大小是以前训练的，所以只能用以前的，当然会产生一定的误差。有符号的是后来改的，但是模型没有训练。x,y是与np的图像相反的，np[y,x,z]!!!
    #"L-temporal lobe":[156, 284, 194, 258, 19, 91],
    "L-temporal lobe":[160, 280, 194, 258, 27, 83, 90, 90, 23],
    "Left TM-joint":[169, 217, 214, 262, 85, 109, 90, 90, 23],
    "Left eye":[134, 166, 199, 231, 33, 81, 90, 90, 23],
    "Left lens":[117, 133, 211, 227, 59, 75, 90, 90, 23],
    "Left optic nerve":[65, 113, 191, 223, 54, 70, 90, 90, 23],
    "Mandible":[91, 211, 118, 254, 36, 132, 90, 90, 23],
    "R-parotid":[167, 223, 93, 149, 94, 174, 90, 90, 23],
    #"R-temporal lobe":[155, 283, 117, 181, 19, 91],
    "R-temporal lobe":[159, 279, 194, 258, 27, 83, 90, 90, 23],
    "Right TM-joint":[169, 217, 112, 160, 85, 109, 90, 90, 23],
    "Right eye":[134, 166, 141, 173, 34, 82, 90, 90, 23],
    "Right lens":[117, 133, 145, 161, 59, 75, 90, 90, 23],
    "Right optic nerve":[65, 113, 148, 180, 54, 70, 90, 90, 23],
    "Spinal cord":[14, 94, 44, 124, 0, 256,195,195,0],
    "Optic chiasm":[91, 139, 161, 209, 58, 74, 90, 90, 23]
}

'''


#TODO : To update the algorithms of in_the_ploy

################Load all the data####################################
def load_scan(path):
    CTpath = os.listdir(path)
    slices = [dicom.read_file(path + '/' + s) for s in CTpath]
    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))
    return slices

def Get_path(PathDicom):
    IstFilesDCM = []
    for dirName, subdirList, fileList in os.walk(PathDicom):
        for filename in fileList:
            if ".dcm" in filename.lower() and re.match('CT',filename):
                IstFilesDCM.append(os.path.join(dirName,filename))
    return IstFilesDCM

def load_CT_dcm(path):
    slices = [dicom.read_file(s) for s in path]
    slice_dict = Get_Frame_UID_Dict(slices)
    return slice_dict
'''
def load_RS_dcm(path):
    slices = {}
    for file in path:
        Contour_dcm = dicom.dcmread(file)
        slices[Contour_dcm[0x30060010][0][0x00200052]] = Contour_dcm
    return slices
'''
def load_Contour_dcm(path):
    slices = {}
    for file in path:
        Contour_dcm = dicomparser.DicomParser(file)
        slices[Contour_dcm.GetFrameOfReferenceUID()] = Contour_dcm
    return slices

def get_pixels_hu(slices):
    # get the pixels
    image = np.stack([s.pixel_array for s in slices],axis = -1)
    image = image.astype(np.int16)
    image[image == -2000] = 0
    for slice_number in range(len(slices)):
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        if slope != 1:
            image[:,:,slice_number] = slope * image[:,:,slice_number].astype(np.float64)
            image[:,:,slice_number] = image[:,:,slice_number].astype(np.int16)
        image[:,:,slice_number] += np.int16(intercept)
    return np.array(image, dtype=np.int16)
def get_sop_UID_Dict(slices):
    #get SOP UID
    slices = [dicom.read_file(s) for s in path]
    sop_UID_Dict={}
    sop_UID=[]
    position_list=[]
    
    for s in slices:
        sop_UID_Dict[s[0x0027,0x0044].value]=s.SOPInstanceUID
        position_list.append(s[0x0027,0x0044].value)
    min=min(position_list)
    max=max(position_list)
    return min, sop_UID_Dict

def Get_Frame_UID_Dict(slices):
    # get series_UID and sort the dicom file in the list
    FUID_Dict = {}
    FrameUID = []
    for s in slices:
        if s.FrameOfReferenceUID not in FrameUID:
            FrameUID.append(s.FrameOfReferenceUID)
            FUID_Dict[s.FrameOfReferenceUID] = []
        FUID_Dict[s.FrameOfReferenceUID].append(s)
    for key in FUID_Dict.keys():
        slice_list = FUID_Dict[key]
        slice_list.sort(key = lambda x: int(x.ImagePositionPatient[2]))
        FUID_Dict[key] = slice_list
    return FUID_Dict

######################Processing functions ################################
def inpolygon(maskData,pointData):
# to check wether the point is in the polygon which includes the boundaries.
    p = path.Path(pointData)
    x,y = np.shape(maskData)
    for i in range(x):
        for j in range(y):
            if p.contains_point([i,j],radius = 0.1):
                maskData[i,j]=1
    return maskData


def Index_of_changed_points(scan):
    thickness_index = []
    thickness = [scan[0].SliceThickness]
    thic = 0
    for lis in scan:
        if lis.SliceThickness !=  scan[thic].SliceThickness:
            if 0 not in thickness_index:
                thickness_index.append(0)
            thickness_index.append(scan.index(lis))
            thickness.append(lis.SliceThickness)
        thic = scan.index(lis)
    if thickness_index:
        thickness_index.append(scan.index(lis)+1)
        thickness.append(lis.SliceThickness)
        return thickness,thickness_index
    else:
        return thickness,thickness_index







def interpolate_thickness_by_thickness_orgin(image,PixelSpacing,new_spacing,order=3):
    preprocessed_slice = []

    img = interpolate_orgin(image[:,:,:],PixelSpacing, new_spacing,order=order)
    preprocessed_slice.append(img)
    return np.concatenate(preprocessed_slice,axis=2)

def interpolate_orgin(image, PixelSpacing, new_spacing =[1,1,1], order=3):
    # interpolate the image with bicubic
    spacing = map(float, (list(PixelSpacing)+[1]))
    spacing = np.array(list(spacing))
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest',order = order)
    return image



def resample_write(path_mask,image, scan, new_spacing=[1,1,1]):#这里的resample我把插值为1的npy和没有插值的npy一起输出了，目的是为了把原始npy的最高点图层与插值之后的对应。因为我发现原始CT用第一张CT的position减去最后一张的有的不等于插值之后的图层数，这导致写入发生一定的偏差。
    # resampling the image and the mask at the same time
    thickness, thickness_index = Index_of_changed_points(scan)
    PatientID = scan[0].PatientID
    Date = scan[0].StudyDate
    PixelSpacing = scan[0].PixelSpacing
    PatientInfo = PatientID+Date
    x_coord=scan[0][0x0027,0x1042].value
    y_coord=scan[0][0x0027,0x1043].value
    position_min=scan[0][0x0027,0x1044].value
    position_max=scan[-1][0x0027,0x1044].value
    if thickness_index:
        preprocessed_slice = interpolate_thickness_by_thickness(image,thickness_index,thickness,PixelSpacing,new_spacing)
        preprocessed_slice = np.flip(preprocessed_slice, 2)
        preprocessed_slice_orgin = interpolate_thickness_by_thickness_orgin(image,PixelSpacing,new_spacing)
        preprocessed_slice_orgin = np.flip(preprocessed_slice_orgin, 2)
        save_data(path_mask+'/CT/'+PatientID+Date,preprocessed_slice)
        save_data(path_mask+'/CT_orgin/'+PatientID+Date,preprocessed_slice_orgin)
        print(': '+PatientID+Date+'已写入')
        return new_spacing,thickness_index,thickness,PixelSpacing,PatientInfo,x_coord,y_coord,position_min,position_max

    else:
        preprocessed_slice = interpolate(image,PixelSpacing,thickness[0],new_spacing)
        preprocessed_slice = np.flip(preprocessed_slice, 2)
        preprocessed_slice_orgin = interpolate_thickness_by_thickness_orgin(image,PixelSpacing,new_spacing)
        preprocessed_slice_orgin = np.flip(preprocessed_slice_orgin, 2)
        save_data(path_mask+'/CT/'+PatientID+Date,preprocessed_slice)
        save_data(path_mask+'/CT_orgin/'+PatientID+Date,preprocessed_slice_orgin)
        print(': '+PatientID+Date+'已写入')
        return new_spacing, thickness_index,thickness,PixelSpacing,PatientInfo,x_coord,y_coord,position_min,position_max



def interpolate_thickness_by_thickness(image,thickness_index,thickness,PixelSpacing,new_spacing,order=3):
    preprocessed_slice = []
    for i in range(len(thickness_index)):
        if i == len(thickness_index) -1:
            break
        else:
            img = interpolate(image[:,:,thickness_index[i]:(thickness_index[i+1]-1)],PixelSpacing,thickness[i], new_spacing,order=order)
            preprocessed_slice.append(img)
    return np.concatenate(preprocessed_slice,axis=2)

def interpolate(image, PixelSpacing, SliceThickness, new_spacing =[1,1,1], order=3):
    # interpolate the image with bicubic
    spacing = map(float, (list(PixelSpacing)+ [SliceThickness]))
    spacing = np.array(list(spacing))
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest',order = order)
    return image


# --------------------------------------------------------------------------
#                                Before Stage 1
# --------------------------------------------------------------------------




#######下面的不用了，我把crop和cut放进organ_based_mask_left_eye中了
'''
def crop(array, new_shape):#crop是把患者的脑袋全部放进来，并减少空白位置，从0开始的是从脑袋上面。也就是从position最大值开始
    """
    input:
        filename: the .npy file
        new_shape: the list of wanted shape Eg. [[100,400],[100,400],[50,100]], 3 dimensions, with its start and end.
    """

    new_shape = np.array(new_shape)
    shape = [new_shape[i, 1] - new_shape[i, 0] for i in range(2)]
    new_array = np.zeros(shape)
    new_array = array[new_shape[0, 0]:new_shape[0, 1],
                new_shape[1, 0]:new_shape[1, 1],
                :]
    return new_array


def check_pixel_number(array,threshold, axis='0'):
    if axis == '0':
        if array.shape[0]>threshold:
            return True
    elif axis == '1':
        if array.shape[1]>threshold:
            return True
    elif axis == '2':
        if array.shape[2]>threshold:
            return True


# TODO: delete the np.flip or flip at the very first time.


def crop_array(path_mask):
    """
    To crop the target part of an mask

        :param maskdir: the mask directory
        :param organ: to determine what which put to be used as step 1's training data
        :return: the cropped array
    """
     # TODO: maybe new_shape should be used as a parameter
    maski_file_list = os.listdir(path_mask + '/CT')
    for file in maski_file_list:
        if '.npy' in file:
            CT = np.load(path_mask + '/CT' + '/' + file)
            l,u,f=center_coor(CT)
            shape = [[l-30, l+250], [u-100, u+100]]
            croped_CT = crop(CT, shape)
            save_data(path_mask + '/crop/' + file, croped_CT)
        else:
            continue

'''
'''
def shrink(array, shape=[64, 64, 32]):
    """
    To shrink the very first array to a typical shape for the stage 1 network  a.k.a the Loc_net

    :param array: The Mask or CT array
    :param shape: The target shape for Loc_net
    :return: the array with target shape
    """
    real_resize_factor = np.array(shape) / array.shape
    image = scipy.ndimage.interpolation.zoom(array, real_resize_factor, mode='nearest', order=0)
    return image.reshape([image.shape[0], image.shape[1], image.shape[2], 1])

#   -----------------------------------
#   The l2r u2d f2b functions are used
#   to locate the start point and end
#   point for each given dimension.
#   -----------------------------------

def shrink_array(path_mask,Organ_path,organ = 'default'):
    """
    The function to shrink the arrays in both CT and mask file to the shrink/mask/CT/file
    :param mask_dir: the directory of the mask file, just the main directory end to the organ's name
                e.g. '/public/home/liulizuo/TrainData/Mandible'
    :return: None all the data will be saved at the subdirectory named shrink
    """
    mask_file_list = os.listdir(path_mask + '/CT')
    for file in mask_file_list:
        if '.npy' in file:
            CT = np.load(path_mask + '/CT/' + file)
            shrink_CT = shrink(CT)
            save_data(Organ_path + '/shrink/' + file, shrink_CT)
        else:
            continue

'''
# --------------------------------------------------------------------------
#                                After Stage 1
# --------------------------------------------------------------------------


#--------------------------------------------------------------------------
#                                Before stage2's training
#--------------------------------------------------------------------------



'''
#直接调用坐标进行切割
def cut_coord_shape(l,u,f,organ='default'):
    
    Organ_mask_coor_stage2={
        "Brain stem":[120, 168, 160, 208, 56, 128],
        "L-parotid":[172, 228, 225, 281, 95, 175],
        #这块只能按照模型的大小来，模型的大小是以前训练的，所以只能用以前的，当然会产生一定的误差。有符号的是后来改的，但是模型没有训练。x,y是相反的np[y,x,z]
        #"L-temporal lobe":[156, 284, 194, 258, 19, 91],
        "L-temporal lobe":[160, 280, 194, 258, 27, 83],
        "Left TM-joint":[169, 217, 214, 262, 85, 109],
        "Left eye":[52, 84, 199, 231, 33, 81],
        "Left lens":[117, 133, 211, 227, 59, 75],
        "Left optic nerve":[65, 113, 191, 223, 54, 70],
        "Mandible":[91, 211, 118, 254, 36, 132],
        "R-parotid":[167, 223, 93, 149, 94, 174],
        #"R-temporal lobe":[155, 283, 117, 181, 19, 91],
        "R-temporal lobe":[159, 279, 194, 258, 27, 83],
        "Right TM-joint":[169, 217, 112, 160, 85, 109],
        "Right eye":[53, 85, 141, 173, 34, 82],
        "Right lens":[117, 133, 145, 161, 59, 75],
        "Right optic nerve":[65, 113, 148, 180, 54, 70],
        "Spinal cord":[14, 94, 44, 124, 0, 256],
        "Optic chiasm":[91, 139, 161, 209, 58, 74]}
        
    Organ_mask_coor_stage2={#这里的参数是x,y,z加上第一次裁剪的初始x，y，z，三个数字
        "Brain stem":[93, 141, -23, 25 ,-58, 14],
        "L-parotid":[72, 128, 31, 87,-13, 67],
        "L-temporal lobe":[65, 185, 8, 72 ,-65, -9],
        "Left TM-joint":[72, 120, 30, 78 ,-20, 4],
        "Left eye":[20, 52, 18, 50 ,-58, -10],
        "Left lens":[22, 38, 27, 43 ,-41, -25],
        "Left optic nerve":[38, 86, 7, 39 ,-43, -27],
        "Mandible":[17, 137, -67, 69 ,-11, 85],
        "R-parotid":[71, 127, -88, -32 ,-13, 67],
        "R-temporal lobe":[60, 180, -74, -10,-65, -9],
        "Right TM-joint":[69, 117, -79, -31 ,-20, 4],
        "Right eye":[22, 54, -43, -11 ,-58, -10],
        "Right lens":[20, 36, -35, -19 ,-41, -25],
        "Right optic nerve":[40, 88, -33, -1 ,-43, -27],
        "Spinal cord":[-17, 63, -87, -7 ,-32, 184],
        "Optic chiasm":[65, 113, -22, 26 ,-43, -27]}


    forward=Organ_mask_coor_stage2[organ][0]+30
    backward=Organ_mask_coor_stage2[organ][1]+30
    left=Organ_mask_coor_stage2[organ][2]+100
    right=Organ_mask_coor_stage2[organ][3]+100
    down=Organ_mask_coor_stage2[organ][4]+f
    up=Organ_mask_coor_stage2[organ][5]+f

    return forward, backward, left, right, down, up 

def cut(CT, x, l,u,f,organ='default'):
    """
    To train the stage 2 net, the feeds of net will be calculated here,
    :param mask: mask array
    :param CT: CT array
    :param organ: to determine the shape
    :return: the mask and CT can be used to feed the net
    """
    
    shape = x[organ]
    a, b = l2r(mask)
    c, d = u2d(mask)
    e, f = f2b(mask)
    a_center, c_center, e_center = int((a + b) / 2), int((c + d) / 2), int((e + f) / 2)
    temp_mask = np.zeros(x[organ])
    temp_CT = np.zeros(x[organ])
    forward, backward = locate(mask.shape[0], shape[0], a_center)
    left, right = locate(mask.shape[1], shape[1], c_center)
    down, up = locate(mask.shape[2], shape[2], e_center)
    
    temp_CT = np.zeros(x[organ])
    shape_ct=CT.shape[2]
    forward, backward,left, right, down, up = cut_coord_shape(l,u,f,organ)
    if organ=='Spinal cord':

        down=shape_ct-256
        up=shape_ct
    temp_CT[:, :, :] = CT[forward:backward, left:right, down:up]#x,y是相反的np[y,x,z]!!
    return temp_CT.reshape(x[organ] + [1])
def center_coor(a):

    l = l2r(a)
    u=u2d(a,l)
    f=f2b(a,l,u)
    return l,u,f

def l2r(array):
    index = []
    for i in range(array.shape[0]):
        if np.max(array[i, :, :]) > 0.5:
            index.append(i)
    return index[0]

def u2d(array,x):
    index = []
    for i in range(array.shape[1]):
        if np.max(array[x, i, :]) > 0.5:
            index.append(i)
    return index[0]


def f2b(array,x,y):
    index = []
    for i in range(array.shape[2]):
        if np.max(array[x, y, i]) > 0.5:
            index.append(i)
    return index[0]



def cut_array(path_mask,Organ_path,organ='default'):
    """
    The main function to cut the array, no return, but saving the data in ~/cutted/
    :param maskdir: just like the mask_dir in the function shrink()
    """

    maskfilelist = os.listdir(path_mask + '/crop')
    for file in maskfilelist:
        if '.npy' in file:
            CT = np.load(path_mask + '/crop/' + file)

            l,u,f=center_coor(CT)
            cutted_CT= cut(CT,Organ_config_stage2,l,u,f,organ)


            save_data(Organ_path + '/cutted/' + file, cutted_CT)
        else:
            continue


def bce_dice_loss(y_true, y_pred):
    loss = losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss
def dice_coeff(y_true, y_pred):
    smooth = 1.
    # Flatten
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss
def pred_mask(Organ_path,organ='default'):
    key=organ
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    save_model_path = '/public/home/liulizuo/step0/'+ key + '_step2.hdf5'
    model = models.load_model(save_model_path, custom_objects={'bce_dice_loss': bce_dice_loss,
                                                       'dice_loss': dice_loss})
    
    DATA_DIR = Organ_path+'/cutted/'
    DATA_PRED=Organ_path+'/pred'
    DATA_LIST =  os.listdir(DATA_DIR)
    for file in DATA_LIST:
        if '.npy' in file:
            test_CT = np.load(DATA_DIR+file)
            test_CT = test_CT.reshape([1]+Organ_config_stage2[key]+[1])
            test_predict = model.predict(test_CT)
            test_predict[test_predict<=0.05] = 0
            test_predict[test_predict>0.05] = 1
            test_predict = test_predict.reshape(Organ_config_stage2[key])
            save_data(DATA_PRED+'/'+file, test_predict)
        else:
            continue

    
'''

#################save the data ################################
def save_data(ds, maskData):
    # save the data for every patient
    outfile = ds
    np.save(outfile, maskData)
#--------------------------------------------------------------------------
#                                Write to dcm
#--------------------------------------------------------------------------


