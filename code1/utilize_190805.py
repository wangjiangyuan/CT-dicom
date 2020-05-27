# -*- coding:utf-8 -*-
import os
import re
import pandas as pd
import numpy as np
from numpy.core.multiarray import ndarray
from scipy import ndimage as ndi
import pydicom as dicom
import scipy.misc
from dicompylercore import dicomparser
from matplotlib import path
import tensorflow as tf
from tensorflow.keras import callbacks
from tensorflow.keras import backend as K
from tensorflow.keras import models
from tensorflow.keras import losses
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import compute_unary, create_pairwise_bilateral, create_pairwise_gaussian, unary_from_softmax
import skimage.io as io





Organ_config_stage2 = {#这一部分是不同器官的大小
    'Mandible':          [120, 136, 96],
    'Spinal cord':          [80,80,256],
    'Right TM-joint':      [48, 48, 24],
    'Right eye':           [32, 32, 48],
    'Right lens':          [16, 16, 16],
    'Right optic nerve':   [48, 32, 16],
    'R-parotid':           [56, 56, 80],
    'R-temporal lobe':    [120, 64, 56],
    'Optic chiasm':        [48, 48, 16],
    'Brain stem':          [48, 48, 72],
    'Left lens':           [16, 16, 16],
    'Left optic nerve':    [48, 32, 16],
    'Left TM-joint':       [48, 48, 24],
    'L-parotid':           [56, 56, 80],
    'L-temporal lobe':    [120, 64, 56],
    'Left eye':            [32, 32, 48],
    'Left eye_cor':        [32,32,88]#Left eye_cor  这一个z是按照50-138=88。这块我做过计算，左眼基本上就在这个范围内。所有的0图层，代表的是从头顶出发，一直到胸部。图形所示就是从一个小圆为起始
}



#TODO : To update the algorithms of in_the_ploy
################Load all the data####################################
def load_scan(path):
    CTpath = os.listdir(path)
    slices = [dicom.read_file(path + '/' + s) for s in CTpath]
    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))
    return slices

def Get_path(PathDicom):
    IstFilesDCM = []
    RsFileDCM = []
    for dirName, subdirList, fileList in os.walk(PathDicom):
        for filename in fileList:
            if ".dcm" in filename.lower() and re.match('CT',filename):
                IstFilesDCM.append(os.path.join(dirName,filename))
            if ".dcm" in filename.lower() and re.match('RS',filename):
                RsFileDCM.append(os.path.join(dirName,filename))
    return IstFilesDCM, RsFileDCM

def load_CT_dcm(path):
    slices = [dicom.read_file(s) for s in path]
    slice_dict = Get_Frame_UID_Dict(slices)
    return slice_dict


def load_Contour_dcm(path):
    slices = dict()
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

def Get_Frame_UID_Dict(slices):
    # get series_UID and sort the dicom file in the list
    FUID_Dict = dict()
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
    
##################################################################################
    
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

def resample(path_DATA, image, scan, mask, new_spacing=[1,1,1]):#将CT转为npy格式，这块我做了修改，可以把同一个图层的不连续的mask，很正确的写进一个图层里面。
    # resampling the image and the mask at the same time
    thickness, thickness_index = Index_of_changed_points(scan)
    PatientID = scan[0].PatientID
    Date = scan[0].StudyDate
    PixelSpacing = scan[0].PixelSpacing
    PatientInfo = PatientID+Date
    if thickness_index:
        preprocessed_slice = interpolate_thickness_by_thickness(image,thickness_index,thickness,PixelSpacing,new_spacing=[1,1,1])
        preprocessed_slice = np.flip(preprocessed_slice, 2)
        save_data(path_DATA+'CT/'+PatientID+Date,preprocessed_slice)
        for key in mask.keys():
            mask_slice = interpolate_thickness_by_thickness(mask[key],thickness_index,thickness,PixelSpacing,new_spacing=[1,1,1],order=0)
            mask_slice = np.flip(mask_slice, 2)
            save_data(path_DATA+key+'/'+PatientID+Date,mask_slice)
        return new_spacing,thickness_index,thickness,PixelSpacing,PatientInfo

    else:
        preprocessed_slice = interpolate(image,PixelSpacing,thickness[0],new_spacing=[1,1,1])
        preprocessed_slice = np.flip(preprocessed_slice, 2)
        for key in mask.keys():
            mask_slice = interpolate(mask[key],PixelSpacing,thickness[0],order=0)
            mask_slice = np.flip(mask_slice, 2)
            save_data(path_DATA+key+'/'+PatientID+Date,mask_slice)
        save_data(path_DATA+'CT/'+PatientID+Date,preprocessed_slice)
        return new_spacing, thickness_index,thickness,PixelSpacing,PatientInfo

def interpolate_thickness_by_thickness(image,thickness_index,thickness,PixelSpacing,new_spacing=[1,1,1],order=3):
    preprocessed_slice = []
    for i in range(len(thickness_index)):
        if i == len(thickness_index) -1:
            break
        else:
            img = interpolate(image[:,:,thickness_index[i]:(thickness_index[i+1]-1)],PixelSpacing,thickness[i], new_spacing=[1,1,1],order=order)
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
    image=image.astype('int16')
    return image

def get_mask(contour,patient_slices,Target_Organ, shape ):

        #contour: list of contour raw data for one SeriesUID
        #patient_slices: list of sorted slices with same SeriesUID
        #shape  is this a needed data
        #Target_Organ: which organ you want to get mask
        #This function can delineate all contours eg:Mandible

    dcmOrigin = patient_slices[0].ImagePositionPatient
    frame2z_axis = [s.SliceLocation for s in patient_slices]
    dcmSpacing = patient_slices[0].PixelSpacing
    structures=contour.GetStructures()
    Organ_ID=pd.DataFrame(list(structures.values()),columns=['id','name'])
    Organ_contour_data_dict =  dict()
    Organ_had = []
    for organ in Target_Organ:
        if organ in list(Organ_ID['name']):
            Organ_had.append(organ)
            maskData = np.zeros([shape[0], shape[1], shape[2]])
            ROINumber = Organ_ID[Organ_ID['name']==organ]['id']
            contour_data = contour.GetStructureCoordinates(ROINumber)
            for value in contour_data:
                number=len(contour_data[value])
                if number ==1:
                    contour_coord = np.array(contour_data[value][0]['data'])
                    NoPoints,  _ = np.shape(contour_coord)
                    pointData = np.zeros([NoPoints, 2])
                    for i in range(NoPoints):
                        pointData[i,0] = np.round( (contour_coord[i,0] - dcmOrigin[0])/dcmSpacing[0])
                        pointData[i,1] = np.round( (contour_coord[i,1] - dcmOrigin[1])/dcmSpacing[1])
                    x = np.zeros([shape[0],shape[1]])
                    in_the_curve = inpolygon(x,pointData)

                else:
                    in_the_curve = np.zeros([shape[0],shape[1]])
                    for i in range(number):
                        
                        contour_coord = np.array(contour_data[value][i]['data'])
                        NoPoints,  _ = np.shape(contour_coord)
                        pointData = np.zeros([NoPoints, 2])
                        for i in range(NoPoints):
                            pointData[i,0] = np.round( (contour_coord[i,0] - dcmOrigin[0])/dcmSpacing[0])
                            pointData[i,1] = np.round( (contour_coord[i,1] - dcmOrigin[1])/dcmSpacing[1])
                        x = np.zeros([shape[0],shape[1]])
                        k = inpolygon(x,pointData)
                        in_the_curve=in_the_curve+k
                index = frame2z_axis.index(contour_coord[1,2])
                maskData[:,:,index] += np.swapaxes(in_the_curve,0,1)
            Organ_contour_data_dict[organ] = maskData
        else:
            continue

    return Organ_contour_data_dict, Organ_had
    
# --------------------------------------------------------------------------
#                                Before Stage 1
# --------------------------------------------------------------------------


#To locate the top of the head
#这三个是定位最高点的，CT是平躺下的，最下面的是后脑勺，患者躺在那，姿势不一定。一般最高点分为三种，一种是额头，大部分是鼻尖，还有一部分是胸部。
def center_coor(a):

    l = l2r_top(a)
    u=u2d_top(a,l)
    f=f2b_top(a,l,u)
    return l,u,f

def l2r_top(array):
    index = []
    for i in range(array.shape[0]):
        if np.max(array[i, :, :]) > 0.5:
            index.append(i)
    return index[0]

def u2d_top(array,x):
    index = []
    for i in range(array.shape[1]):
        if np.max(array[x, i, :]) > 0.5:
            index.append(i)
    return index[0]

def f2b_top(array,x,y):
    index = []
    for i in range(array.shape[2]):
        if np.max(array[x, y, i]) > 0.5:
            index.append(i)
    return index[0]


#To crop the image as the shape[280,200,:]
def crop(array, new_shape):
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
def crop_SpinalCord(array, new_shape):#这里spinal cord器官是从最后一个图层倒着来的，所以限制了它的z坐标。
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
                new_shape[2, 0]:new_shape[2, 1]]
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

def locate(mask_shape, target_shape, center):
    """
    to locate the start point and end point for every dimension
        if the index is out of range(0, len(shape)) then will start at 0 or end at len(shape)

        :param mask_shape: the shape of given mask's one dimension,
        :param target_shape: the target shape of the correspondent dimension
                             you want to cut from the step1's training data,
                             given by the organ's dict
        :param center: the center of of the cut space
        :return: the cut point
    """
    if (center - int(target_shape / 2)) < 0:
        return 0, target_shape
    elif (center + int(target_shape / 2)) > mask_shape:
        return mask_shape - target_shape, mask_shape
    else:
        return center - int(target_shape / 2), center + int(target_shape / 2)


def crop_array(path_DATA,organ='default'):#这个crop就按照最高点的将它第一步切割，把整个头部放进来，外围留一部分的空间
    """
    To crop the target part of an mask

        :param maskdir: the mask directory
        :param organ: to determine what which put to be used as step 1's training data
        :return: the cropped array
    """
     # TODO: maybe new_shape should be used as a parameter
    if organ == 'Spinal cord':
        CT_filedir = path_DATA+'CT'
        maskdir=path_DATA+organ
        maski_file_list = os.listdir(maskdir)
        for file in maski_file_list:
            if '.npy' in file:
                mask = np.load(maskdir + '/' + file)
                CT = np.load(CT_filedir + '/' + file)
                l,u,f=center_coor(CT)
                shape = [[l-30, l+250], [u-100, u+100],[mask.shape[2]-256,mask.shape[2]]]
                croped_mask = crop_SpinalCord(mask, shape)
                croped_CT = crop_SpinalCord(CT, shape)
                save_data(maskdir + '/mask/' + file, croped_mask)
                save_data(maskdir + '/CT/' + file, croped_CT)
            else:
                continue
            
    else:
        
        CT_filedir = path_DATA+'CT'
        maskdir=path_DATA+organ
        maski_file_list = os.listdir(maskdir)
        for file in maski_file_list:
            if '.npy' in file:
                mask = np.load(maskdir + '/' + file)
                CT = np.load(CT_filedir + '/' + file)
                l,u,f=center_coor(CT)
                shape = [[l-30, l+250], [u-100, u+100]]
                croped_mask = crop(mask, shape)
                croped_CT = crop(CT, shape)
                save_data(maskdir + '/mask/' + file, croped_mask)
                save_data(maskdir + '/CT/' + file, croped_CT)
            else:
                continue

# --------------------------------------------------------------------------
#                                After Stage 1
# --------------------------------------------------------------------------
#   -----------------------------------
#   The l2r u2d f2b functions are used
#   to locate the start point and end
#   point for each given dimension.
#   -----------------------------------

def l2r(array):
    """
    The first dimension
    :param array:
    :return:
    """
    index = []
    for i in range(array.shape[0]):
        if np.max(array[i, :, :]) > 0.5:
            index.append(i)
    return index[0], index[-1]


def u2d(array):
    """
    The second dimension
    :param array:
    :return:
    """
    index = []
    for i in range(array.shape[1]):
        if np.max(array[:, i, :]) > 0.5:
            index.append(i)
    return index[0], index[-1]


def f2b(array):
    """
    The third dimension
    :param array:
    :return:
    """
    index = []
    for i in range(array.shape[2]):
        if np.max(array[:, :, i]) > 0.5:
            index.append(i)
    return index[0], index[-1]

def balloon(array, threshold, shape=[320, 320, 192]):
    """
    After the loc_net's work, we need to expand the shape of mask-cutting array to the basic shape e.g [320,320,120]
    this function is to do this.
    :param array: the loc_net's output
    :param threshold: the threshold to determine how many point should be changed to 0,1
    :param shape:  the target shape for the feeding of stage2 net
    :return: the expanded array
    """
    array = array.reshape([64, 64, 32])
    array[array >= threshold] = 1.0
    array[array < threshold] = 0
    a, b = l2r_orgin(array)
    c, d = u2d_orgin(array)
    e, f = f2b_orgin(array)
    array[a:b, c:d, e:f] = 1.0
    real_resize_factor = np.array(shape) / array.shape
    image = scipy.ndimage.interpolation.zoom(array, real_resize_factor, mode='nearest', order=0)
    return image


########################################################################


def cut(mask, CT, x, organ='default'):#这个cut还是按照原来的根据mask定位CT，之后将二者丢进深度学习网络中训练
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
    temp_mask[:, :, :] = mask[forward:backward, left:right, down:up]
    temp_CT[:, :, :] = CT[forward:backward, left:right, down:up]
    return temp_mask.reshape(x[organ] + [1]), temp_CT.reshape(x[organ] + [1])
def coordination(mask,maskdir,organ='default'):
    a, b = l2r(mask)
    c, d = u2d(mask)
    e, f = f2b(mask)
    a_center, c_center, e_center = int((a + b) / 2), int((c + d) / 2), int((e + f) / 2)
    return a, b, c, d, e, f, a_center, c_center, e_center

def cut_array(maskdir,organ='default'):
    """
    The main function to cut the array, no return, but saving the data in ~/cutted/
    :param maskdir: just like the mask_dir in the function shrink()
    """

    maskfilelist = os.listdir(maskdir + '/mask')
    for file in maskfilelist:
        if '.npy' in file:
            mask = np.load(maskdir + '/mask/' + file)
            CT = np.load(maskdir + '/CT/' + file)
            cutted_mask, cutted_CT= cut(mask, CT,Organ_config_stage2,organ)

            save_data(maskdir + '/cutted/mask/' + file, cutted_mask)
            save_data(maskdir + '/cutted/CT/' + file, cutted_CT)
        else:
            continue

#下面是关于预测的，在这个模块中还用不到预测的功能，这个模块主要是为了训练模型使用的。
def crf(pred,CT):#这个crf是针对一个图层进行的，我试了感觉没有什么效果。
    preprocessed_slice = []
    for i in range(pred.shape[2]):
        H, W, NLABELS = pred.shape[0],pred.shape[1], 2
        probs=pred[:,:,i]
        img=CT[:,:,i]
        
        img=img.reshape([img.shape[0],img.shape[1]]+[1])
        
        probs = np.tile(probs[np.newaxis,:,:],(2,1,1))
        probs[1,:,:] = 1 - probs[0,:,:]

        pairwise_energy = create_pairwise_bilateral(sdims=(5,5), schan=(1.5,), img=img, chdim=2)
        img_en = pairwise_energy.reshape((-1, H, W)) 
        
        U = unary_from_softmax(probs)

        U = U.reshape((2,-1))
        d = dcrf.DenseCRF2D(W, H, NLABELS)
        d.setUnaryEnergy(U)
        d.addPairwiseGaussian(sxy=(5,5), compat=7, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
        Q = d.inference(60)
        map_i = np.argmax(Q, axis=0).reshape((pred.shape[0],pred.shape[1],1))
        map=1-map_i[:,:,:]

        #d.stepInference(Q, tmp1, tmp2)
        #kl1 = d.klDivergence(Q) / (H*W)
        #map_soln1 = np.argmax(Q, axis=0).reshape((H,W,1))
        preprocessed_slice.append(map)
    ee=np.concatenate(preprocessed_slice,axis=2)
    return ee





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
def pred_mask(Organ_path,path_model,organ='default'):
    key=organ
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    save_model_path = path_model+ key + '_step3.hdf5'
    model = models.load_model(save_model_path, custom_objects={'bce_dice_loss': bce_dice_loss,
                                                       'dice_loss': dice_loss})
    
    DATA_DIR = Organ_path+'/cutted/CT/'
    DATA_PRED=Organ_path+'/pred'
    DATA_LIST =  os.listdir(DATA_DIR)
    for file in DATA_LIST:
        if '.npy' in file:
            test_CT_orgin = np.load(DATA_DIR+file)
            test_CT_crf=test_CT_orgin.reshape([test_CT_orgin.shape[0],test_CT_orgin.shape[1],test_CT_orgin.shape[2]])
            test_CT = test_CT_orgin.reshape([1]+Organ_config_stage2[key]+[1])
            test_predict = model.predict(test_CT)
            test_predict = test_predict.reshape(Organ_config_stage2[key])
            test_pred=crf(test_predict,test_CT_crf)
            save_data(DATA_PRED+'/'+file, test_pred)



        else:
            continue
#################save the data ################################
def save_data(ds, maskData):
    # save the data for every patient
    outfile = ds
    np.save(outfile, maskData)
