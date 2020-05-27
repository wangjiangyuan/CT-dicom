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
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import compute_unary, create_pairwise_bilateral, create_pairwise_gaussian, unary_from_softmax
import skimage.io as io
from itertools import groupby


path_model='/public/home/liulizuo/new_step/'
path_dir='/public/home/liulizuo/TrainData_1011/'



#crop是把患者的脑袋全部放进来，并减少空白位置，从0开始的是从脑袋上面。也就是从position最大值开始
def crop(array, new_shape):

    new_shape = np.array(new_shape)
    shape = [new_shape[i, 1] - new_shape[i, 0] for i in range(2)]
    new_array = np.zeros(shape)
    new_array = array[new_shape[0, 0]:new_shape[0, 1],
                new_shape[1, 0]:new_shape[1, 1],
                :]
    return new_array
#定位最高点的几个代码
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
def center_coor(a):

    l = l2r_top(a)
    u=u2d_top(a,l)
    f=f2b_top(a,l,u)
    return l,u,f


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

#这部分是原来的，上下左右前后都有的这个定位，这个目的是根据left_eye_cor，进行调整，把left_eye的范围调整合适，利用它去定位其他的器官
def l2r_org(array):
    index = []
    for i in range(array.shape[0]):
        if np.max(array[i, :, :]) > 0.5:
            index.append(i)
    return index[0],index[-1]
def u2d_org(array):
    index = []
    for i in range(array.shape[1]):
        if np.max(array[:, i, :]) > 0.5:
            index.append(i)
    return index[0],index[-1]

def f2b_org(array):
    index = []
    for i in range(array.shape[2]):
        if np.max(array[:,:, i]) > 0.5:
            index.append(i)
    index_with_fault = []
    fun = lambda x: x[1]-x[0]
    for k, g in groupby(enumerate(index), fun):
        l1 = [j for i, j in g ]    
        index_with_fault = index_with_fault + l1
        if index_with_fault:
            break
    return index_with_fault[0],index_with_fault[-1]
def save_data(ds, maskData):
    # save the data for every patient
    outfile = ds
    np.save(outfile, maskData)





#####################################################################
#这部分，我首先从30-118，切了
def cut(CT, Organ_config_stage2, Organ_mask_coor_stage2,organ='default'):

    forward, backward,left, right, down, up =Organ_mask_coor_stage2[organ]
    temp_CT = np.zeros(Organ_config_stage2[organ])
    temp_CT[:, :, :] = CT[forward:backward, left:right, down:up]#x,y是相反的np[y,x,z]!!
    return temp_CT.reshape(Organ_config_stage2[organ] + [1])

def cut_left_eye(CT, Organ_config_stage2, forward, backward,left, right, down, up,organ):

    temp_CT = np.zeros(Organ_config_stage2[organ])
    temp_CT[:, :, :] = CT[forward:backward, left:right, down:up]#x,y是相反的np[y,x,z]!!
    return temp_CT.reshape(Organ_config_stage2[organ] + [1])


def pred(cutted_CT,Organ_config_stage2,key):
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    save_model_path = path_model+ key + '_step3.hdf5'
    model = models.load_model(save_model_path, custom_objects={'bce_dice_loss': bce_dice_loss,'dice_loss': dice_loss})
    test_CT = cutted_CT.reshape([1]+Organ_config_stage2[key]+[1])
    test_CT_crf=cutted_CT.reshape([cutted_CT.shape[0],cutted_CT.shape[1],cutted_CT.shape[2]])
    test_predict = model.predict(test_CT)
    test_predict = test_predict.reshape(Organ_config_stage2[key])
    
    test_pred=crf(test_predict,test_CT_crf)
    return test_pred
def crf(pred,CT):
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

#######################cor########################################

#这部分是根剧 left_eye_cor88张进行大范围的预测，将预测的mask作为left_eye，之后利用left_eye定位right_eye和mandible。在利用这两个器官去预测别的器官。这些数值基本上都是经过excel的统计。
def Left_eye_cor(aa,path_dir,Organ_config_stage2,Organ_mask_coor_stage2,file):
    organ='Left eye_cor'
    forward, backward,left, right, down, up=Organ_mask_coor_stage2['Left eye_cor']

    cutted_CT=cut(aa, Organ_config_stage2, Organ_mask_coor_stage2,organ)
    save_data(path_dir+'/'+organ+'/cutted/'+file,cutted_CT) 
    test_pred=pred(cutted_CT,Organ_config_stage2,organ)
    save_data(path_dir+'/'+organ+'/pred'+'/'+file, test_pred)
    a=test_pred
    a=a.reshape(a.shape[0],a.shape[1],a.shape[2])
    if np.max(a)<0.5:
        ll=20#这部分调整，有的可能最高点在胸部，所以眼睛位置会稍微靠下，这部分是完全一点眼睛都没有。所以加上一个值，能显示出眼睛。
        forward=forward+ll
        backward=backward+ll
        cutted_CT=cut_left_eye(aa, Organ_config_stage2, forward, backward,left, right, down, up,organ)
        save_data(path_dir+'/'+organ+'/cutted/'+file,cutted_CT) 
        test_pred=pred(cutted_CT,Organ_config_stage2,organ)
        save_data(path_dir+'/'+organ+'/pred'+'/'+file, test_pred)
        a=test_pred
    organ='Left eye'
    l,r=l2r_org(a)
    if l==0:#这部分是进行调整，有些眼睛位置偏上，偏下，偏左，偏右。在规定的范围是不全的，但是是有的，能够预测出一部分，所以进行一定的调整，把整个眼睛都能放在范围里
        ll=29-r
        forward=forward-ll
        backward=backward-ll
    if r==31:
        rr=l-2
        forward=forward+rr
        backward=backward+rr
    u,d=u2d_org(a)
    if u==0:
        uu=29-d
        left=left-uu
        right=right-uu
    if d==31:
        dd=u-2
        left=left+dd
        right=right+dd
    f,b=f2b_org(a)

    if f==0:
        up=b+4+30
        down=b-44+30
    elif b==47:
        down=f-4+30
        up=f+44+30
    else:
        center=int((f+b)/2)+30
        down=center-24
        up=center+24
    Organ_mask_coor_stage2["Left eye"]=[forward,backward,left,right,down,up]
    cutted_CT=cut(aa, Organ_config_stage2, Organ_mask_coor_stage2,organ)
    save_data(path_dir+'/'+organ+'/cutted/'+file,cutted_CT) 
    test_pred=pred(cutted_CT,Organ_config_stage2,organ)
    l,r=l2r_org(test_pred)
    u,d=u2d_org(test_pred)
    f,b=f2b_org(test_pred)
    Organ_mask_coor_stage2["Left eye_mask"]=[l,r,u,d,f,b]
    save_data(path_dir+'/'+organ+'/pred'+'/'+file, test_pred)

    print(organ+' has done')



#这个是利用left_eye定位right eye和mandible。另外利用right eye定位right 的一些器官，全部写入到Organ_mask_coor_stage2
def XandY_RightEye_organ(aa,Organ_config_stage2,Organ_mask_coor_stage2):

    l,r,u,d,f,b=Organ_mask_coor_stage2["Left eye_mask"]
    forward,backward,left,right,down,up=Organ_mask_coor_stage2["Left eye"]
    right=left-25#这些数值定位是我参照excel统计的，大部分都是在这个范围里，这部分也可以调整，但是代码太多，就先这样了。或者有什么好一点的算法优化一下。
    left=right-32
    Organ_mask_coor_stage2["Right eye"]=[forward,backward,left,right,down,up]

    forward=r+forward-52
    backward=forward+120
    left=left-25
    right=left+136
    down=f+down+56
    up=down+96
    Organ_mask_coor_stage2["Mandible"]=[forward,backward,left,right,down,up]
    forward,backward,left,right,down,up=Organ_mask_coor_stage2["Right eye"]
    forward=r+forward+30
    backward=forward+48
    right=left+u+20
    left=right-48
    down=b+down-5
    up=down+24
    Organ_mask_coor_stage2["Right TM-joint"]=[forward,backward,left,right,down,up]

    forward,backward,left,right,down,up=Organ_mask_coor_stage2["Right eye"]
    forward=l+forward-5
    backward=forward+16
    center_y=int((u+d)/2)+left
    left=center_y-6
    right=center_y+10
    center_z=int((f+b)/2)+down
    down=center_z-8
    up=center_z+8
    Organ_mask_coor_stage2["Right lens"]=[forward,backward,left,right,down,up]

    forward,backward,left,right,down,up=Organ_mask_coor_stage2["Right eye"]
    forward=r+forward-6
    backward=forward+48
    center_y=int((u+d)/2)+left
    left=center_y
    right=center_y+32
    center_z=int((f+b)/2)+down
    down=center_z-8
    up=center_z+8
    Organ_mask_coor_stage2["Right optic nerve"]=[forward,backward,left,right,down,up]

    forward,backward,left,right,down,up=Organ_mask_coor_stage2["Right eye"]
    forward=r+10+forward
    backward=forward+120
    right=d+5+left
    left=right-64
    down=f-24+down
    up=down+56
    Organ_mask_coor_stage2["R-temporal lobe"]=[forward,backward,left,right,down,up]

    forward,backward,left,right,down,up=Organ_mask_coor_stage2["Right eye"]
    forward=r+forward+15
    backward=forward+48
    left=right-6
    right=left+48
    center_z=int((f+b)/2)+down
    down=center_z-8
    up=center_z+8
    Organ_mask_coor_stage2["Optic chiasm"]=[forward,backward,left,right,down,up]

    forward,backward,left,right,down,up=Organ_mask_coor_stage2["Right eye"]
    forward=r+forward+40
    backward=forward+48
    left=right-6
    right=left+48
    center_z=int((f+b)/2)+down
    down=center_z-28
    up=center_z+44
    Organ_mask_coor_stage2["Brain stem"]=[forward,backward,left,right,down,up]
    
    forward,backward,left,right,down,up=Organ_mask_coor_stage2["Right eye"]
    forward=r+forward+46
    backward=forward+64
    left=right-15
    right=left+64
    up=aa.shape[2]
    down=up-256
    Organ_mask_coor_stage2["Spinal cord"]=[forward,backward,left,right,down,up]

#这个是得到mandible和left eye的mask的结果
def get_mask_Mandible_LeftEye(aa,path_dir,Organ_config_stage2,Organ_mask_coor_stage2,organ,file):
    cutted_CT=cut(aa, Organ_config_stage2, Organ_mask_coor_stage2,organ)
    save_data(path_dir+'/'+organ+'/cutted/'+file,cutted_CT) 
    test_pred=pred(cutted_CT,Organ_config_stage2,organ)
    save_data(path_dir+'/'+organ+'/pred'+'/'+file, test_pred)
    l,r=l2r_org(test_pred)
    u,d=u2d_org(test_pred)
    f,b=f2b_org(test_pred)
    organ_mask=organ+'_mask'
    Organ_mask_coor_stage2[organ_mask]=[l,r,u,d,f,b]

    print(organ+' has done')

#这部分是利用left eye的mask定位其他left的器官
def XandY_LeftEye_organ(aa,Organ_config_stage2,Organ_mask_coor_stage2):

    l,r,u,d,f,b=Organ_mask_coor_stage2["Left eye_mask"]

    forward,backward,left,right,down,up=Organ_mask_coor_stage2["Left eye"]
    forward=r+forward+40
    backward=forward+48
    left=left+d-10
    right=left+48
    down=b+down-5
    up=down+24
    Organ_mask_coor_stage2["Left TM-joint"]=[forward,backward,left,right,down,up]

    forward,backward,left,right,down,up=Organ_mask_coor_stage2["Left eye"]
    forward=l+forward-5
    backward=forward+16
    center_y=int((u+d)/2)+left
    left=center_y-10
    right=center_y+6
    center_z=int((f+b)/2)+down
    down=center_z-8
    up=center_z+8
    Organ_mask_coor_stage2["Left lens"]=[forward,backward,left,right,down,up]

    forward,backward,left,right,down,up=Organ_mask_coor_stage2["Left eye"]
    forward=r+forward-6
    backward=forward+48
    center_y=int((u+d)/2)+left
    left=center_y
    right=center_y+32
    center_z=int((f+b)/2)+down
    down=center_z-8
    up=center_z+8
    Organ_mask_coor_stage2["Left optic nerve"]=[forward,backward,left,right,down,up]

    forward,backward,left,right,down,up=Organ_mask_coor_stage2["Left eye"]
    forward=r+10+forward
    backward=forward+120
    right=d+5+left
    left=right-64
    down=f-24+down
    up=down+56
    Organ_mask_coor_stage2["L-temporal lobe"]=[forward,backward,left,right,down,up]


#这部分是利用mandible的mask定位其他的器官，这里的mandible改过了，不会出现shape不对的bug
def XandY_Mandible_organ(aa,Organ_config_stage2,Organ_mask_coor_stage2):
    l,r,u,d,f,b=Organ_mask_coor_stage2["Mandible_mask"]

    forward,backward,left,right,down,up=Organ_mask_coor_stage2["Mandible"]
    forward=r+forward-12
    backward=forward+56
    left=left+u-30
    right=left+56
    down=f+down-22
    up=down+80
    Organ_mask_coor_stage2["R-parotid"]=[forward,backward,left,right,down,up]

    forward,backward,left,right,down,up=Organ_mask_coor_stage2["Mandible"]
    forward=r+forward-12
    backward=forward+56
    right=left+d+30
    left=right-56
    down=f+down-22
    up=down+80
    Organ_mask_coor_stage2["L-parotid"]=[forward,backward,left,right,down,up]





def get_mask_other(aa,path_dir,Organ_config_stage2,Organ_mask_coor_stage2,organ,file):
    cutted_CT=cut(aa, Organ_config_stage2, Organ_mask_coor_stage2,organ)
    save_data(path_dir+'/'+organ+'/cutted/'+file,cutted_CT) 
    test_pred=pred(cutted_CT,Organ_config_stage2,organ)
    save_data(path_dir+'/'+organ+'/pred'+'/'+file, test_pred)
    print(organ+' has done')


#######################Right eye########################################



#######################left eye########################################







#######################Mandible########################################


#######################################################################

#将计算出来的每个器官的前后左右上下坐标写入一个txt文件中。利用这个文件统一进行cut和pred操作

def main(path_dir,file):
    aa = np.load(path_dir+'/CT/'+file)
    os.mkdir(path_dir + '/crop')
    l_crop,u_crop,f_crop=center_coor(aa)
    shape_crop = [[l_crop-30, l_crop+250], [u_crop-100, u_crop+100]]

    aa_crop = crop(aa, shape_crop)

    save_data(path_dir + '/crop/' + file, aa_crop)
    Organ_config_stage2 = {
    'Mandible':          [120, 136, 96],#mandible这块的shape我修改过了，不会出现shape不对的bug了
    'Spinal cord':          [64,64,256],
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
    'Left eye_cor':         [32,32,88]}

    Target_Organ_list = "Brain stem, L-parotid, L-temporal lobe, Left TM-joint, Left eye, Left lens, Left optic nerve, Mandible, R-parotid, R-temporal lobe, Right TM-joint, Right eye, Right lens, Right optic nerve, Spinal cord, Optic chiasm, Left eye_cor"
    Target_Organ = Target_Organ_list.split(", ")
    for organ in Target_Organ:
        # stage 1
        Organ_path = path_dir+'/'+organ
        os.mkdir(Organ_path)

        os.mkdir(Organ_path +'/cutted')
        
        os.mkdir(Organ_path + '/pred')
    Organ_mask_coor_stage2={"Left eye_cor":[52,84,110,142,30,118]}
    Left_eye_cor(aa_crop,path_dir,Organ_config_stage2,Organ_mask_coor_stage2,file)
    XandY_RightEye_organ(aa_crop,Organ_config_stage2,Organ_mask_coor_stage2)
    get_mask_Mandible_LeftEye(aa_crop,path_dir,Organ_config_stage2,Organ_mask_coor_stage2,'Mandible',file)
    get_mask_Mandible_LeftEye(aa_crop,path_dir,Organ_config_stage2,Organ_mask_coor_stage2,'Left eye',file)
    #del Right_eye_cor,XandY_RightEye_organ
    XandY_LeftEye_organ(aa_crop,Organ_config_stage2,Organ_mask_coor_stage2)
    XandY_Mandible_organ(aa_crop,Organ_config_stage2,Organ_mask_coor_stage2)
    #del XandY_LeftEye_organ,XandY_Mandible_organ
    txtjson=json.dumps(Organ_mask_coor_stage2)
    txtfile=open(path_dir+'txt/'+file+'.txt','w',encoding='utf-8')
    txtfile.write(txtjson)
    txtfile.close
    organ_other_list=['Brain stem','R-temporal lobe', 'Right TM-joint',  'Right lens', 'Right optic nerve','Spinal cord', 'Optic chiasm','L-parotid','R-parotid','L-temporal lobe', 'Left TM-joint', 'Left lens', 'Left optic nerve']
    for organ in organ_other_list:
        get_mask_other(aa_crop,path_dir,Organ_config_stage2,Organ_mask_coor_stage2,organ,file)
    #del get_mask_other


'''
path_dir='/public/home/liulizuo/TrainData_1011/'
DATA_DIR = '/public/home/liulizuo/TrainData_1011'+'/CT/'
DATA_PRED= '/public/home/liulizuo/TrainData_1011'+'/mask/'
DATA_LIST =  os.listdir(DATA_DIR)
for file in DATA_LIST:
    if '.npy' in file:
        print(file)    
        main(DATA_DIR,path_dir,file)
        
    else:
        continue

'''












