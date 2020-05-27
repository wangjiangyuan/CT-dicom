import re
import os
import pydicom
import time 
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure,draw
from math import ceil, floor
from pydicom.dataset import Dataset, FileDataset
PATH='/public/home/liulizuo/OaR_2stage_3d_unet/write2dcm/123/test/'
model_path='/public/home/liulizuo/OaR_2stage_3d_unet/write2dcm/123/test_new/model'
mask_path='/public/home/liulizuo/OaR_2stage_3d_unet/write2dcm/123/test_mask_new/'

#organ_list=['Brain stem','L-parotid','L-temporal lobe','Left TM-joint','Left eye','Left lens','Left optic nerve','Mandible','R-parotid','R-temporal lobe','Right TM-joint','Right eye','Right lens','Right optic nerve','Spinal cord','Optic chiasm']

#color_list=[[0,0,255],[50,205,50],[60,179,113],[60,179,113],[128,255,255],[123,123,192],[123,123,192],[166,255,255],[56,142,142],[255,104,32],[123,123,192],[0,255,255],[56,142,142],[128,128,255],[0,0,255],[56,142,142]]
organ_list=['Brain stem','L-parotid','L-temporal lobe','Left TM-joint','Left eye','Left lens','Left optic nerve','Mandible','R-parotid','R-temporal lobe','Right TM-joint','Right eye','Right lens','Right optic nerve']
#organ_list=['Right eye','Left eye']
color_list=[[0,0,255],[50,205,50],[60,179,113],[60,179,113],[128,255,255],[123,123,192],[123,123,192],[166,255,255],[56,142,142],[255,104,32],[123,123,192],[0,255,255],[56,142,142],[128,128,255]]

#这里需要读取保存的txt的各个器官的坐标，这块我还没来得及写上。用来代替下面的这些坐标。
Organ_mask_coor_stage2={#这里的参数是x,y,z加上第一次裁剪的初始x，y，z，三个数字
    "Brain stem":[93, 141, -23, 25 ,-58, 14,90,90,0],
    "L-parotid":[72, 128, 33, 89,-13, 67,90,90,0],
    "L-temporal lobe":[65, 185, 8, 72 ,-65, -9,90,90,0],
    "Left TM-joint":[72, 120, 30, 78 ,-20, 4,90,90,0],
    "Left eye":[20, 52, 18, 50 ,-58, -10,90,90,0],
    "Left lens":[22, 38, 27, 43 ,-41, -25,90,90,0],
    "Left optic nerve":[38, 86, 7, 39 ,-43, -27,90,90,0],
    "Mandible":[1, 121, -67, 69 ,-11, 85,90,90,0],
    "R-parotid":[71, 127, -88, -32 ,-13, 67,90,90,0],
    "R-temporal lobe":[60, 180, -74, -10,-65, -9,90,90,0],
    "Right TM-joint":[69, 117, -79, -31 ,-20, 4,90,90,0],
    "Right eye":[22, 54, -43, -11 ,-58, -10,90,90,0],
    "Right lens":[20, 36, -35, -19 ,-41, -25,90,90,0],
    "Right optic nerve":[40, 88, -33, -1 ,-43, -27,90,90,0],
    #"Spinal cord":[-17, 63, -147, -67 ,-92, 164],
    "Optic chiasm":[65, 113, -22, 26 ,-43, -27,90,90,0]}


'''
Organ_mask_coor_stage2={#这里的参数是x,y,z加上第一次裁剪的初始x，y，z，三个数字
    "Brain stem":[93, 141, -23, 25 ,137, 209,90,90,23],
    "L-parotid":[72, 128, 33, 89,147, 227,90,90,23],
    "L-temporal lobe":[65, 185, 8, 72 ,131, 187,90,90,23],
    "Left TM-joint":[72, 120, 30, 78 ,155, 179,90,90,23],
    "Left eye":[20, 52, 18, 50 ,128, 176,90,90,23],
    "Left lens":[22, 38, 27, 43 ,139, 155,90,90,23],
    "Left optic nerve":[38, 86, 7, 39 ,142, 158,90,90,23],
    "Mandible":[1, 121, -67, 69 ,150, 246,90,90,23],
    "R-parotid":[71, 127, -88, -32 ,147, 227,90,90,23],
    "R-temporal lobe":[60, 180, -74, -10,130, 186,90,90,23],
    "Right TM-joint":[69, 117, -79, -31 ,147, 177,90,90,23],
    "Right eye":[22, 54, -43, -11 ,128, 176,90,90,23],
    "Right lens":[20, 36, -35, -19 ,138, 154,90,90,23],
    "Right optic nerve":[40, 88, -33, -1 ,142, 158,90,90,23],
    "Spinal cord":[-17, 63, -147, -67 ,89, 345,195,195,0],
    "Optic chiasm":[65, 113, -22, 26 ,142, 158,90,90,23]
}
'''
#这个是把mask写入dcm的脚本，有两种格式，一种是模板有后缀.dcm的，另一种是没有的。这两种格式有两种不同的写入，主要在
def pred_to_dcm(PATH,mask_path):

    dirName_list=[]
    for dirName, subdirList, fileList in os.walk(PATH):
        IstFilesDCM=[]
        if dirName==PATH:
            continue
        else:
            for filename in os.listdir(dirName):
                
                if ".dir" not in filename.lower() and re.match('CT',filename) :
                    IstFilesDCM.append(os.path.join(dirName,filename))
        RT_imageStorage =pydicom.dcmread(model_path,force=True)
        
        CT=pydicom.dcmread(IstFilesDCM[0],force=True)
        now = int(time.time()) 
        timeStruct = time.localtime(now) 
        now_Date = time.strftime("%Y%m%d", timeStruct) 
        now_Time = time.strftime("%H%M%S", timeStruct) 
        InstanceCreationDate=now_Date
        InstanceCreationTime=now_Time
        FrameOfReferenceUID=CT.FrameOfReferenceUID
        ReferencedSOPInstanceUID=CT.StudyInstanceUID
        SeriesInstanceUID=CT.SeriesInstanceUID
        Patient_id=int(CT.PatientID)
        RS_SOPInstanceUID='1.2.246.352.205.'+str(Patient_id)+str(InstanceCreationDate)+'12345.'+str(Patient_id)+str(InstanceCreationTime)+str(InstanceCreationTime)+'1'
        #导出各式一共有两种
        #save_path=dirName+'/'+'RTSTRUCT_'+RS_SOPInstanceUID #这个格式是后缀没有.dcm的，所以需要下面113-148
        save_path=dirName+'/'+'RTSTRUCT_'+RS_SOPInstanceUID+'.dcm'#这个格式是后缀有.dcm的，所以不需要下面113-148
        position_list_cut, CT_SOPInstanceUID_all_dict, path_save, x_position, y_position,position_min,position_max,Study_InstanceUID,Series_InstanceUID,file_name,FrameOfReferenceUID=get_uid_position_list(IstFilesDCM,dirName)
        dic_mask, list_position_dic=get_organMask_dic(file_name, position_list_cut, mask_path,organ_list, position_min,position_max, x_position, y_position,Organ_mask_coor_stage2)
        RT_imageStorage=get_RT_imageStorage(RT_imageStorage,dirName,IstFilesDCM,position_list_cut, CT_SOPInstanceUID_all_dict, x_position, y_position,position_min,position_max,Study_InstanceUID,Series_InstanceUID,file_name,dic_mask, list_position_dic)

        RT_imageStorage=get_Referenced_Frame_of_Reference_Sequence(RT_imageStorage,CT_SOPInstanceUID_all_dict,Series_InstanceUID,Study_InstanceUID,FrameOfReferenceUID)
        RT_imageStorage=get_Structure_SetROI(RT_imageStorage,organ_list,FrameOfReferenceUID)
        RT_imageStorage=get_RT_ROIObservationsSequence(RT_imageStorage,organ_list,FrameOfReferenceUID)
        RT_imageStorage=get_ROI_ContourSequence(RT_imageStorage,organ_list,color_list,dic_mask,list_position_dic,CT_SOPInstanceUID_all_dict)

        
        #RT_imageStorage.is_little_endian = True
        #RT_imageStorage.is_implicit_VR = False
        RT_imageStorage.save_as(save_path)
        print('We have done the RTSTRUCT_'+RS_SOPInstanceUID)
        txtName=dirName+'/CT_1.3.46.670589.33.1.63686679'+str(CT.StudyDate)+'123400002.'+str(CT.StudyDate)+str(CT.StudyDate)+str(CT.StudyDate)+'.dir'
        f=open(txtName,'a+')
        for uid_i in CT_SOPInstanceUID_all_dict:
            UID=CT_SOPInstanceUID_all_dict[uid_i]
            f.write('CT_'+UID+'\r\n')
        f.close()
        print('We have done the uid dir')
        txtName_dir=dirName+'/RTDIR.dir'
        '''
        g=open(txtName_dir,'a+')
        g.write('!Version=2'+'\r\n') 
        g.write('CT'+'\r\n')
        g.write('1.2.840.10008.5.1.4.1.1.2'+'\r\n')
        g.write('1.3.46.670589.33.1.63686679'+str(CT.StudyDate)+'123400002.'+str(CT.StudyDate)+str(CT.StudyDate)+str(CT.StudyDate)+'\r\n')
        g.write(CT.StudyInstanceUID+'\r\n')
        g.write(CT.StudyID+'\r\n')
        g.write(str(CT.PatientName)+'\r\n')
        g.write(str(CT[0x00200052].value)+'\r\n')
        g.write(str(CT.ImagePositionPatient[0])+'\r\n')
        g.write(str(CT.ImagePositionPatient[1])+'\r\n')
        g.write(str(CT.ImagePositionPatient[2])+'\r\n')
        g.write(str(CT.ImageOrientationPatient[0])+'\r\n')
        g.write(str(CT.ImageOrientationPatient[1])+'\r\n')
        g.write(str(CT.ImageOrientationPatient[2])+'\r\n')
        g.write(str(CT.ImageOrientationPatient[3])+'\r\n')
        g.write(str(CT.ImageOrientationPatient[4])+'\r\n')
        g.write(str(CT.ImageOrientationPatient[5])+'\r\n')
        g.write('RTSTRUCT'+'\r\n')
        g.write('1.2.840.10008.5.1.4.1.1.481.3'+'\r\n')
        g.write('2312'+'\r\n')
        g.write('1.3.46.670589.33.1.63686679'+str(CT.StudyDate)+'123400002.'+str(CT.StudyDate)+str(CT.StudyDate)+str(CT.StudyDate)+'\r\n')
        g.write('NO_ID'+'\r\n')
        g.write('NO_DESCRIPTION'+'\r\n')
        g.write('\r\n')
        g.write('\r\n')
        g.write('\r\n')
        g.write('\r\n')
        g.write('0'+'\r\n')
        g.write('0'+'\r\n')
        g.write('0'+'\r\n')
        g.write('0'+'\r\n')
        g.write('0'+'\r\n')
        g.write('0'+'\r\n')
        g.close()
        print('We have done the DIR dir')
        '''


def get_uid_position_list(IstFilesDCM,dirName):
    SOP_UID_Dict={}
    ReferenceUID={}
    position_list_cut=[]
    first_UID=pydicom.dcmread(IstFilesDCM[0],force=True).FrameOfReferenceUID
    FrameUID = []
    FrameUID.append(first_UID)
    CT_SOPInstanceUID_all_dict={}
    for s in IstFilesDCM:
        CT=pydicom.dcmread(s,force=True)
        if CT.FrameOfReferenceUID in FrameUID:
            slice_CT=float(CT[0x0020,0x0032][2])
            CT_SOPInstanceUID_all_dict[slice_CT]=CT.SOPInstanceUID
            FrameOfReferenceUID=CT.FrameOfReferenceUID

            PatientID =CT.PatientID
            Date = CT.StudyDate
            x_PixelSpacing=CT.PixelSpacing[0]
            y_PixelSpacing=CT.PixelSpacing[1]
            x_position=CT[0x0020,0x0032][0]
            y_position=CT[0x0020,0x0032][1]
            Rows_px=CT.Rows
            Columns_px=CT.Columns
            Study_InstanceUID=CT.StudyInstanceUID
            Series_InstanceUID=CT.SeriesInstanceUID
            FrameUID.append(CT.FrameOfReferenceUID)
            slice_CT_cut=float(CT[0x0020,0x0032][2])
            position_list_cut.append(slice_CT_cut)

        else: 
            continue
    path_save=str(dirName)+'/'
    file_name=str(PatientID)+str(Date)
    position_min=min(position_list_cut)
    position_max=max(position_list_cut)
    position_list_cut.sort(reverse=True)
    print('We have got the message for '+str(PatientID))
    return position_list_cut, CT_SOPInstanceUID_all_dict, path_save, x_position, y_position,position_min,position_max,Study_InstanceUID,Series_InstanceUID,file_name,FrameOfReferenceUID


def get_RT_imageStorage(RT_imageStorage,dirName,IstFilesDCM,position_list_cut, CT_SOPInstanceUID_all_dict, x_position, y_position,position_min,position_max,Study_InstanceUID,Series_InstanceUID,file_name,dic_mask, list_position_dic):


    CT=pydicom.dcmread(IstFilesDCM[0],force=True)
    now = int(time.time()) 
    timeStruct = time.localtime(now) 
    now_Date = time.strftime("%Y%m%d", timeStruct) 
    now_Time = time.strftime("%H%M%S", timeStruct) 
    InstanceCreationDate=now_Date
    InstanceCreationTime=now_Time
    FrameOfReferenceUID=CT.FrameOfReferenceUID
    ReferencedSOPInstanceUID=CT.StudyInstanceUID
    SeriesInstanceUID=CT.SeriesInstanceUID
    Patient_id=int(CT.PatientID)
    RS_SOPInstanceUID='1.2.246.352.205.'+str(Patient_id)+str(InstanceCreationDate)+'12345.'+str(Patient_id)+str(InstanceCreationTime)+str(InstanceCreationTime)+'1'
    save_path=dirName+'/'+'RTSTRUCT_'+RS_SOPInstanceUID
    RT_imageStorage.SpecificCharacterSet=CT.SpecificCharacterSet#CT
    RT_imageStorage.InstanceCreationDate=InstanceCreationDate#
    RT_imageStorage.InstanceCreationTime=InstanceCreationTime#
    #RT_imageStorage.SOPClassUID='1.2.840.10008.5.1.4.1.1.481.3'
    RT_imageStorage.SOPInstanceUID= RS_SOPInstanceUID#
    RT_imageStorage.StudyDate=CT.StudyDate#studydate
    RT_imageStorage.StudyTime=CT.StudyTime#CTstudytime
    RT_imageStorage.AccessionNumber=''#
    RT_imageStorage.Modality='RTSTRUCT'
    RT_imageStorage.Manufacturer=''#
    RT_imageStorage.ReferringPhysicianName=''
    RT_imageStorage.StationName=''
    RT_imageStorage.StudyDescription=CT.StudyDescription#CT
    RT_imageStorage.SeriesDescription=''
    RT_imageStorage.PatientName=CT.PatientName#CT
    RT_imageStorage.PatientID=CT.PatientID#CT
    RT_imageStorage.PatientBirthDate=CT.PatientBirthDate#CT
    RT_imageStorage.PatientSex=CT.PatientSex#CT
    RT_imageStorage.StudyInstanceUID=CT.StudyInstanceUID#CT
    RT_imageStorage.SeriesInstanceUID=CT.SeriesInstanceUID#CT
    RT_imageStorage.StudyID=CT.StudyID#CT
    RT_imageStorage.SeriesNumber=CT.SeriesNumber#CT
    RT_imageStorage.InstanceNumber=1
    RT_imageStorage.StructureSetLabel='PREDICT'
    RT_imageStorage.StructureSetDate=InstanceCreationDate#InstanceCreationDate
    RT_imageStorage.StructureSetTime=InstanceCreationTime#InstanceCreationTime
    RT_imageStorage.ApprovalStatus='UNAPPROVED'
    return RT_imageStorage

def get_Referenced_Frame_of_Reference_Sequence(RT_imageStorage,CT_SOPInstanceUID_all_dict,Series_InstanceUID,Study_InstanceUID,FrameOfReferenceUID):

    
    n=0
    for all_lis in CT_SOPInstanceUID_all_dict:#CT_SOPInstanceUID_all_dict 
        n+=1
        a_30060016_00081150=RT_imageStorage[0x30060010][0][0x30060012][0][0x30060014][0][0x30060016][0].ReferencedSOPClassUID

        ReferencedSOPInstanceUID=CT_SOPInstanceUID_all_dict[all_lis]
        a_30060016=Dataset()
        a_30060016.add_new(0x00081150,"UI",str(a_30060016_00081150))
        a_30060016.add_new(0x00081155,"UI",str(ReferencedSOPInstanceUID))

        RT_imageStorage[0x30060010][0][0x30060012][0][0x30060014][0][0x30060016].value.append(a_30060016)
    RT_imageStorage[0x30060010][0][0x00200052].value=FrameOfReferenceUID
    RT_imageStorage[0x30060010][0][0x30060012][0][0x00081155].value=Study_InstanceUID
    RT_imageStorage[0x30060010][0][0x30060012][0][0x30060014][0][0x0020000e].value=Series_InstanceUID
    RT_imageStorage[0x30060010][0][0x30060012][0][0x30060014][0][0x30060016].value.remove(RT_imageStorage[0x30060010][0][0x30060012][0][0x30060014][0][0x30060016][0])

    return RT_imageStorage

def get_Structure_SetROI(RT_imageStorage,organ_list,FrameOfReferenceUID):

    for index in range(len(organ_list)):
        organ_name=organ_list[index]
        
        Number=index+1
        tag_30060020=Dataset()
        tag_30060020.add_new(0x30060022,"IS",Number)
        tag_30060020.add_new(0x30060024,"UI",FrameOfReferenceUID)
        tag_30060020.add_new(0x30060026,"LO",organ_name)
        tag_30060020.add_new(0x30060036,"CS","MANUAL")
        
        RT_imageStorage[0x30060020].value.append(tag_30060020)

    RT_imageStorage[0x30060020].value.remove(RT_imageStorage[0x30060020][0])
    return RT_imageStorage

def get_RT_ROIObservationsSequence(RT_imageStorage,organ_list,FrameOfReferenceUID):
    #（4）3006,0080

    for index in range(len(organ_list)):
        Number=index+1
        organ_name=organ_list[index]
        tag_30060080=Dataset()
        tag_30060080.add_new(0x30060082,"IS",Number)
        tag_30060080.add_new(0x30060084,"IS",Number)
        tag_30060080.add_new(0x30060085,"SH",organ_name)
        tag_30060080.add_new(0x300600a4,"CS","ORGAN")
        tag_30060080.add_new(0x300600a6,"PN","")
        
        RT_imageStorage[0x30060080].value.append(tag_30060080)
        

    RT_imageStorage[0x30060080].value.remove(RT_imageStorage[0x30060080][0])
    return RT_imageStorage

def get_ROI_ContourSequence(RT_imageStorage,organ_list,color_list,dic_mask,list_position_dic,CT_SOPInstanceUID_all_dict):#
    for index in range(len(organ_list)):
        organ_name=organ_list[index]
        Number=index+1
        Color=color_list[index]
        
        
        tag_30060039=Dataset()
        tag_30060039.add_new(0x3006002a,"IS",Color)
        tag_30060039.add_new(0x30060084,"IS",Number)
        tag_30060039.add_new(0x30060040,"SQ","")
        

        RT_imageStorage[0x30060039].value.append(tag_30060039)


        
        slice_organ_list=list_position_dic[organ_name]
        dic_mask_organ=dic_mask[organ_name]

        number_0048=0
        for s in slice_organ_list:
            tag_0040_30060016=Dataset()
            ReferenceSOPInstanceUID=CT_SOPInstanceUID_all_dict[s]
            tag_0040_30060016.add_new(0x00081150,"UI","1.2.840.10008.5.1.4.1.1.2")
            tag_0040_30060016.add_new(0x00081155,"UI",ReferenceSOPInstanceUID)
            
            ContourData=dic_mask_organ[s]
            ContourPoints=len(ContourData)/ 3
            number_0048+=1
            tag_30060040=Dataset()
            tag_30060040.add_new(0x30060042,"CS","CLOSED_PLANAR")
            tag_30060040.add_new(0x30060046,"IS",ContourPoints)
            tag_30060040.add_new(0x30060048,"IS",number_0048)
            tag_30060040.add_new(0x30060050,"DS",ContourData)
            tag_30060040.add_new(0x30060016,"SQ","")
            tag_30060040[0x30060016].value.append(tag_0040_30060016)
            RT_imageStorage[0x30060039][Number][0x30060040].value.append(tag_30060040)

        
    RT_imageStorage[0x30060039].value.remove(RT_imageStorage[0x30060039][0])
    return RT_imageStorage


def isInt(num):
    try:
        a = int(str(num))
        return a
    except:
        a=round(num)
        return a
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

#这部分是写入mask的代码，一个是插值之后的CT，另外一个是原始的没有插值的。
def get_organMask_dic(file_name, position_list_cut, mask_path,organ_list, position_min,position_max, x_position, y_position, Organ_mask_coor_stage2):
    b=np.load(mask_path+'/crop/'+file_name+'.npy')
    bb=np.load(mask_path+'/CT_orgin/'+file_name+'.npy')
    l,u,f=center_coor(b)
    l_bb,u_bb,f_bb=center_coor(bb)
    real_bb_position=position_list_cut[f_bb-1]#position_list_cut这个是根据CT的position从大到小进行排列，正好对上f_bb。
    a_value=real_bb_position+f #将原始CT的position和插值之后的相加，插值这个值是从0开始，而CT的position是从大到小，所以他们加起来是一个固定的值
    dic_mask={}
    list_position_dic={}
    for organ in organ_list:
        dic={}

        a=np.load(mask_path+organ+'/pred/'+file_name+'.npy')
        
        list_position=[]

        shape_coord=Organ_mask_coor_stage2[organ]


        for kk in position_list_cut:
            k=a_value-kk #k是转化为npy的z值
            if ((shape_coord[4] <= k) and (shape_coord[5] > k)):
                print(kk)
                print(position_max)
                print(shape_coord[4])
                print(shape_coord[5])
                print(organ)
                i=isInt(k)#因为有些CT的值会产生0.5或者0.25,0.75这种，所以得让它变成临近的整数，对应到npy的z值上。
                if (i == a.shape[2]):
                    i=i-1
                    img=a[:,:,i]
                    contours = measure.find_contours(img, 0.5)
                else:
                    img=a[:,:,i]
                    contours = measure.find_contours(img, 0.5)

                dict_coor=[]

                for n, contour in enumerate(contours):
                    for j in range(np.shape(contour)[0]):
                        x_1=contour[j,1]+ float(x_position)+shape_coord[2]+u_bb-u#* x_PixelSpacing + x_position x,y是相反的np[y,x,z]
                        x=float(x_1)
                        y_1=contour[j,0]+ float(y_position)+shape_coord[0]+l_bb-l#*  + y_position x,y是相反的np[y,x,z]
                        y=float( y_1)
                        i_2=float(kk)
                        dict_coor.append(str(x))
                        dict_coor.append(str(y))
                        dict_coor.append(str(i_2))
                if dict_coor:
                    dic[kk]=dict_coor
                    list_position.append(kk)
                else:
                    continue
                

            else:
                continue

        dic_mask[organ]=dic
        list_position_dic[organ]=list_position
    return dic_mask, list_position_dic

