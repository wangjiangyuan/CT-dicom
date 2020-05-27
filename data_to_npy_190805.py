# -*- coding:utf-8 -*-
import os
import utilize_190805
import logging
import traceback

Train_data_path='/public/home/liulizuo/TrainData/'
PATHList='/public/home/liulizuo/NPC295/'
path_log='/public/home/liulizuo/log_files/'


Target_Organ_list = "Brain stem, L-parotid, L-temporal lobe, Left TM-joint, Left eye, Left lens, Left optic nerve, Mandible, R-parotid, R-temporal lobe, Right TM-joint, Right eye, Right lens, Right optic nerve, Spinal cord, Optic chiasm"

#这个代码的主要作用是完成CT到npy文件的转化工作。集合成这个代码中

def get_data(PATHList,Train_data_path,path_log,i):

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename=path_log+'CT'+ str(i) + '.log',
                        filemode='w')

    os.mkdir(Train_data_path+'/CT')
    Target_Organ = Target_Organ_list.split(", ")
    for organ in Target_Organ:
        # stage 1
        Organ_path = Train_data_path+organ
        os.mkdir(Organ_path )
    counter = 1
    PATH=os.listdir(PATHList)
    for PathDicom in PATH:
        try:
            IstFilesDCM, RsFileDCM = utilize_190805.Get_path(PATHList+PathDicom)
            print('已完成Get_path'+'共计'+str(len(IstFilesDCM))+'个CT文件，以及'+str(len(RsFileDCM))+'个RS文件')
            ORGAN_dict = utilize_190805.load_Contour_dcm(RsFileDCM)
            CT_dict = utilize_190805.load_CT_dcm(IstFilesDCM)

            for key in CT_dict.keys():
                logging.info('Path:%s' % PathDicom)
                logging.info('key:%s' % key)
                pixels = utilize_190805.get_pixels_hu(CT_dict[key])
                shape = pixels.shape
                #mask, Organ_had = utilize_190805.get_mask(ORGAN_dict[key],CT_dict[key],Target_Organ, shape )
                new_spacing, thickness_index, thickness, PixelSpacing, PatientInfo = utilize_190805.resample(Train_data_path, pixels, CT_dict[key], [1,1,1])
                logging.info('Target organs this patient had:%s' % Organ_had)
                logging.info('new_spacing:%s' % new_spacing)
                logging.info('PatientsInfo:%s' % PatientInfo)
                logging.info('thickness:%s' % thickness)
                logging.info('thickness_index:%s' % thickness_index)
                print('已完成'+str(counter)+'个resampple')
            logging.debug("I have finished %d resampling samples." % counter)
            counter += 1
        except:
            s = traceback.format_exc()
            logging.error(s)


