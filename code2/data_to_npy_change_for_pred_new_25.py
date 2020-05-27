# -*- coding:utf-8 -*-
import os
import utilize_write_for_pred_new_25
#import predict2rs_new_fix_bug_for_hunan
import logging
import traceback
path_mask='/public/home/liulizuo/test2020_4/'
PATHList='/public/home/liulizuo/tmp/'
def get_data(PATHList,path_mask,i):

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename='/public/home/liulizuo/log_files/CT'+ str(i) + '.log',
                        filemode='w')


    os.mkdir(path_mask+'/CT')
    os.mkdir(path_mask+'/CT_orgin')

    PATH=os.listdir(PATHList)
    for PathDicom in PATH:
        try:
            IstFilesDCM = utilize_write_for_pred_new_25.Get_path(PATHList+PathDicom)
            print('已完成Get_path'+'共计'+str(len(IstFilesDCM))+'个CT文件')
            CT_dict = utilize_write_for_pred_new_25.load_CT_dcm(IstFilesDCM)
            for key in CT_dict.keys():
                logging.info('Path:%s' % PathDicom)
                logging.info('key:%s' % key)
                pixels = utilize_write_for_pred_new_25.get_pixels_hu(CT_dict[key])
                shape = pixels.shape
                
                new_spacing, thickness_index, thickness, PixelSpacing, PatientInfo,x_coord,y_coord,position_min,position_max = utilize_write_for_pred_new_25.resample_write(path_mask,pixels, CT_dict[key], [1,1,1])

                logging.info('x_coord:%s' % x_coord)
                logging.info('y_coord:%s' % y_coord)
                logging.info('new_spacing:%s' % new_spacing)
                logging.info('PatientsInfo:%s' % PatientInfo)
                logging.info('thickness:%s' % thickness)
                logging.info('thickness_index:%s' % thickness_index)

            logging.debug("I have finished %d resampling samples." % counter)
            counter += 1
        except:
            s = traceback.format_exc()
            logging.error(s)

'''
    Target_Organ_list = "Brain stem, L-parotid, L-temporal lobe, Left TM-joint, Left eye, Left lens, Left optic nerve, Mandible, R-parotid, R-temporal lobe, Right TM-joint, Right eye, Right lens, Right optic nerve, Spinal cord, Optic chiasm"
    #Target_Organ_list ="Mandible, Spinal cord"
    Target_Organ = Target_Organ_list.split(", ")
    for organ in Target_Organ:
        # stage 1
        Organ_path = path_mask+'/'+organ
        os.mkdir(Organ_path)

        os.mkdir(Organ_path +'/cutted')
        
        os.mkdir(Organ_path + '/pred')
        utilize_write_for_pred_new_25.crop_array(path_mask,organ)
        print(organ+': crop ok')

        utilize_write_for_pred_new_25.cut_array(path_mask,Organ_path,organ)
        print(organ+': cut ok')
        utilize_write_for_pred_new_25.pred_mask(Organ_path,organ)
        print(organ+': pred ok')
'''

