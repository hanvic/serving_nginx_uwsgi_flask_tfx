#! /usr/bin/env python
# coding=utf-8

import os
import cv2
import random
import numpy as np

from PIL import Image


classes = ['background','aeroplane','bicycle','bird','boat',
           'bottle','bus','car','cat','chair','cow','diningtable',
           'dog','horse','motorbike','person','potted plant',
           'sheep','sofa','train','tv/monitor']
# RGB color for each class
colormap = [[0,0,0],[0,0,255],[255,0,0], [255,255,0], [0,128,255],[255,0,255],
            #,[0,128,128],[128,128,128],[64,0,0],[192,0,0],
            [64,128,0],[192,128,0],[64,0,128],[192,0,128],
            [64,128,128],[192,128,128],[0,64,0],[128,64,0]]
            #[0,192,0],[128,192,0],[0,64,128]]

rgb_mean = np.array([0.485, 0.456, 0.406])
rgb_std = np.array([0.229, 0.224, 0.225])

def single_image_visual_result(image,  teeth_prediction, prediction, root_target_features,target_features, threshold, alpha=0.5):
    """
    image shape -> [H, W, C]
    label shape -> [H, W]
    """
    # image = (image * rgb_std + rgb_mean) * 255
    image=np.asarray(image)


    teeth_prediction=np.asarray(teeth_prediction)
    prediction=np.asarray(prediction)


    teeth_prediction=teeth_prediction[0]
    prediction=prediction[0]
    numpy_prediction = np.zeros_like(prediction)
    numpy_teeth_prediction = np.zeros_like(teeth_prediction)

    H, W, C = image.shape
    # H, W, nC = prediction.shape
    masks_color = np.zeros(shape=[H, W, C])
    teeth_masks_color = np.zeros(shape=[H, W, C])
    tar_masks_color = np.zeros(shape=[H, W, C])
    lables_color = np.zeros(shape=[H, W, C])
    inv_masks_color = np.zeros(shape=[H, W, C])
    result_image = np.zeros_like(image)

    result_image=np.zeros_like(image)
    no2name_dict = dict()
    no2name_dict['1'] = 'dental_calculus'
    no2name_dict['2'] = 'tooth_root'
    no2name_dict['3'] = 'gingiva_dehiscence'
    no2name_dict['4'] = 'normal'
    no2name_dict['5'] = 'tar'

    result_dict = dict()
    result_dict['dental_calculus'] = 0
    result_dict['tooth_root'] = 0
    result_dict['gingiva_dehiscence'] = 0
    result_dict['normal'] = 0
    result_dict['tar'] = 0

    result_prob_dict = dict()
    result_prob_dict['dental_calculus'] = 0
    result_prob_dict['tooth_root'] = 0
    result_prob_dict['gingiva_dehiscence'] = 0
    result_prob_dict['normal'] = 0
    result_prob_dict['tar'] = 0

    temp_valid_label_dict = dict()

    cls = []
    c_i = 1
    nC = prediction.shape[-1]
    for k in range(nC):
        tag_label = 0
        temp_valid_label_dict['1'] = 0
        temp_valid_label_dict['2'] = 0
        temp_valid_label_dict['3'] = 0
        temp_valid_label_dict['4'] = 0
        if k in target_features:
            NofValidPixels = 0
            teeth_NofValidPixels = 0
            tar_NofValidPixels = 0
            maxRatio = 0
            maxtarRatio = 0
        # if np.max(label[...,k])> 0 and k> 0:
            # print("no",no, np.max(label[...,k]), len(np.where(label[...,k]>0)[0]))
            print("no", np.max(prediction[..., k]))
            for i in range(H):
                for j in range(W):
                    if prediction[i, j, k] > 0.1:
                        numpy_prediction[i, j, k] = 1
                        # numpy_target_prediction[i, j, k] = 1
                        NofValidPixels += 1
                        tar_NofValidPixels+=1
                        if maxtarRatio < prediction[i, j, k]:
                            maxtarRatio = prediction[i, j, k]
                    else:
                        numpy_prediction[i, j, k] = 0
                        # numpy_target_prediction[i, j, k] = 0
                    # cls_idx = label[i, j, k]

                    if teeth_prediction[i, j, k] > 0.1:
                        numpy_teeth_prediction[i, j, k] = 1
                        # numpy_target_prediction[i, j, k] = 1
                        teeth_NofValidPixels += 1
                        if maxRatio < prediction[i, j, k]:
                            maxRatio = prediction[i, j, k]
                    else:
                        numpy_teeth_prediction[i, j, k] = 0
                        # numpy_target_prediction[i, j, k] = 0

                    # for m in target_features:
                    #     if label[i, j, m] > 0:
                    #         temp_valid_label_dict[str(m)] = 1

                    # if cls_idx >0 and k>0:
                    #     tag_label=1
                    #     # temp_valid_label_dict[str(k)]=1
                    #     lables_color[i, j] = np.array(colormap[3])
                    # cls.append(cls_idx)
                    # else:
                    #     lables_color[i, j] = np.array(colormap[0])

                    if numpy_prediction[i, j, k] > 0.5 and k > 0:
                        masks_color[i, j] = np.array(colormap[1])
                        # cls.append(cls_idx)
                    else:
                        masks_color[i, j] = np.array(colormap[0])

                    if numpy_teeth_prediction[i, j, k] > 0.5 and k == 2:
                        teeth_masks_color[i, j] = np.array(colormap[2])
                        # cls.append(cls_idx)
                    # else:
                    #     tar_masks_color[i, j] = np.array(colormap[0])

                    if numpy_prediction[i, j, k] > 0.5 and k == 1:
                        tar_masks_color[i, j] = np.array(colormap[1])
                        # cls.append(cls_idx)
                    # else:
                    #     teeth_masks_color[i, j] = np.array(colormap[0])

            c_i += 1

            NofPixels = masks_color.shape[0] * masks_color.shape[1]
            if tar_NofValidPixels / NofPixels < 0.9:
                if tar_NofValidPixels / NofPixels > 0.001:
                    result_dict['tar'] = 1
                    result_prob_dict['tar'] = maxtarRatio

            if k == 1 or k == 2:
                masks_color = teeth_masks_color
                masks_color = masks_color.astype(np.uint8)
                tar_masks_color = tar_masks_color.astype(np.uint8)
                teeth_masks_color = teeth_masks_color.astype(np.uint8)
                show_image = np.zeros(shape=[512, 1024, 3])
                cls = set(cls)
                # /
                NofPixels = masks_color.shape[0] * masks_color.shape[1]

                # NofValidPixels = tar_NofValidPixels

                print("ratio valid pixels : ", NofValidPixels / NofPixels)
                if NofValidPixels / NofPixels < 0.9:
                    if NofValidPixels / NofPixels > 0.001:
                        result_image += masks_color + lables_color.astype(np.uint8)
                        disease_name = no2name_dict[str(c_i - 1)]
                        result_dict[disease_name] = 1
                        result_prob_dict[disease_name] = maxRatio



                    else:
                        masks_color = np.zeros(shape=[H, W, C]).astype(np.uint8)
                        # temp_sum_not_back = 0


                            # temp_sum_not_back += temp_valid_label_dict[str(m)]
                        # if temp_sum_not_back == 0:
                        #     confusion_matrix[c_i - 1, 0] += 1

                # show_image = np.floor((1-alpha)*image) + np.floor((alpha*2/3)*masks_color) + np.floor((alpha/3)*lables_color)
                # show_image = (1-alpha)*image + alpha*lables_color
                show_image[:, :512, :] = image
                show_image[:, 512:, :] = masks_color + lables_color.astype(np.uint8)
                # print("no", k, np.max(masks_color), len(np.where(masks_color > 0)[0]))
                # show_image = Image.fromarray(np.uint8(show_image))
                show_image = show_image.astype(np.uint8)
                # base_save_folder = '/home/projects/src/refineData/outputs_2nd/snapshot_tar/'
                # if not os.path.isdir(base_save_folder):
                #     os.makedirs(base_save_folder)
                #     os.chmod(base_save_folder, 0o777)
            # else:
            #     result_image += np.zeros_like(masks_color)
            # show_image = np.floor((1-alpha)*image) + np.floor((alpha*2/3)*masks_color) + np.floor((alpha/3)*lables_color)
            # show_image = (1-alpha)*image + alpha*lables_color
            # result_image = (1-alpha)*image + alpha*result_image
            # show_image = np.floor(show_image).astype(np.uint8)
            # show_image[:,:512,:] = image
            # show_image[:,512:,:] = masks_color
            # print("no", no, np.max(masks_color), len(np.where(masks_color > 0)[0]))
            # show_image = Image.fromarray(np.uint8(show_image))
            # show_image = show_image.astype(np.uint8)
            # base_save_folder = '/home/projects/src/refineData/outputs/snapshot/'
            # if not os.path.isdir(base_save_folder):
            #     os.makedirs(base_save_folder)
            # cv2.imwrite(base_save_folder+'/one_shot_no_{}.png'.format(k),show_image)
    # cv2.imwrite(base_save_folder+'/result_no.png'.format(k),result_image)
            # cv2.imwrite(base_save_folder+'/original_no_{}_{}.png'.format(no,k),image)
            # cv2.WaitKey(0)
    # print(result_dict)
    print(result_dict)
    return result_dict, result_prob_dict, result_image

def visual_result(no, target_features, image, prediction,label,alpha=0.5):
    """
    image shape -> [H, W, C]
    label shape -> [H, W]
    """
    image, label = np.asarray(image).astype(np.uint8), np.asarray(label).astype(np.uint8)
    prediction = np.asarray(prediction)
    image = image[0]

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    prediction = prediction[0]
    numpy_prediction = np.zeros_like(prediction)
    label = label[0]

    H, W, C = image.shape
    H, W, nC = label.shape
    masks_color = np.zeros(shape=[H, W, C])
    lables_color = np.zeros(shape=[H, W, C])
    inv_masks_color = np.zeros(shape=[H, W, C])
    result_image = np.zeros_like(image)
    no2name_dict = dict()
    no2name_dict['1'] = 'Third_eyelid_protrude'
    no2name_dict['2'] = 'blepharitis_inflammation'
    no2name_dict['3'] = 'blepharitis_inner_inflammation'
    no2name_dict['4'] = 'corneal_pus'
    no2name_dict['5'] = 'corneal_scratch'
    no2name_dict['6'] = 'corneal'
    no2name_dict['7'] = 'conjunctivitis_flare'
    no2name_dict['8'] = 'conjunctivitis_swll'
    no2name_dict['9'] = 'conjunctivitis_white_inflammation'
    no2name_dict['10'] = 'gataract'
    no2name_dict['11'] = 'gataract_initial'

    result_dict = dict()
    result_dict['Third_eyelid_protrude'] = 0
    result_dict['blepharitis_inflammation'] = 0
    result_dict['blepharitis_inner_inflammation'] = 0
    result_dict['corneal_pus'] = 0
    result_dict['corneal_scratch'] = 0
    result_dict['corneal'] = 0
    result_dict['conjunctivitis_flare'] = 0
    result_dict['conjunctivitis_swll'] = 0
    result_dict['conjunctivitis_white_inflammation'] = 0
    result_dict['gataract'] = 0
    result_dict['gataract_initial'] = 0

    result_prob_dict = dict()
    result_prob_dict['Third_eyelid_protrude'] = 0
    result_prob_dict['blepharitis_inflammation'] = 0
    result_prob_dict['blepharitis_inner_inflammation'] = 0
    result_prob_dict['corneal_pus'] = 0
    result_prob_dict['corneal_scratch'] = 0
    result_prob_dict['corneal'] = 0
    result_prob_dict['conjunctivitis_flare'] = 0
    result_prob_dict['conjunctivitis_swll'] = 0
    result_prob_dict['conjunctivitis_white_inflammation'] = 0
    result_prob_dict['gataract'] = 0
    result_prob_dict['gataract_initial'] = 0

    cls = []
    c_i=1
    for k in range(nC):
        tag_label=0
        if k in target_features:
            NofValidPixels = 0
            maxRatio = 0
        # if np.max(label[...,k])> 0 and k> 0:
            print("no",no, np.max(prediction[...,k]))
            for i in range(H):
                for j in range(W):
                    if prediction[i, j,k] > 0.3:
                        numpy_prediction[i, j, k] = 1
                        NofValidPixels += 1
                        if maxRatio < prediction[i, j, k]:
                            maxRatio = prediction[i, j, k]
                    else:
                        numpy_prediction[i, j, k] = 0
                    cls_idx = label[i, j, k]

                    if cls_idx >0 and k>0:
                        tag_label=1
                        lables_color[i, j] = np.array(colormap[2])
                        # cls.append(cls_idx)
                    else:
                        lables_color[i, j] = np.array(colormap[0])

                    if numpy_prediction[i, j, k] >0.5 and k>0:
                        masks_color[i, j] = np.array(colormap[1])
                        # cls.append(cls_idx)
                    else:
                        masks_color[i, j] = np.array(colormap[0])



            c_i += 1
            masks_color = masks_color.astype(np.uint8)
            show_image = np.zeros(shape=[512, 1024, 3])
            cls = set(cls)
            # /
            NofPixels = masks_color.shape[0] * masks_color.shape[1]
            NofValidPixels = NofValidPixels
            print("ratio valid pixels : ", NofValidPixels / NofPixels)
            if NofValidPixels/NofPixels<0.9:
                if NofValidPixels/NofPixels>0.01:
                    result_image += masks_color + lables_color.astype(np.uint8)
                    disease_name=no2name_dict[str(c_i-1)]
                    result_dict[disease_name]+=1
                    result_prob_dict[disease_name]=maxRatio
                else:
                    masks_color = np.zeros(shape=[H, W, C]).astype(np.uint8)



            # show_image = np.floor((1-alpha)*image) + np.floor((alpha*2/3)*masks_color) + np.floor((alpha/3)*lables_color)
            # show_image = (1-alpha)*image + alpha*lables_color
            show_image[:,:512,:] = image
            show_image[:,512:,:] = masks_color+lables_color.astype(np.uint8)
            # print("no", k, np.max(masks_color), len(np.where(masks_color > 0)[0]))
            # show_image = Image.fromarray(np.uint8(show_image))
            show_image = show_image.astype(np.uint8)
            base_save_folder = '/home/projects/src/refineData/outputs_2nd/snapshot/'
            if not os.path.isdir(base_save_folder):
                os.makedirs(base_save_folder)
                os.chmod(base_save_folder,0o777)
            if tag_label==1:
                cv2.imwrite(base_save_folder + '/sol_no_{}_{}.png'.format(no, k), show_image)
            else:
                cv2.imwrite(base_save_folder + '/snapshot_no_{}_{}.png'.format(no, k), show_image)
    # cv2.imwrite(base_save_folder + '/result_no_{}_{}.png'.format(no, k), result_image)
            # cv2.imwrite(base_save_folder+'/original_no_{}_{}.png'.format(no,k),image)
            # cv2.WaitKey(0)
    print(result_dict)

def visual_result_teeth(no, target_features, image, prediction,label,confusion_matrix,alpha=0.5):
    """
    image shape -> [H, W, C]
    label shape -> [H, W]
    """
    image, label = np.asarray(image).astype(np.uint8), np.asarray(label).astype(np.uint8)
    prediction = np.asarray(prediction)
    image = image[0]

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    prediction = prediction[0]
    numpy_prediction = np.zeros_like(prediction)
    label = label[0]

    H, W, C = image.shape
    H, W, nC = label.shape
    masks_color = np.zeros(shape=[H, W, C])
    lables_color = np.zeros(shape=[H, W, C])
    inv_masks_color = np.zeros(shape=[H, W, C])
    result_image = np.zeros_like(image)
    # no2name_dict = dict()
    # no2name_dict['1'] = 'dental_calculus'
    # no2name_dict['2'] = 'tooth_root'
    # no2name_dict['3'] = 'gingiva_dehiscence'
    # no2name_dict['4'] = 'normal'
    #
    # result_dict = dict()
    # result_dict['dental_calculus'] = 0
    # result_dict['tooth_root'] = 0
    # result_dict['gingiva_dehiscence'] = 0
    # result_dict['normal'] = 0
    #
    # result_prob_dict = dict()
    # result_prob_dict['dental_calculus'] = 0
    # result_prob_dict['tooth_root'] = 0
    # result_prob_dict['gingiva_dehiscence'] = 0
    # result_prob_dict['normal'] = 0
    #
    #
    # temp_valid_label_dict = dict()


    no2name_dict = dict()
    no2name_dict['1'] = 'yellow_pus'
    no2name_dict['2'] = 'dark_brown_earwax'
    no2name_dict['3'] = 'dark_brown_scab'
    no2name_dict['4'] = 'light_shadow'
    no2name_dict['5'] = 'deep_shadow'
    # no2name_dict['4'] = 'normal'

    result_dict = dict()
    result_dict['dental_calculus'] = 0
    result_dict['dark_brown_earwax'] = 0
    result_dict['dark_brown_scab'] = 0
    result_dict['light_shadow'] = 0
    result_dict['deep_shadow'] = 0
    # result_dict['normal'] = 0

    result_prob_dict = dict()
    result_prob_dict['dental_calculus'] = 0
    result_prob_dict['dark_brown_earwax'] = 0
    result_prob_dict['dark_brown_scab'] = 0
    result_prob_dict['light_shadow'] = 0
    result_prob_dict['deep_shadow'] = 0
    # result_prob_dict['normal'] = 0


    temp_valid_label_dict = dict()



    cls = []
    c_i=1
    for k in range(nC):
        tag_label=0
        temp_valid_label_dict['1'] = 0
        temp_valid_label_dict['2'] = 0
        temp_valid_label_dict['3'] = 0
        temp_valid_label_dict['4'] = 0
        temp_valid_label_dict['5'] = 0
        # temp_valid_label_dict['4'] = 0
        if k in target_features:
            NofValidPixels = 0
            maxRatio = 0
        # if np.max(label[...,k])> 0 and k> 0:
            print("no",no, np.max(prediction[...,k]))
            for i in range(H):
                for j in range(W):
                    if prediction[i, j,k] > 0.1:
                        numpy_prediction[i, j, k] = 1
                        NofValidPixels += 1
                        if maxRatio < prediction[i, j, k]:
                            maxRatio = prediction[i, j, k]
                    else:
                        numpy_prediction[i, j, k] = 0
                    cls_idx = label[i, j, k]

                    for m in target_features:
                        if label[i, j, m] > 0:
                            temp_valid_label_dict[str(m)] = 1


                    if cls_idx >0 and k>0:
                        tag_label=1
                        # temp_valid_label_dict[str(k)]=1
                        lables_color[i, j] = np.array(colormap[2])
                        # cls.append(cls_idx)
                    else:
                        lables_color[i, j] = np.array(colormap[0])

                    if numpy_prediction[i, j, k] >0.5 and k>0:
                        masks_color[i, j] = np.array(colormap[1])
                        # cls.append(cls_idx)
                    else:
                        masks_color[i, j] = np.array(colormap[0])
            c_i += 1
            masks_color = masks_color.astype(np.uint8)
            show_image = np.zeros(shape=[512, 1024, 3])
            cls = set(cls)
            # /
            NofPixels = masks_color.shape[0] * masks_color.shape[1]
            NofValidPixels = NofValidPixels

            print("ratio valid pixels : ", NofValidPixels / NofPixels)
            if NofValidPixels/NofPixels<0.9:
                if NofValidPixels/NofPixels>0.001:
                    result_image += masks_color + lables_color.astype(np.uint8)
                    disease_name=no2name_dict[str(c_i-1)]
                    result_dict[disease_name]=1
                    result_prob_dict[disease_name]=maxRatio
                    temp_sum_not_back=0
                    for m in target_features:
                        if temp_valid_label_dict[str(m)] > 0:
                            confusion_matrix[c_i-1, m] += 1
                        temp_sum_not_back+=temp_valid_label_dict[str(m)]
                    if temp_sum_not_back==0:
                        confusion_matrix[c_i - 1, 0] += 1


                else:
                    masks_color = np.zeros(shape=[H, W, C]).astype(np.uint8)
                    # temp_sum_not_back = 0
                    for m in target_features:
                        if temp_valid_label_dict[str(m)] > 0:
                            confusion_matrix[0, m] += 1
            else:
                for m in target_features:
                    if temp_valid_label_dict[str(m)] > 0:
                        confusion_matrix[0, m] += 1
                        # temp_sum_not_back += temp_valid_label_dict[str(m)]
                    # if temp_sum_not_back == 0:
                    #     confusion_matrix[c_i - 1, 0] += 1




            # show_image = np.floor((1-alpha)*image) + np.floor((alpha*2/3)*masks_color) + np.floor((alpha/3)*lables_color)
            # show_image = (1-alpha)*image + alpha*lables_color
            show_image[:,:512,:] = image
            show_image[:,512:,:] = masks_color+lables_color.astype(np.uint8)
            # print("no", k, np.max(masks_color), len(np.where(masks_color > 0)[0]))
            # show_image = Image.fromarray(np.uint8(show_image))
            show_image = show_image.astype(np.uint8)
            base_save_folder = '/home/projects/src/refineData/outputs_2nd/snapshot_ear/'
            if not os.path.isdir(base_save_folder):
                os.makedirs(base_save_folder)
                os.chmod(base_save_folder,0o777)
            if tag_label==1:
                cv2.imwrite(base_save_folder + '/sol_no_{}_{}.png'.format(no, k), show_image)
            else:
                cv2.imwrite(base_save_folder + '/snapshot_no_{}_{}.png'.format(no, k), show_image)
    # cv2.imwrite(base_save_folder + '/result_no_{}_{}.png'.format(no, k), result_image)
            # cv2.imwrite(base_save_folder+'/original_no_{}_{}.png'.format(no,k),image)
            # cv2.WaitKey(0)
    print(result_dict)
    return confusion_matrix
    # return show_image

def visual_result_teeth_tar(no, target_features, image, prediction,teeth_prediction,label,confusion_matrix,alpha=0.5):
    """
    image shape -> [H, W, C]
    label shape -> [H, W]
    """
    image, label = np.asarray(image).astype(np.uint8), np.asarray(label).astype(np.uint8)
    prediction = np.asarray(prediction)
    teeth_prediction = np.asarray(teeth_prediction)
    image = image[0]

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    prediction = prediction[0]
    teeth_prediction = teeth_prediction[0]
    numpy_prediction = np.zeros_like(prediction)
    numpy_teeth_prediction = np.zeros_like(teeth_prediction)

    label = label[0]

    H, W, C = image.shape
    H, W, nC = label.shape
    masks_color = np.zeros(shape=[H, W, C])
    teeth_masks_color = np.zeros(shape=[H, W, C])
    tar_masks_color = np.zeros(shape=[H, W, C])
    lables_color = np.zeros(shape=[H, W, C])
    inv_masks_color = np.zeros(shape=[H, W, C])
    result_image = np.zeros_like(image)
    no2name_dict = dict()
    no2name_dict['1'] = 'dental_calculus'
    no2name_dict['2'] = 'tooth_root'
    no2name_dict['3'] = 'gingiva_dehiscence'
    no2name_dict['4'] = 'normal'

    result_dict = dict()
    result_dict['dental_calculus'] = 0
    result_dict['tooth_root'] = 0
    result_dict['gingiva_dehiscence'] = 0
    result_dict['normal'] = 0

    result_prob_dict = dict()
    result_prob_dict['dental_calculus'] = 0
    result_prob_dict['tooth_root'] = 0
    result_prob_dict['gingiva_dehiscence'] = 0
    result_prob_dict['normal'] = 0


    temp_valid_label_dict = dict()

    cls = []
    c_i=1
    for k in range(nC):
        tag_label=0
        temp_valid_label_dict['1'] = 0
        temp_valid_label_dict['2'] = 0
        temp_valid_label_dict['3'] = 0
        temp_valid_label_dict['4'] = 0
        if k in target_features:
            NofValidPixels = 0
            teeth_NofValidPixels = 0
            maxRatio = 0
        # if np.max(label[...,k])> 0 and k> 0:
            print("no",no, np.max(prediction[...,k]))

            for i in range(H):
                for j in range(W):
                    if prediction[i, j,k] > 0.1:
                        numpy_prediction[i, j, k] = 1
                        # numpy_target_prediction[i, j, k] = 1
                        NofValidPixels += 1
                        if maxRatio < prediction[i, j, k]:
                            maxRatio = prediction[i, j, k]
                    else:
                        numpy_prediction[i, j, k] = 0
                        # numpy_target_prediction[i, j, k] = 0
                    cls_idx = label[i, j, k]

                    if teeth_prediction[i, j,k] > 0.1:
                        numpy_teeth_prediction[i, j, k] = 1
                        # numpy_target_prediction[i, j, k] = 1
                        teeth_NofValidPixels += 1
                        if maxRatio < prediction[i, j, k]:
                            maxRatio = prediction[i, j, k]
                    else:
                        numpy_teeth_prediction[i, j, k] = 0
                        # numpy_target_prediction[i, j, k] = 0


                    for m in target_features:
                        if label[i, j, m] > 0:
                            temp_valid_label_dict[str(m)] = 1


                    if cls_idx >0 and k>0:
                        tag_label=1
                        # temp_valid_label_dict[str(k)]=1
                        lables_color[i, j] = np.array(colormap[3])
                        # cls.append(cls_idx)
                    # else:
                    #     lables_color[i, j] = np.array(colormap[0])

                    if numpy_prediction[i, j, k] >0.5 and k>0:
                        masks_color[i, j] = np.array(colormap[1])
                        # cls.append(cls_idx)
                    else:
                        masks_color[i, j] = np.array(colormap[0])

                    if numpy_teeth_prediction[i, j, k] >0.5 and k==2:
                        teeth_masks_color[i, j] = np.array(colormap[2])
                        # cls.append(cls_idx)
                    # else:
                    #     tar_masks_color[i, j] = np.array(colormap[0])

                    if numpy_prediction[i, j, k] >0.5 and k==1:
                        tar_masks_color[i, j] = np.array(colormap[1])
                        # cls.append(cls_idx)
                    # else:
                    #     teeth_masks_color[i, j] = np.array(colormap[0])


            c_i += 1
            if k==1 or k==2:
                masks_color = teeth_masks_color
                masks_color = masks_color.astype(np.uint8)
                tar_masks_color = tar_masks_color.astype(np.uint8)
                teeth_masks_color = teeth_masks_color.astype(np.uint8)
                show_image = np.zeros(shape=[512, 1024, 3])
                cls = set(cls)
                # /
                NofPixels = masks_color.shape[0] * masks_color.shape[1]
                NofValidPixels = teeth_NofValidPixels

                print("ratio valid pixels : ", NofValidPixels / NofPixels)
                if NofValidPixels/NofPixels<0.9:
                    if NofValidPixels/NofPixels>0.001:
                        result_image += masks_color + lables_color.astype(np.uint8)
                        disease_name=no2name_dict[str(c_i-1)]
                        result_dict[disease_name]=1
                        result_prob_dict[disease_name]=maxRatio
                        temp_sum_not_back=0
                        for m in target_features:
                            if temp_valid_label_dict[str(m)] > 0:
                                confusion_matrix[c_i-1, m] += 1
                            temp_sum_not_back+=temp_valid_label_dict[str(m)]
                        if temp_sum_not_back==0:
                            confusion_matrix[c_i - 1, 0] += 1


                    else:
                        masks_color = np.zeros(shape=[H, W, C]).astype(np.uint8)
                        # temp_sum_not_back = 0
                        for m in target_features:
                            if temp_valid_label_dict[str(m)] > 0:
                                confusion_matrix[0, m] += 1
                else:
                    for m in target_features:
                        if temp_valid_label_dict[str(m)] > 0:
                            confusion_matrix[0, m] += 1
                            # temp_sum_not_back += temp_valid_label_dict[str(m)]
                        # if temp_sum_not_back == 0:
                        #     confusion_matrix[c_i - 1, 0] += 1




                # show_image = np.floor((1-alpha)*image) + np.floor((alpha*2/3)*masks_color) + np.floor((alpha/3)*lables_color)
                # show_image = (1-alpha)*image + alpha*lables_color
                show_image[:,:512,:] = image
                show_image[:,512:,:] = masks_color+lables_color.astype(np.uint8)
                # print("no", k, np.max(masks_color), len(np.where(masks_color > 0)[0]))
                # show_image = Image.fromarray(np.uint8(show_image))
                show_image = show_image.astype(np.uint8)
                base_save_folder = '/home/projects/src/refineData/outputs_2nd/snapshot_tar/'
                if not os.path.isdir(base_save_folder):
                    os.makedirs(base_save_folder)
                    os.chmod(base_save_folder,0o777)
                # if tag_label==1:
                #     cv2.imwrite(base_save_folder + '/sol_no_{}_{}.png'.format(no, k), show_image)
                # else:
                #     cv2.imwrite(base_save_folder + '/snapshot_no_{}_{}.png'.format(no, k), show_image)
        # cv2.imwrite(base_save_folder + '/result_no_{}_{}.png'.format(no, k), result_image)
                # cv2.imwrite(base_save_folder+'/original_no_{}_{}.png'.format(no,k),image)
                # cv2.WaitKey(0)
    show_image = np.zeros(shape=[512, 1024, 3])
    show_image = show_image.astype(np.uint8)
    show_image[:, :512, :] = image
    show_image[:, 512:, :] = teeth_masks_color + tar_masks_color + lables_color
    # print("no", k, np.max(masks_color), len(np.where(masks_color > 0)[0]))
    # show_image = Image.fromarray(np.uint8(show_image))
    show_image = show_image.astype(np.uint8)
    base_save_folder = '/home/projects/src/refineData/outputs_teeth/snapshot_tar_teeth/'
    if not os.path.isdir(base_save_folder):
        os.makedirs(base_save_folder)
        os.chmod(base_save_folder, 0o777)

    cv2.imwrite(base_save_folder + '/sol_no_{}.png'.format(no), show_image)

    # cv2.imwrite(base_save_folder + '/result_no_{}_{}.png'.format(no, k), result_image)
    # cv2.imwrite(base_save_folder+'/original_no_{}_{}.png'.format(no,k),image)
    # cv2.WaitKey(0)



    print(result_dict)
    return confusion_matrix
    # return show_image

def visual_result_only_img_teeth_tar(no, target_features, image, prediction,teeth_prediction,confusion_matrix,alpha=0.5):
    """
    image shape -> [H, W, C]
    label shape -> [H, W]
    """
    image = np.asarray(image).astype(np.uint8)
    prediction = np.asarray(prediction)
    teeth_prediction = np.asarray(teeth_prediction)
    image = image[0]

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    prediction = prediction[0]
    teeth_prediction = teeth_prediction[0]
    numpy_prediction = np.zeros_like(prediction)
    numpy_teeth_prediction = np.zeros_like(teeth_prediction)



    H, W, C = image.shape

    masks_color = np.zeros(shape=[H, W, C])
    teeth_masks_color = np.zeros(shape=[H, W, C])
    tar_masks_color = np.zeros(shape=[H, W, C])
    lables_color = np.zeros(shape=[H, W, C])
    inv_masks_color = np.zeros(shape=[H, W, C])
    result_image = np.zeros_like(image)
    no2name_dict = dict()
    no2name_dict['1'] = 'dental_calculus'
    no2name_dict['2'] = 'tooth_root'
    no2name_dict['3'] = 'gingiva_dehiscence'
    no2name_dict['4'] = 'normal'

    result_dict = dict()
    result_dict['dental_calculus'] = 0
    result_dict['tooth_root'] = 0
    result_dict['gingiva_dehiscence'] = 0
    result_dict['normal'] = 0

    result_prob_dict = dict()
    result_prob_dict['dental_calculus'] = 0
    result_prob_dict['tooth_root'] = 0
    result_prob_dict['gingiva_dehiscence'] = 0
    result_prob_dict['normal'] = 0


    temp_valid_label_dict = dict()

    cls = []
    c_i=1
    nC = prediction.shape[-1]
    for k in range(nC):
        tag_label=0
        temp_valid_label_dict['1'] = 0
        temp_valid_label_dict['2'] = 0
        temp_valid_label_dict['3'] = 0
        temp_valid_label_dict['4'] = 0
        if k in target_features:
            NofValidPixels = 0
            teeth_NofValidPixels = 0
            maxRatio = 0
        # if np.max(label[...,k])> 0 and k> 0:
            print("no",no, np.max(prediction[...,k]))

            for i in range(H):
                for j in range(W):
                    if prediction[i, j,k] > 0.1:
                        numpy_prediction[i, j, k] = 1
                        # numpy_target_prediction[i, j, k] = 1
                        NofValidPixels += 1
                        if maxRatio < prediction[i, j, k]:
                            maxRatio = prediction[i, j, k]
                    else:
                        numpy_prediction[i, j, k] = 0
                        # numpy_target_prediction[i, j, k] = 0
                    # cls_idx = label[i, j, k]

                    if teeth_prediction[i, j,k] > 0.1:
                        numpy_teeth_prediction[i, j, k] = 1
                        # numpy_target_prediction[i, j, k] = 1
                        teeth_NofValidPixels += 1
                        if maxRatio < prediction[i, j, k]:
                            maxRatio = prediction[i, j, k]
                    else:
                        numpy_teeth_prediction[i, j, k] = 0
                        # numpy_target_prediction[i, j, k] = 0


                    # for m in target_features:
                    #     if label[i, j, m] > 0:
                    #         temp_valid_label_dict[str(m)] = 1


                    # if cls_idx >0 and k>0:
                    #     tag_label=1
                    #     # temp_valid_label_dict[str(k)]=1
                    #     lables_color[i, j] = np.array(colormap[3])
                        # cls.append(cls_idx)
                    # else:
                    #     lables_color[i, j] = np.array(colormap[0])

                    if numpy_prediction[i, j, k] >0.5 and k>0:
                        masks_color[i, j] = np.array(colormap[1])
                        # cls.append(cls_idx)
                    else:
                        masks_color[i, j] = np.array(colormap[0])

                    if numpy_teeth_prediction[i, j, k] >0.5 and k==2:
                        teeth_masks_color[i, j] = np.array(colormap[2])
                        # cls.append(cls_idx)
                    # else:
                    #     tar_masks_color[i, j] = np.array(colormap[0])

                    if numpy_prediction[i, j, k] >0.5 and k==1:
                        tar_masks_color[i, j] = np.array(colormap[1])
                        # cls.append(cls_idx)
                    # else:
                    #     teeth_masks_color[i, j] = np.array(colormap[0])


            c_i += 1
            if k==1 or k==2:
                masks_color = teeth_masks_color
                masks_color = masks_color.astype(np.uint8)
                tar_masks_color = tar_masks_color.astype(np.uint8)
                teeth_masks_color = teeth_masks_color.astype(np.uint8)
                show_image = np.zeros(shape=[512, 1024, 3])
                cls = set(cls)
                # /
                NofPixels = masks_color.shape[0] * masks_color.shape[1]
                NofValidPixels = teeth_NofValidPixels

                print("ratio valid pixels : ", NofValidPixels / NofPixels)
                if NofValidPixels/NofPixels<0.9:
                    if NofValidPixels/NofPixels>0.001:
                        result_image += masks_color + lables_color.astype(np.uint8)
                        disease_name=no2name_dict[str(c_i-1)]
                        result_dict[disease_name]=1
                        result_prob_dict[disease_name]=maxRatio
                        temp_sum_not_back=0
                        for m in target_features:
                            if temp_valid_label_dict[str(m)] > 0:
                                confusion_matrix[c_i-1, m] += 1
                            temp_sum_not_back+=temp_valid_label_dict[str(m)]
                        if temp_sum_not_back==0:
                            confusion_matrix[c_i - 1, 0] += 1


                    else:
                        masks_color = np.zeros(shape=[H, W, C]).astype(np.uint8)
                        # temp_sum_not_back = 0
                        for m in target_features:
                            if temp_valid_label_dict[str(m)] > 0:
                                confusion_matrix[0, m] += 1
                else:
                    for m in target_features:
                        if temp_valid_label_dict[str(m)] > 0:
                            confusion_matrix[0, m] += 1
                            # temp_sum_not_back += temp_valid_label_dict[str(m)]
                        # if temp_sum_not_back == 0:
                        #     confusion_matrix[c_i - 1, 0] += 1




                # show_image = np.floor((1-alpha)*image) + np.floor((alpha*2/3)*masks_color) + np.floor((alpha/3)*lables_color)
                # show_image = (1-alpha)*image + alpha*lables_color
                show_image[:,:512,:] = image
                show_image[:,512:,:] = masks_color+lables_color.astype(np.uint8)
                # print("no", k, np.max(masks_color), len(np.where(masks_color > 0)[0]))
                # show_image = Image.fromarray(np.uint8(show_image))
                show_image = show_image.astype(np.uint8)
                base_save_folder = '/home/projects/src/refineData/outputs_2nd/snapshot_tar/'
                if not os.path.isdir(base_save_folder):
                    os.makedirs(base_save_folder)
                    os.chmod(base_save_folder,0o777)
                # if tag_label==1:
                #     cv2.imwrite(base_save_folder + '/sol_no_{}_{}.png'.format(no, k), show_image)
                # else:
                #     cv2.imwrite(base_save_folder + '/snapshot_no_{}_{}.png'.format(no, k), show_image)
        # cv2.imwrite(base_save_folder + '/result_no_{}_{}.png'.format(no, k), result_image)
                # cv2.imwrite(base_save_folder+'/original_no_{}_{}.png'.format(no,k),image)
                # cv2.WaitKey(0)
    show_image = np.zeros(shape=[512, 1024, 3])
    show_image = show_image.astype(np.uint8)
    show_image[:, :512, :] = image
    show_image[:, 512:, :] = teeth_masks_color + tar_masks_color + lables_color
    # print("no", k, np.max(masks_color), len(np.where(masks_color > 0)[0]))
    # show_image = Image.fromarray(np.uint8(show_image))
    show_image = show_image.astype(np.uint8)
    base_save_folder = '/home/projects/src/refineData/outputs_teeth/snapshot_tar_teeth_normal/'
    if not os.path.isdir(base_save_folder):
        os.makedirs(base_save_folder)
        os.chmod(base_save_folder, 0o777)

    cv2.imwrite(base_save_folder + '/sol_no_{}.png'.format(no), show_image)

    # cv2.imwrite(base_save_folder + '/result_no_{}_{}.png'.format(no, k), result_image)
    # cv2.imwrite(base_save_folder+'/original_no_{}_{}.png'.format(no,k),image)
    # cv2.WaitKey(0)



    print(result_dict)
    return confusion_matrix
    # return show_image

def create_image_label_path_generator(images_filepath, labels_filepath):
    image_paths = open(images_filepath).readlines()
    all_label_txts = os.listdir(labels_filepath)
    image_label_paths = []
    for label_txt in all_label_txts:
        label_name = label_txt[:-4]
        label_path = labels_filepath + "/" + label_txt
        for image_path in image_paths:
            image_path = image_path.rstrip()
            image_name = image_path.split("/")[-1][:-4]
            if label_name == image_name:
                image_label_paths.append((image_path, label_path))
    while True:
        random.shuffle(image_label_paths)
        for i in range(len(image_label_paths)):
            yield image_label_paths[i]

def process_image_label(image_path, label_path):
    # image = misc.imread(image_path)
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_NEAREST)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # data augmentation here
    # randomly shift gamma
    gamma = random.uniform(0.8, 1.2)
    image = image.copy() ** gamma
    image = np.clip(image, 0, 255)
    # randomly shift brightness
    brightness = random.uniform(0.5, 2.0)
    image = image.copy() * brightness
    image = np.clip(image, 0, 255)
    # image transformation here
    image = (image / 255. - rgb_mean) / rgb_std

    label = open(label_path).readlines()
    label = [np.array(line.rstrip().split(" ")) for line in label]
    label = np.array(label, dtype=np.int)
    label = cv2.resize(label, (224, 224), interpolation=cv2.INTER_NEAREST)
    label = label.astype(np.int)

    return image, label


def DataGenerator(train_image_txt, train_labels_dir, batch_size):
    """
    generate image and mask at the same time
    """
    image_label_path_generator = create_image_label_path_generator(
        train_image_txt, train_labels_dir
    )
    while True:
        images = np.zeros(shape=[batch_size, 224, 224, 3])
        labels = np.zeros(shape=[batch_size, 224, 224], dtype=np.float)
        for i in range(batch_size):
            image_path, label_path = next(image_label_path_generator)
            image, label = process_image_label(image_path, label_path)
            images[i], labels[i] = image, label
        yield images, labels
