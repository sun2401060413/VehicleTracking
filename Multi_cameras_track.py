'''
MCT: Multi cameras tracking.
Multi-objects tracking with multi-cameras.
written by sunzhu, 2019-03-20, version 1.0
'''

import os,sys
sys.path.append(r'D:\Project\tensorflow_model\VehicleTracking\AIC2018_iamai\ReID\ReID_CNN')
import pandas as pd
import cv2
import json
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

import compute_VeRi_dis as dist
from Model_Wrapper import ResNet_Loader

def load_crop_img_list(save_path):
    json_file_path = os.path.join(save_path,'crop_img_info.json')
    with open(json_file_path,'r') as doc:
        data = json.load(doc)
    return data


if __name__=="__main__":
    # root path
    dataset_root = r"E:\DataSet\trajectory\concatVD"

    # save root
    save_root = r'D:\Project\tensorflow_model\VehicleTracking\data_generator\gen_data'

    # images
    cam_1 = 'wuqi1B'
    cam_2 = 'wuqi2B'
    cam_3 = 'wuqi3B'
    cam_4 = 'wuqi4B'

    # bbox information
    box_info_1 = 'wuqiyuce1.csv'
    box_info_2 = 'wuqiyuce2.csv'
    box_info_3 = 'wuqiyuce3.csv'
    box_info_4 = 'wuqiyuce4.csv'

    img_savepath_1 = os.path.join(save_root,cam_1)
    img_savepath_2 = os.path.join(save_root,cam_2)
    
    # cmp_1,cmp_2 = get_CVPR_VehReId_data()
    # plt.figure("Image") #
    # for i in range(len(cmp_1)):
        # img_1 = Image.open(os.path.join(cmp_1[i]))
        # plt.subplot(5,2,2*i+1)
        # plt.imshow(img_1)
        # img_2 = Image.open(os.path.join(cmp_2[i]))
        # plt.subplot(5,2,2*i+2)
        # plt.imshow(img_2)
    # plt.show()
    
    load_ckpt = r"D:\Project\tensorflow_model\VehicleTracking\models\model_880_base.ckpt"
    n_layer = 50
    gallery_txt = r"E:\DataSet\BoxCars\Reid_dataset\BoxCars\BoxCars_train.txt"
    query_txt = r"E:\DataSet\BoxCars\Reid_dataset\BoxCars\BoxCars_train.txt"
    dis_mat = r"E:\DataSet\BoxCars\Reid_dataset\BoxCars"
    batch_size = 1
    print('loading model....')
    model = ResNet_Loader(load_ckpt,n_layer,output_color=False,batch_size=1)

    data_1 = load_crop_img_list(img_savepath_1)
    data_2 = load_crop_img_list(img_savepath_2)
    list_1 = [data_1['0'][2],data_1['1'][2],data_1['2'][2],data_1['3'][2],data_1['4'][2]]
    list_2 = [data_2['0'][2],data_2['1'][2],data_2['2'][2],data_2['3'][2],data_2['4'][2]]
    
    plt.figure("Image") #
    for i,elem in enumerate(list_1):
        name,_ = os.path.splitext(os.path.split(elem)[1])
        img = Image.open(os.path.join(elem))
        plt.subplot(2,5,i+1)
        plt.imshow(img)
        print(name)
        plt.title(name)
    for i,elem in enumerate(list_2):
        name,_ = os.path.splitext(os.path.split(elem)[1])
        img = Image.open(os.path.join(elem))
        plt.subplot(2,5,i+6)
        plt.imshow(img)
        print(name)
        plt.title(name)

    q_features = model.inference(list_1)
    g_features = model.inference(list_2)
            
    q_features = nn.functional.normalize(q_features,dim=1).cuda()
    g_features = nn.functional.normalize(g_features,dim=1).transpose(0,1).cuda()

    print('compute distance')
    SimMat = -1 * torch.mm(q_features,g_features)
    SimMat = SimMat.cpu().transpose(0,1)

    SimMat = SimMat.numpy()
    data1 = pd.DataFrame(SimMat)
    data1.to_csv(os.path.join(r'D:\Project\tensorflow_model\VehicleTracking\data_generator\gen_data','data1.csv'))
    
    plt.show()
    
    # with open(args.query_txt,'r') as f:
        # # query_txt = [q.strip() for q in f.readlines()]
        # # query_txt = query_txt[1:]
        # chosen_text = [[q.strip().split(' ')[0],q.strip().split(' ')[1]] for q in f.readlines() if q.strip()!='']
        
        # query_txt,id_txt = zip(*chosen_text)
        # query_txt = query_txt[1:21]
        # id_txt = id_txt[1:21]
    # with open(args.gallery_txt,'r') as f:
        # # gallery_txt = [q.strip() for q in f.readlines()]
        # # gallery_txt = gallery_txt[1:]
        # gallery_txt = [q.strip().split(' ')[0] for q in f.readlines() if q.strip()!='']
        # gallery_txt = query_txt[1:21]
    
    # print(query_txt)
    # print(id_txt)
    
    # plt.figure("Image") #
    # for i,elem in enumerate(query_txt):
        # img = Image.open(os.path.join(elem))
        # plt.subplot(4,5,i+1)
        # plt.imshow(img)
        # print(id_txt[i])
        # plt.title(id_txt[i])
    # plt.show()
        
    # print('inferencing q_features')
    # q_features = model.inference(query_txt)
    # print('inferencing g_features')
    # g_features = model.inference(query_txt)
    
    # q_features = nn.functional.normalize(q_features,dim=1).cuda()
    # g_features = nn.functional.normalize(g_features,dim=1).transpose(0,1).cuda()
    
    # print('compute distance')
    # SimMat = -1 * torch.mm(q_features,g_features)
    # SimMat = SimMat.cpu().transpose(0,1)

    # print(SimMat.size())
    
    # SimMat = SimMat.numpy()
    # # import scipy.io as sio
    # # sio.savemat(args.dis_mat,{'dist_CNN':SimMat})
    
    # data1 = pd.DataFrame(SimMat)
    # data1.to_csv(os.path.join(r'E:\DataSet\BoxCars\Reid_dataset','data1.csv'))
    # # doc = open(os.path.join(r'E:\DataSet\BoxCars\Reid_dataset','boxcar.txt'),'w')
    # # print(SimMat,file=doc)
    