import os
#import multiprocessing
#lock = multiprocessing.Lock()
import pickle
import time
import copy

from PIL import Image
import numpy as np
import pandas as pd
import cv2
os.environ['OPEN3D_CPU_RENDERING'] = 'true'
vis = None
#import open3d as o3d

canvassizeh = 1080
canvassizew = 1920

import torch as th
from torch.utils import data
from torchvision import transforms



class GazePointAllDataset(data.Dataset):
    def __init__(self, root_dir, faceimagesize=224, phase="train", foldno=-1, augnum=1, usesmallface=False, onlyeyeinput=False, targettype='gazedirection', **kwargs):
        self.root_dir = root_dir
        self.faceimagesize = faceimagesize
        self.phase = phase
        self.foldno = foldno
        self.augnum = augnum
        self.usesmallface = usesmallface
        self.onlyeyeinput = onlyeyeinput
        self.targettype = targettype
        self.kwargs = kwargs
        self.createdaugrenderwindow = False
        
        if foldno == -1:
            if os.path.exists(os.path.join(root_dir, phase+"_meta.csv")):
                metaname = phase+"_meta.csv"
                self.anno = pd.read_csv(os.path.join(root_dir, metaname), index_col=0)
            else:
                metaname = "_meta_0.csv"
                self.anno1 = pd.read_csv(os.path.join(root_dir, 'train'+metaname), index_col=0)
                self.anno2 = pd.read_csv(os.path.join(root_dir, 'val'+metaname), index_col=0)
                self.anno = pd.concat([self.anno1, self.anno2], axis=0)
        else:
            metaname = phase + "_meta_%d.csv" % self.foldno
            self.anno = pd.read_csv(os.path.join(root_dir, metaname), index_col=0)
            self.anno = self.anno.query("not face_image.str.contains('_FT_')")
        #print('img'+str(self.anno.__sizeof__()))
        # if os.path.isfile(os.path.join(root_dir, "depth_stat.pkl")):
        #     with open(os.path.join(root_dir, "depth_stat.pkl"), "rb") as fp:
        #         stat = pickle.load(fp)
        #         anno.drop(anno.iloc[stat[0]])
        root_dir = root_dir.rstrip("/").rstrip("\\")
        # should improve in the future using metadata file
        if root_dir.lower().find('shanghaitech') >= 0 :
            self.datasettype = 'shanghaitech'
        elif root_dir.lower().find('eyediap') >= 0 :
            self.datasettype = 'eyediap'
        elif root_dir.lower().find('mpii') >= 0 :
            self.datasettype = 'mpii'
        elif root_dir.lower().find('gaze360') >=0 :
            self.datasettype = 'gaze360'
        self.face_image_list = (root_dir + "/" + self.anno["face_image"]).tolist()
        #print('path'+str(self.face_image_list.__sizeof__()))
        #self.face_depth_list = (root_dir + "/" + self.anno["face_depth"]).tolist()
        self.face_bbox_list = (root_dir + "/" + self.anno["face_bbox"]).tolist()
        self.le_image_list = (root_dir + "/" + self.anno["left_eye_image"]).tolist()
        self.re_image_list = (root_dir + "/" + self.anno["right_eye_image"]).tolist()
        #self.le_depth_list = (root_dir + "/" + self.anno["left_eye_depth"]).tolist()
        #self.re_depth_list = (root_dir + "/" + self.anno["right_eye_depth"]).tolist()
        self.le_bbox_list = (root_dir + "/" + self.anno["left_eye_bbox"]).tolist()
        self.re_bbox_list = (root_dir + "/" + self.anno["right_eye_bbox"]).tolist()
        # self.le_coord_list = (root_dir + "/" + self.anno["left_eye_coord"]).tolist()
        # self.re_coord_list = (root_dir + "/" + self.anno["right_eye_coord"]).tolist()
        self.gt_name_list = (root_dir + "/" + self.anno["gaze_point"]).tolist()
        #self.haslandmarklist = self.anno['has_landmark'].tolist()

        for data_item in kwargs.keys():
            if data_item not in ("face_image", "face_depth", "eye_image", "eye_depth",
                                 "face_bbox", "eye_bbox", "gt", "eye_coord", "info"):
                raise ValueError(f"unrecognized dataset item: {data_item}")

    def __len__(self):
        return len(self.face_image_list)

    def __getitem__(self, idx):
        global vis
        if self.createdaugrenderwindow == False and self.augnum > 1:

            vis = o3d.visualization.Visualizer()
            vis.create_window("aa", canvassizew, canvassizeh, visible=False)
            self.createdaugrenderwindow = True
        with open(self.le_bbox_list[idx].replace('\\', '/')) as fp:
            le_bbox = list(map(float, fp.readline().split()))
        with open(self.re_bbox_list[idx].replace('\\', '/')) as fp:
            re_bbox = list(map(float, fp.readline().split()))
        with open(self.face_bbox_list[idx].replace('\\', '/')) as fp:
            face_bbox = list(map(float, fp.readline().split()))
        '''
        try:
            landmarks = []
            with open(self.face_bbox_list[idx].replace("face","landmarks")) as fp:
                fp.readline() # skip face bbox dataline
                while(True):
                    line = fp.readline()
                    if not line:
                        break
                    landmarks.append(list(map(float, line.split())))
                landmarks = np.array(landmarks)
                le_coor = (landmarks[42] + landmarks[43] + landmarks[44] + landmarks[45] + landmarks[46] + landmarks[47]) / 6.0
                re_coor = (landmarks[36] + landmarks[37] + landmarks[38] + landmarks[39] + landmarks[40] + landmarks[41]) / 6.0
        except:
            #le_coor = np.load(self.le_coord_list[idx])
            #re_coor = np.load(self.re_coord_list[idx])
        '''
        le_coor = 0.5*np.array([le_bbox[2] + le_bbox[0], le_bbox[3] + le_bbox[1]], np.int32)
        re_coor = 0.5*np.array([re_bbox[2] + re_bbox[0], re_bbox[3] + re_bbox[1]], np.int32)
        gt = np.load(self.gt_name_list[idx].replace('\\', '/'))
        headposefilename = self.gt_name_list[idx].replace('\\', '/').replace("gaze.npy","head.npy")
        if os.path.exists(headposefilename):
            headposematrix = np.load(headposefilename)
        else:
            headposematrix = np.eye(3)
        if self.datasettype == 'mpii': # to fix the wrong name in mpii preprocessing code
            facename = 'oriEach'
        else:
            facename = 'face'

        if os.path.exists(self.gt_name_list[idx].replace('\\', '/').replace("gaze.npy",facename+".npy")):
            face3dmmarr = np.load(self.gt_name_list[idx].replace('\\', '/').replace("gaze.npy",facename+".npy"), allow_pickle=True)
        else:
            face3dmmarr = np.load(self.gt_name_list[idx].replace('\\', '/').replace("gaze.npy",'ori'+".npy"), allow_pickle=True)

        with open(self.gt_name_list[idx].replace('\\', '/').replace("gaze.npy", 'ori'+".pkl"),'rb') as f:
            face3dmmmeshattrib = pickle.load(f)['mesh'] # 9976*25
        face3dmmmeshattrib = np.transpose(face3dmmmeshattrib,(1,0)) # 25*9976
        facenormalimg = (face3dmmarr[()]['normalimages'][0]+1)*0.5 # range is 0 to 1
        facenormalimg = np.transpose(facenormalimg, (1,2,0))
        facedepthimg = face3dmmarr[()]['depthimages'][0]
        facedepthimg[np.where(facedepthimg==-1)]=0
        #+1 # range is 0 to x
        #headposematrix = np.matmul(headposematrix,np.array([[1,0,0],[0,-1,0],[0,0,-1]])) #fix for eyediap preprocessing script view_session.py for vga videos
        headposevector = cv2.Rodrigues(headposematrix)[0].T[0]

        # should improve in the future using metadata file
        if self.datasettype == 'shanghaitech':
            TARGETSCREEN_WIDTH = 59.77
            TARGETSCREEN_HEIGHT = 33.62
            TARGETSCREEN_COORD_TOREALWORLDSCALE = 0.01 #cm to m
            TARGETSCREEN_COORDX_TOREALWORLDCOEF = -1
            TARGETSCREEN_COORDX_TOREALWORLDBIAS = -TARGETSCREEN_WIDTH*0.5
            TARGETSCREEN_COORDY_TOREALWORLDCOEF = 1
            TARGETSCREEN_COORDY_TOREALWORLDBIAS = TARGETSCREEN_HEIGHT*0.5 - TARGETSCREEN_HEIGHT
            TARGETSCREEN_COORDZ_TOREALWORLDBIAS = 0
            DEPTHSENSOR_CENTROIDVAL = 500
            DEPTHSENSOR_TOREALWORLDSCALE = 0.000125
            COLORSENSOR_WIDTH = 1920
            COLORSENSOR_HEIGHT = 1080
            COLORSENSOR_R = np.transpose(np.array([[0.999746,0.000291,-0.022534], [0.000156,0.999803,0.019831], [0.022535,-0.019830,0.999549]]))
            COLORSENSOR_T = 0.01 * np.transpose(np.array([23.590678,-0.954391,-4.049306]))
            COLORSENSOR_K = np.transpose(np.array([[1384.2,0,0], [0,1381.3,0], [972.8,539.0,1.0]]))
            gt[0] -= TARGETSCREEN_WIDTH / 2
            gt[1] -= TARGETSCREEN_HEIGHT / 2            
        elif (self.datasettype == 'eyediap') or (self.datasettype == 'mpii') or (self.datasettype == 'gaze360'):
            #print("not implemented.")
            pass

        data_transforms_ShanghaiTechGaze_face = {
            'train': transforms.Compose([
                transforms.Resize([96, 96], antialias=None),
                transforms.Grayscale(3)
                #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize([96, 96], antialias=None),
                transforms.Grayscale(3)
                #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }
        
        #data_transforms_ShanghaiTechGaze_facedepth = transforms.Resize([self.faceimagesize, self.faceimagesize], antialias=None)

        data_transforms_ShanghaiTechGaze_eye = transforms.Compose([
                transforms.Resize([64,96], antialias=None),
                transforms.Grayscale(3)
                #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        
        data_transforms_ShanghaiTechGaze_eyenormalordepth = transforms.Resize([64, 96], antialias=None)

        data_transforms_ShanghaiTechGaze_facenormalordepth = transforms.Resize([96, 96], antialias=None)

        sample = {}
        sample['idx'] = th.LongTensor([idx])
        sample['headposematrix'] = th.FloatTensor(headposematrix)
        sample['headposevector'] = th.FloatTensor(headposevector)
        #sample['facenormalimg'] = th.FloatTensor(facenormalimg)
        sample['facedepthimg'] = th.FloatTensor(facedepthimg)
        sample['facemeshattrib'] = th.FloatTensor(face3dmmmeshattrib)
        #sample["index"] = th.LongTensor([idx])
        #index = f"{self.anno.index[idx]:010d}"
        #sample["pid"] = th.LongTensor([int(index[:5])])
        #sample["sid"] = th.LongTensor([int(index[5:])])
        
        theta, phi = gt
        vector3d = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
        #vector3d = np.array([vector3d[0],-vector3d[2],vector3d[1]])
        if self.datasettype == 'gaze360': # to fix the dataset preprocessing bug of gaze360
            vector3d[0]=-vector3d[0]
            vector3d[1]=-vector3d[1]
        yaw = np.arctan2(-vector3d[0], -vector3d[2])
        pitch = np.arcsin(-vector3d[1]/np.linalg.norm(vector3d))
        gt = np.array([yaw,pitch])
        
        #sample['gt'] = gt

        if self.kwargs.get('face_image'):
            if self.datasettype == 'mpii':
                self.face_image_list[idx] = self.face_image_list[idx].replace("_face.","_ori.") # to fix the wrong name in mpii preprocessing code
            face_image_ori = Image.open(self.face_image_list[idx].replace('\\', '/'))
            sample['face_image_ori'] = np.array(face_image_ori)
            scale_factor_ori = sample['face_image_ori'].shape[0] / (face_bbox[2] - face_bbox[0])
            if self.usesmallface:
                smallfacebbox_infaceimage = [scale_factor_ori*(re_bbox[0]-face_bbox[0]), scale_factor_ori*(re_bbox[1]-face_bbox[1]), scale_factor_ori*(le_bbox[2]-face_bbox[0]), scale_factor_ori*(le_bbox[3]-face_bbox[1])]
                for i, elem in enumerate(smallfacebbox_infaceimage):
                    if elem < 0:
                        smallfacebbox_infaceimage[i] = 0
                    if elem > sample['face_image_ori'].shape[0]:
                        smallfacebbox_infaceimage[i] = sample['face_image_ori'].shape[0]
                smallfacebboxcenter = [(smallfacebbox_infaceimage[0]+smallfacebbox_infaceimage[2])*0.5, (smallfacebbox_infaceimage[1]+smallfacebbox_infaceimage[3])*0.5]
                smallfacebboxheight = smallfacebbox_infaceimage[3] - smallfacebbox_infaceimage[1]
                smallfacebboxlength = smallfacebbox_infaceimage[2] - smallfacebbox_infaceimage[0]
                if smallfacebboxlength > smallfacebboxheight:
                    Ymin = smallfacebboxcenter[1] - 0.5*smallfacebboxlength
                    Ymax = smallfacebboxcenter[1] + 0.5*smallfacebboxlength
                    if Ymin < 0:
                        Ymax+=-Ymin
                        Ymin=0
                    if Ymax > sample['face_image_ori'].shape[0]:
                        Ymin-=Ymax-sample['face_image_ori'].shape[0]
                        Ymax=sample['face_image_ori'].shape[0]
                    smallfacebbox_infaceimage[1] = Ymin
                    smallfacebbox_infaceimage[3] = Ymax
                if smallfacebboxheight >= smallfacebboxlength:
                    Xmin = smallfacebboxcenter[0] - 0.5*smallfacebboxheight
                    Xmax = smallfacebboxcenter[0] + 0.5*smallfacebboxheight
                    if Xmin < 0:
                        Xmax+=-Xmin
                        Xmin=0
                    if Xmax > sample['face_image_ori'].shape[0]:
                        Xmin-=Xmax-sample['face_image_ori'].shape[0]
                        Xmax=sample['face_image_ori'].shape[0]
                    smallfacebbox_infaceimage[0] = Xmin
                    smallfacebbox_infaceimage[2] = Xmax
                face_image_cropped = sample['face_image_ori'][int(smallfacebbox_infaceimage[1]):int(smallfacebbox_infaceimage[3]), int(smallfacebbox_infaceimage[0]):int(smallfacebbox_infaceimage[2])]
                sample['face_image'] = []
                sample['face_image'].append(transforms.ToTensor()(transforms.functional.equalize(data_transforms_ShanghaiTechGaze_face[self.phase](Image.fromarray(face_image_cropped.copy())))))
                smallfacebbox = [face_bbox[0]+(smallfacebbox_infaceimage[0]/scale_factor_ori), face_bbox[1]+(smallfacebbox_infaceimage[1]/scale_factor_ori), face_bbox[0]+(smallfacebbox_infaceimage[2]/scale_factor_ori), face_bbox[1]+(smallfacebbox_infaceimage[3]/scale_factor_ori)]
                face_bbox = smallfacebbox
                scale_factor = sample['face_image'][0].shape[1] / (face_bbox[2] - face_bbox[0])
                sample['face_scale_factor'] = th.FloatTensor([scale_factor])
                K_CROPPED = COLORSENSOR_K.copy()
                K_CROPPED[0][2] = (K_CROPPED[0][2] - face_bbox[0]) * scale_factor #cx
                K_CROPPED[1][2] = (K_CROPPED[1][2] - face_bbox[1]) * scale_factor #cy
                K_CROPPED[0][0] = K_CROPPED[0][0] * scale_factor #fx
                K_CROPPED[1][1] = K_CROPPED[1][1] * scale_factor #fy
                COLORSENSOR_K = K_CROPPED
            else:
                sample['face_image'] = []
                sample['face_image'].append((transforms.ToTensor()(transforms.functional.equalize(data_transforms_ShanghaiTechGaze_face[self.phase](face_image_ori.copy()))))[0:1])
                scale_factor = scale_factor_ori
                sample['face_scale_factor'] = th.FloatTensor([scale_factor])

        '''
        if self.kwargs.get('face_depth'):
            assert np.abs((face_bbox[3] - face_bbox[1]) - (face_bbox[2] - face_bbox[0])) <= 2, f"invalid face bbox @ {self.face_bbox_list[idx]}"
            # scale_factor = min(scale_factor, 1.004484)
            # scale_factor = max(scale_factor, 0.581818)
            face_depth = cv2.imread(self.face_depth_list[idx], -1)
            # face_depth = np.int32(face_depth)
            # face_depth[face_depth<500] = 500
            # face_depth[face_depth > 1023] = 1023
            # face_depth -= 512
            #if self.transform is not None:
            # sample['face_depth'] = th.FloatTensor(face_depth / 883)
            if self.usesmallface:
                face_depth = face_depth[int(smallfacebbox_infaceimage[1]):int(smallfacebbox_infaceimage[3]), int(smallfacebbox_infaceimage[0]):int(smallfacebbox_infaceimage[2])]
            face_depth = face_depth[np.newaxis, :, :]# / scale_factor            
            face_depth = data_transforms_ShanghaiTechGaze_facedepth(th.FloatTensor(face_depth.astype('float')))
            sample['face_depth'] = []
            #a = time.time()
            sample['face_depth'].append(th.clamp((face_depth - DEPTHSENSOR_CENTROIDVAL) / DEPTHSENSOR_CENTROIDVAL, 0., 1.))
            #b = time.time()
            #print('append time: %f' % (b-a))
            #else:
            #    sample['face_depth'] = face_depth
            # print('max: {}, min:{}'.format((face_depth / 430).max(), (face_depth / 430).min()), flush=True)
        '''

        if self.kwargs.get('eye_image'):
            le_image = Image.open(self.le_image_list[idx].replace('\\', '/'))
            re_image = Image.open(self.re_image_list[idx].replace('\\', '/'))
            le_image = np.array(le_image)
            re_image = np.array(re_image)
            le_depth = facedepthimg[int(le_bbox[1]):int(le_bbox[3]),int(le_bbox[0]):int(le_bbox[2])]
            le_depth = cv2.resize(le_depth, (224,224), interpolation=cv2.INTER_LINEAR)
            #print(le_depth.shape)
            #print(facenormalimg.shape)
            le_normal = facenormalimg[int(le_bbox[1]):int(le_bbox[3]),int(le_bbox[0]):int(le_bbox[2])]
            #print(le_normal.shape)
            le_normal = cv2.resize(le_normal, (224,224), interpolation=cv2.INTER_LINEAR)
            re_depth = facedepthimg[int(re_bbox[1]):int(re_bbox[3]),int(re_bbox[0]):int(re_bbox[2])]
            re_depth = cv2.resize(re_depth, (224,224), interpolation=cv2.INTER_LINEAR)
            re_normal = facenormalimg[int(re_bbox[1]):int(re_bbox[3]),int(re_bbox[0]):int(re_bbox[2])]
            re_normal = cv2.resize(re_normal, (224,224), interpolation=cv2.INTER_LINEAR)

            l_stride = 222
            w_stride = int(l_stride*64/96)
            l_center = int(le_image.shape[1]/2)
            w_center = int(le_image.shape[0]/2)
            l_corner = int(l_center-l_stride/2)
            w_corner = int(w_center-w_stride/2)

            le_image = np.array(le_image[w_corner:w_corner+w_stride, l_corner:l_corner+l_stride])
            re_image = np.array(re_image[w_corner:w_corner+w_stride, l_corner:l_corner+l_stride])
            le_depth = np.array(le_depth[w_corner:w_corner+w_stride, l_corner:l_corner+l_stride])
            #print(le_depth.shape)
            le_normal = np.array(le_normal[w_corner:w_corner+w_stride, l_corner:l_corner+l_stride])
            re_depth = np.array(re_depth[w_corner:w_corner+w_stride, l_corner:l_corner+l_stride])
            re_normal = np.array(re_normal[w_corner:w_corner+w_stride, l_corner:l_corner+l_stride])

            sample['left_eye_image'] = data_transforms_ShanghaiTechGaze_eye(Image.fromarray(le_image))
            sample['left_eye_image'] = (transforms.ToTensor()(transforms.functional.equalize(sample['left_eye_image'])))[0:1]
            sample['right_eye_image'] = data_transforms_ShanghaiTechGaze_eye(Image.fromarray(re_image))
            sample['right_eye_image'] = (transforms.ToTensor()(transforms.functional.equalize(sample['right_eye_image'])))[0:1]
            sample['left_eye_depth'] = data_transforms_ShanghaiTechGaze_eyenormalordepth(th.FloatTensor(np.expand_dims(le_depth, 0)))
            sample['left_eye_normal'] = data_transforms_ShanghaiTechGaze_eyenormalordepth(th.FloatTensor(le_normal.transpose(2,0,1)))
            sample['right_eye_depth'] = data_transforms_ShanghaiTechGaze_eyenormalordepth(th.FloatTensor(np.expand_dims(re_depth,0)))
            sample['right_eye_normal'] = data_transforms_ShanghaiTechGaze_eyenormalordepth(th.FloatTensor(re_normal.transpose(2,0,1)))
            sample['facenormalimg'] = data_transforms_ShanghaiTechGaze_facenormalordepth(th.FloatTensor(facenormalimg.transpose(2,1,0)))
            #cv2.imwrite("")

        '''
        if self.kwargs.get('eye_depth'):
            le_depth = cv2.imread(self.le_depth_list[idx], -1)
            re_depth = cv2.imread(self.re_depth_list[idx], -1)
            #if self.transform is not None:
            le_depth = le_depth[np.newaxis, :, :].astype('float') # / le_scale_factor  # the new dim is the dim with np.newaxis
            re_depth = re_depth[np.newaxis, :, :].astype('float') # / re_scale_factor
            # sample['left_depth'] = torch.FloatTensor(le_depth/1000)
            # sample['right_depth'] = torch.FloatTensor(re_depth/1000)
            sample['left_eye_depth'] = th.FloatTensor(le_depth)
            sample['right_eye_depth'] = th.FloatTensor(re_depth)
            #else:
            #    sample['left_eye_depth'] = le_depth
            #    sample['right_eye_depth'] = re_depth
        '''

        if self.kwargs.get('face_bbox'):
            sample['face_bbox'] = th.FloatTensor(face_bbox)

        if self.kwargs.get('eye_bbox'):
            #assert le_bbox[3] - le_bbox[1] == le_bbox[2] - le_bbox[0], f"invalid left eye bbox @ {self.le_bbox_list[idx]}"
            le_scale_factor = sample['left_eye_image'].shape[0] / (le_bbox[2] - le_bbox[0])
            # le_scale_factor = min(le_scale_factor, 1.004484)
            # le_scale_factor = max(le_scale_factor, 0.581818)
            #assert re_bbox[3] - re_bbox[1] == re_bbox[2] - re_bbox[0], f"invalid right eye bbox @ {self.re_bbox_list[idx]}"
            re_scale_factor = sample['right_eye_image'].shape[0] / (re_bbox[2] - re_bbox[0])
            # re_scale_factor = min(re_scale_factor, 1.004484)
            # re_scale_factor = max(re_scale_factor, 0.581818)
            sample["left_eye_scale_factor"] = th.FloatTensor([le_scale_factor])
            sample["right_eye_scale_factor"] = th.FloatTensor([re_scale_factor])
            sample['left_eye_bbox'] = th.FloatTensor(le_bbox)
            sample['right_eye_bbox'] = th.FloatTensor(re_bbox)
            sample['left_eye_bbox'][:2] -= sample['face_bbox'][:2]
            sample['left_eye_bbox'][2:] -= sample['face_bbox'][:2]
            sample['right_eye_bbox'][:2] -= sample['face_bbox'][:2]
            sample['right_eye_bbox'][2:] -= sample['face_bbox'][:2]
            sample['left_eye_bbox'] = th.clamp(scale_factor * sample['left_eye_bbox'], min=0, max=self.faceimagesize-1)
            sample['right_eye_bbox'] = th.clamp(scale_factor * sample['right_eye_bbox'], min=0, max=self.faceimagesize-1)

        '''
        if self.kwargs.get('eye_coord'):
            sample['left_eye_coord'] = th.FloatTensor(np.float32(le_coor))
            sample['right_eye_coord'] = th.FloatTensor(np.float32(re_coor))
        '''

        if self.kwargs.get('info'): # this is only used when estimating screen targets
            if self.datasettype == 'shanghaitech':
                le_depth = np.clip((cv2.imread(self.le_depth_list[idx], -1) - DEPTHSENSOR_CENTROIDVAL) / DEPTHSENSOR_CENTROIDVAL, 0, 1.)
                re_depth = np.clip((cv2.imread(self.re_depth_list[idx], -1) - DEPTHSENSOR_CENTROIDVAL) / DEPTHSENSOR_CENTROIDVAL, 0, 1.)
                # get info
                le_depth_ = le_depth[le_depth > 0]
                if len(le_depth_) > 0:
                    le_info = [le_coor[0] / COLORSENSOR_WIDTH, le_coor[1] / COLORSENSOR_HEIGHT, np.mean(le_depth_)]
                else:
                    le_info = [le_coor[0] / COLORSENSOR_WIDTH, le_coor[1] / COLORSENSOR_HEIGHT] + [0.]

                re_depth_ = re_depth[re_depth > 0]
                if len(re_depth_) > 0:
                    re_info = [re_coor[0] / COLORSENSOR_WIDTH, re_coor[1] / COLORSENSOR_HEIGHT, np.mean(re_depth_)]
                else:
                    re_info = [re_coor[0] / COLORSENSOR_WIDTH, re_coor[1] / COLORSENSOR_HEIGHT] + [0.]
            else:
                le_info = [le_coor[0], le_coor[1], 0]
                re_info = [re_coor[0], re_coor[1], 0]
            sample['left_eye_info'] = th.FloatTensor(le_info)
            sample['right_eye_info'] = th.FloatTensor(re_info)

        if self.targettype == 'gazedirection' and self.datasettype == 'shanghaitech':
            predictedpoint_screencoord = gt
            lefteye_pixelcoord = sample['left_eye_info']
            righteye_pixelcoord = sample['right_eye_info']
            #print(facebbox)
            #print(lefteye_pixelcoord)
            #print(righteye_pixelcoord)
            # Below are extrinsics, extracted from 'ex.txt' of the 'camera' folder of the repo:
            # Although the author do not specify, referring to the common best practice, 
            # we first guess and assume that the depth image is aligned to color image, and the extrinsics is the color camera's.
            # After observation on the Figure1 of the paper and the T value, we guess and assume that the original T value is in centimeters,
            # and the world coordinate system is originated in the left lower corner of the screen,
            # with the same axis direction as the camera coordinate system.
            # Note that the guessed world coordinate system is not the O'X'Y'Z' coordinate system on the Figure1.
            # After test, we found that only in above case can we extract correct coordinates.
            # R = RCOLOR = np.transpose(np.array([[0.999746,0.000291,-0.022534], [0.000156,0.999803,0.019831], [0.022535,-0.019830,0.999549]]))
            # T = TCOLOR = 0.01 * np.transpose(np.array([23.590678,-0.954391,-4.049306]))
            # Below are intrinsics for color camera, extracted from 'color.mat'(matlab matfile) of the 'camera folder' of the repo:
            # K = KCOLOR = np.transpose(np.array([[1384.2,0,0], [0,1381.3,0], [972.8,539.0,1.0]]))
            # Below are intrinsics for depth camera, extracted from 'depth.mat'(matlab matfile) of the 'camera folder' of the repo:
            # KDEPTH = np.transpose(np.array([[486.2349,0,0], [0,483.7896,0], [322.1105,244.0338,1.0]]))

            # Then we restore the preprocessed value according to gaze_dataset.py,
            # and retrieve the actual depth value in meter, according to the tech parameters of the Intel RealSense SR300 device
            # which is declared in the paper to be used as the RGBD camera of the repo.
            #a = time.time()
            lefteye_pixelcoord = np.array([COLORSENSOR_WIDTH*lefteye_pixelcoord[0], COLORSENSOR_HEIGHT*lefteye_pixelcoord[1], DEPTHSENSOR_TOREALWORLDSCALE*(lefteye_pixelcoord[2]*DEPTHSENSOR_CENTROIDVAL+DEPTHSENSOR_CENTROIDVAL)])
            righteye_pixelcoord = np.array([COLORSENSOR_WIDTH*righteye_pixelcoord[0], COLORSENSOR_HEIGHT*righteye_pixelcoord[1], DEPTHSENSOR_TOREALWORLDSCALE*(righteye_pixelcoord[2]*DEPTHSENSOR_CENTROIDVAL+DEPTHSENSOR_CENTROIDVAL)])
            # pixel coordinate -> camera coordinate
            lefteye_cameracoord = lefteye_pixelcoord[2] * np.matmul(np.linalg.inv(COLORSENSOR_K), np.array([lefteye_pixelcoord[0], lefteye_pixelcoord[1], 1]))
            righteye_cameracoord = righteye_pixelcoord[2] * np.matmul(np.linalg.inv(COLORSENSOR_K), np.array([righteye_pixelcoord[0], righteye_pixelcoord[1], 1]))
            central_cameracoord = 0.5 * (lefteye_cameracoord + righteye_cameracoord)
            #b = time.time()
            #print('inv time: %f' % (b-a))
            #print(lefteye_cameracoord)
            #print(righteye_cameracoord)

            # We first assume the screencoordinate is in centimeters,
            # and then transform the screen coordinate to world coordinate using this and above assumption. 
            predictedpoint_worldcoord = TARGETSCREEN_COORD_TOREALWORLDSCALE * np.array([TARGETSCREEN_COORDX_TOREALWORLDCOEF*predictedpoint_screencoord[0]+TARGETSCREEN_COORDX_TOREALWORLDBIAS, TARGETSCREEN_COORDY_TOREALWORLDCOEF*predictedpoint_screencoord[1]+TARGETSCREEN_COORDY_TOREALWORLDBIAS, TARGETSCREEN_COORDZ_TOREALWORLDBIAS])
            # world coordinate -> camera coordinate
            predictedpoint_cameracoord = np.matmul(COLORSENSOR_R, predictedpoint_worldcoord) + COLORSENSOR_T
            lgazevector = predictedpoint_cameracoord - lefteye_cameracoord
            rgazevector = predictedpoint_cameracoord - righteye_cameracoord
            lphi = np.math.atan2(lgazevector[1], lgazevector[0])
            ltheta = np.math.acos(lgazevector[2]/np.linalg.norm(lgazevector))
            rphi = np.math.atan2(rgazevector[1], rgazevector[0])
            rtheta = np.math.acos(rgazevector[2]/np.linalg.norm(rgazevector))
            sample['gt'] = []
            sample['gt'].append(th.FloatTensor(np.array([ltheta, lphi, rtheta, rphi])))
            #print(predictedpoint_cameracoord)
            sample['left_eye_info'] = []
            sample['right_eye_info'] = []
            sample['left_eye_info'].append(th.FloatTensor(lefteye_cameracoord))
            sample['right_eye_info'].append(th.FloatTensor(righteye_cameracoord))
        elif self.targettype == 'gazepoint' and (self.datasettype == 'eyediap' or self.datasettype == 'mpii' or self.datasettype == 'gaze360'):
            print('not implemented.')
        else:
            sample['gt'] = []
            sample['gt'].append(th.FloatTensor(gt))
            sample['left_eye_info'] = []
            sample['right_eye_info'] = []
            sample['left_eye_info'].append(th.FloatTensor(le_info))
            sample['right_eye_info'].append(th.FloatTensor(re_info))

        if self.augnum > 1 and self.targettype == 'gazedirection':
            #print(scale_factor)
            #print(KCOLOR_CROPPED)
            facedepth = np.array(sample['face_depth'][0])*DEPTHSENSOR_CENTROIDVAL+DEPTHSENSOR_CENTROIDVAL
            #print(facedepth.max())
            #print(facedepth.min())
            gt = np.array(sample['gt'][0])
            #print(face_image_cropped.shape)
            #print(facedepth.shape)
            face_image_cropped = cv2.resize(face_image_cropped.copy(), [self.faceimagesize, self.faceimagesize])
            #print(face_image_cropped.shape)
            #cv2.imwrite('0.jpg', face_image_cropped)
            faceimagenpy = face_image_cropped.astype(order="C", dtype=np.uint8)
            faceimageo3d = o3d.geometry.Image(faceimagenpy)
            #print(faceimagenpy.shape)
            facedepthnpy = facedepth.transpose(1,2,0).astype(order="C", dtype=np.float32)
            facedeptho3d = o3d.geometry.Image(facedepthnpy)
            #depthtest=np.asarray(facedeptho3d)
            #print(depthtest.max())
            #print(depthtest.min())
            #print(facedepthnpy.shape)
            facergbd = o3d.geometry.RGBDImage.create_from_color_and_depth(faceimageo3d, facedeptho3d, depth_scale=1.0, depth_trunc=1.0, convert_rgb_to_intensity=False)
            #print(facergbd.get_min_bound())
            #print(facergbd.get_max_bound())
            COLORSENSOR_K_O3D = o3d.camera.PinholeCameraIntrinsic(face_image_cropped.shape[0], face_image_cropped.shape[1], COLORSENSOR_K)
            facepcd = o3d.geometry.PointCloud().create_from_rgbd_image(facergbd, COLORSENSOR_K_O3D)
            #facepcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
            for i in range(0, self.augnum-1):
                facepcdcpy = facepcd.clone()
                rotateaxisangle = (np.random.rand(3)-0.5)*3*[np.pi/180,np.pi/180,np.pi/180]
                rotatematrix = facepcdcpy.get_rotation_matrix_from_axis_angle(rotateaxisangle)
                facepcdcpy.rotate(rotatematrix)
                o3d.io.write_point_cloud("cc.ply", facepcdcpy)
                #lock.acquire()
                
                canvassizecandidateh = max(2 * int(COLORSENSOR_K[1][2]), -2 * int(COLORSENSOR_K[1][2]) + 2 * face_image_cropped.shape[1]) + 10
                canvassizecandidatew = max(2 * int(COLORSENSOR_K[0][2]), -2 * int(COLORSENSOR_K[0][2]) + 2 * face_image_cropped.shape[0]) + 10
                #canvassize = 10 + max(canvassizecandidateh, canvassizecandidatew)

                if canvassizecandidateh > canvassizeh or canvassizecandidatew > canvassizew:
                    actualcanvassizeh = canvassizecandidateh
                    actualcanvassizew = canvassizecandidatew

                    vis = o3d.visualization.Visualizer()
                    vis.create_window("bb", canvassizecandidatew, canvassizecandidateh, visible=False)
                    self.createdaugrenderwindow = False
                else:
                    actualcanvassizeh = canvassizeh
                    actualcanvassizew = canvassizew
                
                cam = o3d.camera.PinholeCameraParameters()
                cam.extrinsic = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
                cam.intrinsic = COLORSENSOR_K_O3D
                vis.clear_geometries()
                vis.add_geometry(facepcdcpy)
                vis.get_view_control().convert_from_pinhole_camera_parameters(cam, True)
                #print(cam.intrinsic.intrinsic_matrix)
                vis.poll_events()
                vis.update_renderer()
                #lock.release()
                cap = np.asarray(vis.capture_screen_float_buffer(do_render=True))*255
                sp = cap.shape
                depthcap = np.asarray(vis.capture_depth_float_buffer(do_render=True))
                '''
                render = o3d.visualization.rendering.OffscreenRenderer(actualcanvassizew, actualcanvassizeh)
                fakematerial = o3d.visualization.rendering.MaterialRecord()
                render.scene.add_geometry("aa", facepcdcpy, fakematerial)
                render.setup_camera(COLORSENSOR_K_O3D, np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]))
                cap = np.asarray(render.render_to_image())
                depthcap = np.asarray(render.render_to_depth_image(z_in_view_space=True))
                '''
                
                cap = cap[round((actualcanvassizeh/2)-round(COLORSENSOR_K[1][2], 5)):round((actualcanvassizeh/2)-round(COLORSENSOR_K[1][2], 5)+face_image_cropped.shape[1]),round((actualcanvassizew/2)-round(COLORSENSOR_K[0][2], 5)):round((actualcanvassizew/2)-round(COLORSENSOR_K[0][2], 5)+face_image_cropped.shape[0])]
                print("colormin:",cap.min())
                print("colormax:",cap.max())
                faceimagerotate = np.asarray(cap, dtype=np.uint8)
                cv2.imwrite('b.jpg', faceimagerotate)
                
                depthcap = depthcap[round((actualcanvassizeh/2)-round(COLORSENSOR_K[1][2], 5)):round((actualcanvassizeh/2)-round(COLORSENSOR_K[1][2], 5)+face_image_cropped.shape[1]),round((actualcanvassizew/2)-round(COLORSENSOR_K[0][2], 5)):round((actualcanvassizew/2)-round(COLORSENSOR_K[0][2], 5)+face_image_cropped.shape[0])]
                print("depthmin:",depthcap.min())
                print("depthmax:",depthcap.max())
                facedepthrotate = np.clip((depthcap-DEPTHSENSOR_CENTROIDVAL)/DEPTHSENSOR_CENTROIDVAL, 0, 1.)
                if (faceimagerotate.shape[0] != face_image_cropped.shape[0]) or (faceimagerotate.shape[1] != face_image_cropped.shape[1]):
                    print("warning, malformed rotated image:")
                    print(sp)
                    print([COLORSENSOR_K[1][2], round(COLORSENSOR_K[1][2], 5)])
                    print([round((actualcanvassizeh/2)-round(COLORSENSOR_K[1][2], 5)),round((actualcanvassizeh/2)-round(COLORSENSOR_K[1][2], 5)+face_image_cropped.shape[1]),round((actualcanvassizew/2)-round(COLORSENSOR_K[0][2], 5)),round((actualcanvassizew/2)-round(COLORSENSOR_K[0][2], 5)+face_image_cropped.shape[0])])
                    print(COLORSENSOR_K)
                    print(cam.intrinsic.intrinsic_matrix)
                    print([canvassizeh, canvassizew, canvassizecandidateh, canvassizecandidatew, actualcanvassizeh, actualcanvassizew])
                    print(face_image_cropped.shape)
                    print(faceimagerotate.shape)

                sample['face_image'].append(data_transforms_ShanghaiTechGaze_face[self.phase](Image.fromarray(faceimagerotate)))
                sample['face_depth'].append(th.FloatTensor(facedepthrotate).unsqueeze(0))
                gazevectornew = np.matmul(rotatematrix, gazevector)
                phi = np.math.atan2(gazevectornew[1], gazevectornew[0])
                theta = np.math.acos(gazevectornew[2]/np.linalg.norm(gazevectornew))
                sample['gt'].append(th.FloatTensor(np.array([theta, phi])))
        sample['face_image'] = th.stack(sample['face_image'])
        #sample['face_depth'] = th.stack(sample['face_depth'])
        sample['gt'] = th.stack(sample['gt'])
        sample['left_eye_info'] = th.stack(sample['left_eye_info'])
        sample['right_eye_info'] = th.stack(sample['right_eye_info'])
        #c = time.time()
        #print('stack time: %f' % (c-b))

        return sample


if __name__ == '__main__':
    from tqdm import tqdm
    dataset = GazePointAllDataset(
        root_dir=r"D:\\data\\gaze",
        phase="train",
        face_image=True,
        face_depth=True,
        face_bbox=True,
        eye_image=True,
        eye_depth=True,
        eye_bbox=True,
        eye_coord=True
    )

    for sample in tqdm(dataset, desc="testing"):
        print(sample)
