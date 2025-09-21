import copy
import cv2
import models.wtdcnn
import models.mixtr
from utils.trainer import Trainer
import torch as th
from utils.dataloaderwrapper import getDataLoader
from torch import optim
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms 
from torchvision.utils import make_grid, save_image
from data.gaze_dataset import GazePointAllDataset
import numpy as np
import fire
import logging
import os
import time
import json
import models

th.backends.cudnn.deterministic = False
th.backends.cudnn.benchmark = True

def ignorenumpy(npy):
    return 'nonserialize'

class GazeTrainer(Trainer):
    def __init__(self,
                 # data parameters
                 data_root: str = r'D:\data\gaze',
                 batch_size_train: int = 8,
                 batch_size_val: int = 8,
                 num_workers: int = 20,
                 # trainer parameters
                 is_cuda=True,
                 exp_name="gaze_aaai",
                 fold_no=-1,
                 visualize=True,
                 multitask=False,
                 onlyeyeinput=False,
                 augnum=1,
                 facesize=112,
                 usesmallface=False,
                 targettype='gazedirection',
                 oneortwotarget=1,
                 val_data_root = None
                 ):
        super(GazeTrainer, self).__init__(checkpoint_dir='./results/' + exp_name + "_" + str(fold_no), is_cuda=is_cuda)
        self.data_root = data_root
        if val_data_root is not None:
            self.val_data_root = val_data_root
        else:
            self.val_data_root = self.data_root
        self.exp_name = exp_name
        self.fold_no = fold_no
        self.result_root = './results/' + exp_name + "_" + str(fold_no)
        self.batch_size_train = batch_size_train
        self.batch_size_val = batch_size_val
        self.num_workers = num_workers
        self.multitask = multitask
        self.onlyeyeinput = onlyeyeinput
        self.augnum = augnum
        self.facesize = facesize
        self.usesmallface = usesmallface
        self.targettype = targettype
        self.oneortwotarget = oneortwotarget
        self.actualtrain = True
        self.iseyediapnotation=True

        # initialize models
        self.exp_name = exp_name
        model = models.__dict__["gaze_aaai"]
        #self.models.resnet = model.resnet34(pretrained=True)
        # self.models.decoder = models.mixtr.MANet()#model.Decoder()
        self.models.decoder = models.wtdcnn.GANet()
        self.models.valdecoder = None
        #self.models.depth_loss = model.DepthBCE(0)
        #self.models.refine_depth = model.resnet34(pretrained=True)

        self.weights_init(self.models.decoder)
        #self.weights_init(self.models.refine_depth)

        self.scaler = GradScaler()

        # initialize extra variables
        self.extras.best_loss_base_val = 99999
        self.extras.best_loss_refine_val = 99999
        self.extras.last_epoch_headpose = -1
        self.extras.last_epoch_base = -1

        # initialize meter variables
        self.meters.loss_coord_train = {}
        self.meters.loss_depth_train = {}
        self.meters.loss_noncoplanar_train = {}
        self.meters.loss_coord_val = {}
        self.meters.loss_depth_val = {}
        self.meters.loss_noncoplanar_val = {}
        self.meters.prec_coord_train = {}
        self.meters.prec_coord_val = {}

        self.temps.base_solver = None
        self.temps.valbase_solver = None
        self.temps.scheduler = None

        if not is_cuda:
            self.logger.info("cpu-only mode specified, so using all detected CPU threads: "+str(os.cpu_count())+" as threads for PyTorch.")
            th.set_num_threads(os.cpu_count())
        
        # initialize visualizing
        self.visualize = visualize
        if visualize:
            import visdom
            import utils.vispyplot as vplt
            vplt.vis_config.server = 'http://localhost'
            vplt.vis_config.port = 5027
            vplt.vis_config.env = exp_name
            vplt.viz = visdom.Visdom(**vplt.vis_config)
            self.vplt = vplt

        if os.path.isfile(os.path.join(self.checkpoint_dir, "epoch_latest.pth.tar")):
            self.logger.info(f"load checkpoint @ {os.path.join(self.checkpoint_dir, 'epoch_latest.pth.tar')}")
            self.load_state_dict("epoch_latest.pth.tar")

    def log_experiment(self, separatebyexpname=False, okmark="okmark.txt", lastfold=999):
        with open(okmark, 'w') as f:
            f.write("ok")
        if separatebyexpname:
            logfile = './results/results_%s.txt' % self.exp_name
        else:
            logfile = './results/results.txt'
        lastepoch = self.extras.last_epoch_base
        if not self.actualtrain:
            print("experiment parameters and results are NOT logged because no actual training in this process call.")
            return self
        bestprecision = np.nanmin(list(self.meters.prec_coord_val.values()))
        bestprecisionepoch = np.nanargmin(list(self.meters.prec_coord_val.values()))
        lastprecision = self.meters.prec_coord_val[lastepoch]
        qk1 = self.meters.loss_noncoplanar_val[lastepoch]['qk1']
        qb1 = self.meters.loss_noncoplanar_val[lastepoch]['qb1']
        qk2 = self.meters.loss_noncoplanar_val[lastepoch]['qk2']
        qb2 = self.meters.loss_noncoplanar_val[lastepoch]['qb2']
        cb1 = self.meters.loss_noncoplanar_val[lastepoch]['cb1']
        cb2 = self.meters.loss_noncoplanar_val[lastepoch]['cb2']
        prec_LSQ10 = self.meters.loss_noncoplanar_val[lastepoch]['prec_LSQ10']
        prec_C10 = self.meters.loss_noncoplanar_val[lastepoch]['prec_C10']
        tqk1 = self.meters.loss_noncoplanar_train[lastepoch]['qk1']-qk1
        tqb1 = self.meters.loss_noncoplanar_train[lastepoch]['qb1']-qb1
        tqk2 = self.meters.loss_noncoplanar_train[lastepoch]['qk2']-qk2
        tqb2 = self.meters.loss_noncoplanar_train[lastepoch]['qb2']-qb2
        tcb1 = self.meters.loss_noncoplanar_train[lastepoch]['cb1']-cb1
        tcb2 = self.meters.loss_noncoplanar_train[lastepoch]['cb2']-cb2
        logstr = "%s,%d,%d,%d,%s,%s,%d,%d,%s,%s,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%d,%d,%f,%f\n" % (time.strftime("%Y-%m-%d %H:%M:%S "+self.exp_name+" "+os.path.basename(self.data_root)), self.fold_no, self.batch_size_train, self.batch_size_val, self.multitask, self.onlyeyeinput, self.augnum, self.facesize, self.usesmallface, self.targettype, qk1,qb1,qk2,qb2,cb1,cb2, tqk1,tqb1,tqk2,tqb2,tcb1,tcb2, prec_LSQ10, prec_C10, lastepoch, bestprecisionepoch, bestprecision, lastprecision)
        with open(logfile, '+a') as f:
            if f.tell() == 0:
                f.write("experimentname,foldno,trainbatchsize,valbatchsize,multitask,onlyeyeinput,augnum,facesize,usesmallface,targettype,qk1,qb1,qk2,qb2,cb1,cb2,tqk1,tqb1,tqk2,tqb2,tcb1,tcb2,prec_LSQ10,prec_C10,lastepoch,bestprecisionepoch,bestprecision,lastprecision\n")
            f.write(logstr)
        print('experiment parameters and results are logged to '+str(logfile))
        if self.fold_no == lastfold:
            bestpre = []
            lastpre = []
            lsqpre = []
            cpre = []
            with open(logfile, "r") as f:
                csvlines = f.readlines()
                csvlines = csvlines[1:]
                for line in csvlines:
                    values = line.split(",")
                    lsqpre.append(float(values[-6]))
                    cpre.append(float(values[-5]))
                    bestpre.append(float(values[-2]))
                    lastpre.append(float(values[-1]))
            bestpre = np.mean(bestpre)
            lastpre = np.mean(lastpre)
            lsqpre = np.mean(lsqpre)
            cpre = np.mean(cpre)
            
            print("best: %.2f last: %.2f lsq: %.2f c: %.2f" % (bestpre, lastpre, lsqpre, cpre))
            ent = input("Train complete, keep this log? [yes/no]:")
            if not (ent == 'yes'):
                os.remove(logfile)
            else:
                newlogfile = logfile.replace(".txt", "_%.2f_%.2f_%.2f_%.2f.txt" % (bestpre, lastpre, lsqpre, cpre))
                os.rename(logfile,newlogfile)
                print('experiment results of mean value of bestprecision and lastprecision of all folds are recorded to logfile name '+str(newlogfile))
        return self

    def train_base(self, epochs, lr=1e-4, use_refined_depth=False, fine_tune_headpose=True):
        # prepare logger
        self.temps.base_logger = self.logger.getChild('train_base')
        self.temps.base_logger.info('preparing for base training loop.')

        # prepare dataloader
        self.temps.train_loader = self._get_trainloader()
        self.temps.val_loader = self._get_valloader()
        self.temps.num_train_iters = len(self.temps.train_loader)
        self.temps.num_val_iters = len(self.temps.val_loader)
        self.temps.lr = lr
        self.temps.epochs = epochs

        self.temps.use_refined_depth = use_refined_depth
        self.temps.fine_tune_headpose = fine_tune_headpose
        # start training loop
        self.temps.epoch = self.extras.last_epoch_base
        self.temps.base_logger.info(f'start base training loop @ epoch {self.extras.last_epoch_base + 1}.')
        if self.extras.last_epoch_base + 1 == epochs:
            self.actualtrain = False
        for epoch in range(self.extras.last_epoch_base + 1, epochs):
            self.temps.epoch = epoch
            first_entrance=self.temps.base_solver is None
            if (((epoch+1) % 40) == 0) or first_entrance:
                # prepare solvers
                if self.temps.fine_tune_headpose:
                    self.temps.base_solver = optim.Adam(
                                                    [{'params':self.models.decoder.parameters(), 'initial_lr':self.temps.lr}], lr=self.temps.lr)
                    self.temps.scheduler = optim.lr_scheduler.StepLR(self.temps.base_solver, step_size=10, gamma=0.1, last_epoch=-1)
                    # self.temps.base_solver = optim.SGD(list(resnet.parameters()) +
                    #                                    list(decoder.parameters()) +
                    #                                    list(refine_depth.parameters()),
                    #                                    lr=self.temps.lr, weight_decay=5e-4)
                    if first_entrance and (self.extras.last_epoch_base>-1):
                        self.temps.base_solver.load_state_dict(self.temps.base_solver_dict)
                        if self.is_cuda:
                            for state in self.temps.base_solver.state.values():
                                for k,v in state.items():
                                    if isinstance(v, th.Tensor):
                                        state[k]=v.cuda()
                        self.temps.scheduler.load_state_dict(self.temps.scheduler_dict)
                else:
                    self.temps.base_solver = optim.SGD(self._group_weight(self.models.resnet, lr=self.temps.lr) +
                                                    self._group_weight(self.models.decoder, lr=self.temps.lr),
                                                    weight_decay=5e-4)
            # initialize meters for new epoch
            self._init_base_meters()
            # train one epoch
            self._train_base_epoch()
            # test on validation set
            #self._test_base(train=True)
            self._test_base(train=False)
            # save checkpoint
            self.extras.last_epoch_base = epoch
            self.save_state_dict(f'epoch_{epoch}.pth.tar',trainable=False)
            '''
            bestprecision = np.nanmin(list(self.meters.prec_coord_val.values()))
            lastprecision = self.meters.prec_coord_val[epoch]
            if (lastprecision - bestprecision) < 0.01:
                self.save_state_dict(f'epoch_best.pth.tar')
            '''
            self.save_state_dict(f'epoch_latest.pth.tar')
            self.temps.valbase_solver = None
            self.models.valdecoder = None
            # plot result
            if self.visualize:
                self._plot_base()
            # logging
            self._log_base()

        # cleaning
        # self.models.resnet.cpu()
        # self.models.decoder.cpu()
        del self.temps.train_loader
        del self.temps.val_loader

        return self

    def train_headpose(self, epochs, lr=2e-4, lambda_loss_mse=1):
        self.temps.lambda_loss_mse = lambda_loss_mse

        os.makedirs(os.path.join(self.result_root, "train", "depth"), exist_ok=True)
        os.makedirs(os.path.join(self.result_root, "val", "depth"), exist_ok=True)

        # prepare logger
        self.temps.headpose_logger = self.logger.getChild('train_headpose')
        self.temps.headpose_logger.info('preparing for headpose training loop.')

        # prepare dataloader
        self.temps.train_loader = self._get_trainloader()
        self.temps.val_loader = self._get_valloader()
        self.temps.num_iters = len(self.temps.train_loader)
        self.temps.epochs = epochs
        self.temps.lr = lr

        # start training loop
        self.temps.epoch = self.extras.last_epoch_headpose
        self.temps.headpose_logger.info(
            'start headpose training loop @ epoch {}.'.format(self.extras.last_epoch_headpose + 1))
        for epoch in range(self.extras.last_epoch_headpose + 1, epochs):
            self.temps.epoch = epoch
            # initialize meters for new epoch
            self._init_headpose_meters()
            # train one epoch
            self._train_headpose_epoch()
            # test on validation set
            self._test_headpose()
            # save checkpoint
            self.extras.last_epoch_headpose = epoch
            self.save_state_dict(f'epoch_{epoch}.pth.tar')
            self.save_state_dict(f'epoch_latest.pth.tar')

            # plot result
            if self.visualize:
                self._plot_headpose()
            # logging
            self._log_headpose()

        # cleaning
        self.models.refine_depth.cpu()
        del self.temps.train_loader
        del self.temps.val_loader

        return self

    def resume(self, filename):
        path = os.path.join(self.checkpoint_dir, filename)
        if os.path.isfile(path):
            self.load_state_dict(filename)
            self.logger.info('load checkpoint from {}'.format(path))
        return self

    def _prepare_model(self, model, train=True):
        if self.is_cuda:
            model.cuda()
            if not isinstance(model, nn.DataParallel):
                model = nn.DataParallel(model)
        if train:
            model.train()
        else:
            model.eval()
        return model

    def _get_trainloader(self):
        logger = self.logger

        transformed_train_dataset = GazePointAllDataset(root_dir=self.data_root,foldno = self.fold_no,
                                                        phase='train', faceimagesize=self.facesize,
                                                        face_image=True, eye_image=True,
                                                        info=True, eye_bbox=True, face_bbox=True, augnum=self.augnum, usesmallface=self.usesmallface, onlyeyeinput=self.onlyeyeinput, targettype=self.targettype)
        logger.info('The size of training data is: {}'.format(len(transformed_train_dataset)))
        train_loader = getDataLoader(transformed_train_dataset, batch_size=self.batch_size_train, shuffle=True,
                                       num_workers=self.num_workers, pin_memory=True)

        return train_loader

    def _get_valloader(self):
        logger = self.logger

        transformed_test_dataset = GazePointAllDataset(root_dir=self.val_data_root,foldno = self.fold_no,
                                                       phase='val', faceimagesize=self.facesize,
                                                       face_image=True, eye_image=True,
                                                       info=True, eye_bbox=True, face_bbox=True, usesmallface=self.usesmallface, onlyeyeinput=self.onlyeyeinput, targettype=self.targettype)

        logger.info('The size of testing data is: {}'.format(len(transformed_test_dataset)))

        test_loader = getDataLoader(transformed_test_dataset, batch_size=self.batch_size_val, shuffle=False,
                                      num_workers=self.num_workers, pin_memory=True)
        return test_loader

    def _init_base_meters(self):
        epoch = self.temps.epoch
        self.meters.loss_coord_train[epoch] = []
        self.meters.loss_coord_val[epoch] = 0
        self.meters.loss_noncoplanar_train[epoch] = []
        self.meters.loss_noncoplanar_val[epoch] = 0
        self.meters.prec_coord_train[epoch] = []
        self.meters.prec_coord_val[epoch] = 0

    def _init_headpose_meters(self):
        epoch = self.temps.epoch
        self.meters.loss_depth_train[epoch] = []
        self.meters.loss_depth_val[epoch] = 0

    def _plot_base(self):
        with self.vplt.set_draw(name='loss_base', figsize=(10,6)) as ax:
            ax.plot(list(self.meters.loss_coord_train.keys()),
                    np.mean(list(self.meters.loss_coord_train.values()), axis=1), label='loss_coord_train')
            ax.plot(list(self.meters.loss_coord_val.keys()),
                    list(self.meters.loss_coord_val.values()), label='loss_coord_val')
            ax.set_title('loss base')
            ax.set_xlabel('epoch')
            ax.set_ylabel('loss')
            ax.set_xlim(0, self.temps.epochs)
            ax.set_ylim(0, 5)
            ax.legend()
            ax.grid(True)
        with self.vplt.set_draw(name='prec_base', figsize=(20,12)) as ax:
            ax.plot(list(self.meters.prec_coord_train.keys()),
                    np.mean(list(self.meters.prec_coord_train.values()), axis=1), label='prec_coord_train')
            ax.plot(list(self.meters.prec_coord_val.keys()),
                    list(self.meters.prec_coord_val.values()), label='prec_coord_val')
            for record in self.meters.loss_noncoplanar_train[0].keys():
                if record[0] == 'p':
                    ax.plot(list(self.meters.loss_noncoplanar_train.keys()),
                            list([v[record] for v in self.meters.loss_noncoplanar_train.values()]), label='prec_coord_%s_train' % record)
            for record in self.meters.loss_noncoplanar_val[0].keys():
                if record[0] == 'p':
                    ax.plot(list(self.meters.loss_noncoplanar_val.keys()),
                            list([v[record] for v in self.meters.loss_noncoplanar_val.values()]), label='prec_coord_%s_val' % record)
            ax.set_title('prec base')
            ax.set_xlabel('epoch')
            ax.set_ylabel('prec')
            ax.set_xlim(0, self.temps.epochs)
            ax.set_ylim(0, 20)
            ax.legend()
            ax.grid(True)

        for phase in ['train', 'val']:
            for newold in ['new', 'old']:
                for angle in ['theta', 'phi']:
                    with self.vplt.set_draw(name='%s_%s_%s'%(phase,newold,angle), figsize=(4,4)) as ax:
                        #phase, newold, angle = ax.name.split('_')
                        if phase == 'train':
                            dct = self.meters.loss_noncoplanar_train
                        else:
                            dct = self.meters.loss_noncoplanar_val
                        if newold == 'old':
                            prefix = 'o'
                        else:
                            prefix = ''
                        if angle == 'theta':
                            idx = 0
                        else:
                            idx = 1
                        data = dct[max(list(dct.keys()))]
                        gt = data[prefix+'target'][:,idx]
                        pred = data[prefix+'coord'][:,idx]
                        x_line = np.linspace(-180, 180, 100)
                        y_line = x_line
                        k = data['qk'+str(idx+1)]
                        b = data['qb'+str(idx+1)]
                        c = data['cb'+str(idx+1)]
                        ax.plot(x_line, y_line, label='Pred=Gt', color='blue')
                        ax.plot(x_line*k+b, x_line, label='Linear fitting', color='green')
                        ax.plot(x_line+c, x_line, label='Constant fitting', color='yellow')
                        ax.scatter(gt, pred, color='red', s=1)
                        ax.set_title(ax.name)
                        ax.set_xlabel('Groundtruth')
                        ax.set_ylabel('Prediction')
                        ax.set_xlim(-180, 180)
                        ax.set_ylim(-180, 180)
                        ax.set_xticks([-180,-120,-60,0,60,120,180])
                        ax.set_yticks([-180,-120,-60,0,60,120,180])
                        ax.legend()

    def _plot_each_iter_base_ShanghaiTechGaze(self, facepic, facebbox, predictedpoint_screencoord, gtpoint_screencoord, lefteye_pixelcoord, righteye_pixelcoord, precision):
        print(facebbox)
        print(lefteye_pixelcoord)
        print(righteye_pixelcoord)
        # Below are extrinsics, extracted from 'ex.txt' of the 'camera' folder of the repo:
        # Although the author do not specify, referring to the common best practice, 
        # we first guess and assume that the depth image is aligned to color image, and the extrinsics is the color camera's.
        # After observation on the Figure1 of the paper and the T value, we guess and assume that the original T value is in centimeters,
        # and the world coordinate system is originated in the left lower corner of the screen,
        # with the same axis direction as the camera coordinate system.
        # Note that the guessed world coordinate system is not the O'X'Y'Z' coordinate system on the Figure1.
        # After test, we found that only in above case can we extract correct coordinates.
        RCOLOR = R = np.transpose(np.array([[0.999746,0.000291,-0.022534], [0.000156,0.999803,0.019831], [0.022535,-0.019830,0.999549]]))
        TCOLOR = T = 0.01 * np.transpose(np.array([23.590678,-0.954391,-4.049306]))
        # Below are intrinsics for color camera, extracted from 'color.mat'(matlab matfile) of the 'camera folder' of the repo:
        KCOLOR = np.transpose(np.array([[1384.2,0,0], [0,1381.3,0], [972.8,539.0,1.0]]))
        # Below are intrinsics for depth camera, extracted from 'depth.mat'(matlab matfile) of the 'camera folder' of the repo:
        KDEPTH = np.transpose(np.array([[486.2349,0,0], [0,483.7896,0], [322.1105,244.0338,1.0]]))

        # Then we restore the preprocessed value according to gaze_dataset.py,
        # and retrieve the actual depth value in meter, according to the tech parameters of the Intel RealSense SR300 device
        # which is declared in the paper to be used as the RGBD camera of the repo.
        lefteye_pixelcoord = np.array([1920*lefteye_pixelcoord[0], 1080*lefteye_pixelcoord[1], 0.000125*(lefteye_pixelcoord[2]*500+500)])
        righteye_pixelcoord = np.array([1920*righteye_pixelcoord[0], 1080*righteye_pixelcoord[1], 0.000125*(righteye_pixelcoord[2]*500+500)])
        # We transform the original pixel coordinate to the facepic pixel coordinate using the facebbox record.
        lefteye_facepicpixelcoord = facepic.shape[0] * (lefteye_pixelcoord[:2] - facebbox[:2]) / (facebbox[2] - facebbox[0])
        righteye_facepicpixelcoord = facepic.shape[0] * (righteye_pixelcoord[:2] - facebbox[:2]) / (facebbox[2] - facebbox[0])
        # pixel coordinate -> camera coordinate
        lefteye_cameracoord = lefteye_pixelcoord[2] * np.matmul(np.linalg.inv(KCOLOR), np.array([lefteye_pixelcoord[0], lefteye_pixelcoord[1], 1]))
        righteye_cameracoord = righteye_pixelcoord[2] * np.matmul(np.linalg.inv(KCOLOR), np.array([righteye_pixelcoord[0], righteye_pixelcoord[1], 1]))
        print(lefteye_cameracoord)
        print(righteye_cameracoord)

        # We first assume the screencoordinate is in centimeters,
        # and then transform the screen coordinate to world coordinate using this and above assumption. 
        predictedpoint_worldcoord = 0.01 * np.array([-predictedpoint_screencoord[0]-59.77*0.5, predictedpoint_screencoord[1]+33.62*0.5-33.62, 0])
        # world coordinate -> camera coordinate
        predictedpoint_cameracoord = np.matmul(RCOLOR, predictedpoint_worldcoord) + TCOLOR
        print(predictedpoint_cameracoord)

        # We compute each eye's gaze visual endpoint(center point of the line from eye to predictedpoint) on facepic.
        a = predictedpoint_cameracoord[0]
        b = predictedpoint_cameracoord[1]
        c = predictedpoint_cameracoord[2]
        dl = lefteye_cameracoord[0]
        el = lefteye_cameracoord[1]
        fl = lefteye_cameracoord[2]
        dr = righteye_cameracoord[0]
        er = righteye_cameracoord[1]
        fr = righteye_cameracoord[2]
        zl = lefteye_cameracoord[2] * 0.5 # z value of left eye's gaze visual point
        zr = righteye_cameracoord[2] * 0.5 # z value of right eye's gaze visual point
        # We compute the gaze visual point using geometric rules.
        leftgazevisendpoint_cameracoord = np.array([((a-dl)*zl + dl*c - a*fl)/c-fl, ((b-el)*zl + el*c - b*fl)/c-fl, zl])
        rightgazevisendpoint_cameracoord = np.array([((a-dr)*zr + dr*c - a*fr)/c-fr, ((b-er)*zr + er*c - b*fr)/c-fr, zr])
        # camera coordinate -> pixel coordinate
        leftgazevisendpoint_pixelcoord = np.matmul(KCOLOR, leftgazevisendpoint_cameracoord)[:2] / zl
        rightgazevisendpoint_pixelcoord = np.matmul(KCOLOR, rightgazevisendpoint_cameracoord)[:2] / zr
        # We transform the original pixel coordinate to the facepic pixel coordinate using the facebbox record.
        leftgazevisendpoint_facepicpixelcoord = facepic.shape[0] * (leftgazevisendpoint_pixelcoord - facebbox[:2]) / (facebbox[2] - facebbox[0])
        rightgazevisendpoint_facepicpixelcoord = facepic.shape[0] * (rightgazevisendpoint_pixelcoord - facebbox[:2]) / (facebbox[2] - facebbox[0])
        print(lefteye_facepicpixelcoord)
        print(righteye_facepicpixelcoord)
        print(leftgazevisendpoint_facepicpixelcoord)
        print(rightgazevisendpoint_facepicpixelcoord)

        facepic = facepic.copy()
        cv2.line(facepic, lefteye_facepicpixelcoord.astype(np.int), leftgazevisendpoint_facepicpixelcoord.astype(np.int), (0, 0, 255), 8)
        cv2.line(facepic, righteye_facepicpixelcoord.astype(np.int), rightgazevisendpoint_facepicpixelcoord.astype(np.int), (0, 0, 255), 8)

        with vplt.set_draw(name='facepic', figsize=(10,10)) as ax:
            ax.set_title('input face picture', fontsize=20)
            ax.imshow(facepic)
        with vplt.set_draw(name='coord', figsize=(10,10)) as ax:
            ax.scatter(predictedpoint_screencoord[0], predictedpoint_screencoord[1], marker='.', c=['red'], s=250, label="prediction")
            ax.scatter(gtpoint_screencoord[0], gtpoint_screencoord[1], marker='.', c=['green'], s=250, label="ground truth")
            #ax.annotate('precision: %.2f(cm)' % prec,xy=(coord[0],coord[1]),xytext=(+10,-10),textcoords='offset points',
            # fontsize=10,arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=.2'))
            ax.set_title('output screen coord, precision: %.2f(cm)' % precision, fontsize=20)
            ax.set_xlabel('screen_x(cm)', fontsize=20)
            ax.set_ylabel('screen_y(cm)', fontsize=20)
            ax.set_xlim(-59.77*0.5, 59.77*0.5)
            ax.set_ylim(-33.62*0.5, 33.62*0.5)
            ax.xaxis.set_ticks_position('top')
            ax.yaxis.set_ticks_position('left')
            ax.invert_yaxis()
            ax.legend(loc="lower left", bbox_to_anchor=(0,0,1,1), fontsize=20)
            ax.grid(True)

    def _plot_headpose(self):
        with vplt.set_draw(name='loss_depth') as ax:
            ax.plot(list(self.meters.loss_depth_train.keys()),
                    np.mean(list(self.meters.loss_depth_train.values()), axis=1), label='loss_depth_train')
            ax.plot(list(self.meters.loss_depth_val.keys()),
                    list(self.meters.loss_depth_val.values()), label='loss_depth_val')
            ax.set_title('loss depth')
            ax.set_xlabel('epoch')
            ax.set_ylabel('loss')
            ax.set_xlim(0, self.temps.epochs)
            ax.legend()
            ax.grid(True)

    def _log_base(self):
        infofmt = "*[{temps.epoch}]\t" \
                  "prec_coord_train: {prec_coord_train:.4f} prec_coord_val: {prec_coord_val:.4f}\t" \
                  "loss_coord_train: {loss_coord_train:.4f} loss_coord_val: {loss_coord_val:.4f}\t" \
                  "loss_noncoplanar_train: {loss_noncoplanar_train:.4f} loss_noncoplanar_val: {loss_noncoplanar_val:.4f}"
        infodict = dict(
            temps=self.temps,
            prec_coord_train=np.mean(self.meters.prec_coord_train[self.temps.epoch]),
            prec_coord_val=np.mean(self.meters.prec_coord_val[self.temps.epoch]),
            loss_coord_train=np.mean(self.meters.loss_coord_train[self.temps.epoch]),
            loss_coord_val=np.mean(self.meters.loss_coord_val[self.temps.epoch]),
            loss_noncoplanar_train=np.mean(self.meters.loss_noncoplanar_train[self.temps.epoch]['prec_LSQF']),
            loss_noncoplanar_val=np.mean(self.meters.loss_noncoplanar_val[self.temps.epoch]['prec_LSQF'])
        )
        self.temps.base_logger.info(infofmt.format(**infodict))

    def _log_headpose(self):
        infofmt = "*[{temps.epoch}]\t" \
                  "loss_depth_train: {loss_depth_train:.4f} loss_depth_val: {loss_depth_val:.4f}\t"
        infodict = dict(
            temps=self.temps,
            loss_depth_train=np.mean(self.meters.loss_depth_train[self.temps.epoch]),
            loss_depth_val=np.mean(self.meters.loss_depth_val[self.temps.epoch]),
        )
        self.temps.headpose_logger.info(infofmt.format(**infodict))

    def _train_base_epoch(self):
        logger = self.temps.base_logger.getChild('epoch')
        # prepare models
        #resnet = self._prepare_model(self.models.resnet)
        decoder = self._prepare_model(self.models.decoder)
        #refine_depth = self._prepare_model(self.models.refine_depth, train=True)
        #depth_loss = self.models.depth_loss
        device = th.device("cpu") if not self.is_cuda else th.device("cuda")

        self.timeit()
        coord_cumu=[]
        target_cumu=[]
        headposematrix_cumu=[]
        horizonflip = transforms.RandomHorizontalFlip(p=1)
        for i, batch in enumerate(self.temps.train_loader):
            self.temps.iter = i
            # prepare data
            #a = time.time()
            face_image, face_bbox, \
            left_eye_image, left_eye_depth, left_eye_normal, \
            right_eye_image, right_eye_depth, right_eye_normal, \
            target, \
            face_factor, headposematrix, headposevector, face_depth, face_normal, face_meshattrib = \
                batch['face_image'].to(device, non_blocking=False), \
                batch["face_bbox"].to(device, non_blocking=False), \
                batch["left_eye_image"].to(device, non_blocking=False), \
                batch["left_eye_depth"].to(device, non_blocking=False), \
                batch["left_eye_normal"].to(device, non_blocking=False), \
                batch["right_eye_image"].to(device, non_blocking=False), \
                batch["right_eye_depth"].to(device, non_blocking=False), \
                batch["right_eye_normal"].to(device, non_blocking=False), \
                batch["gt"].to(device, non_blocking=False), \
                batch["face_scale_factor"].to(device, non_blocking=False), \
                batch["headposematrix"].to(device, non_blocking=False), \
                batch["headposevector"].to(device, non_blocking=False), \
                batch["facedepthimg"].to(device, non_blocking=False),\
                batch["facenormalimg"].to(device, non_blocking=False),\
                batch["facemeshattrib"].to(device, non_blocking=False)
            #b = time.time()
            #print('copy time:%f' % (b-a))
            # measure data loading time
            #print(type(batch['face_image']))
            #print(face_image.shape)
            #ttarget = []
            #for iaug in range(otarget.shape[1]):
            #    ttarget.append(th.FloatTensor(self._convheadpose_numpy(otarget[:,iaug].cpu().numpy(), headposematrix.cpu().numpy(), self.iseyediapnotation)))
            #target = th.stack(ttarget, dim=1)
            #target = target.to(device, non_blocking=False)
            self.temps.data_time = self._timeit()

            # forward
            loss_coord_cumu = th.zeros(1).to(device)
            loss_noncoplanar = th.zeros(1).to(device)
            loss_noncoplanar_cumu = th.zeros(1).to(device)
            prec_coord = th.zeros(1).to(device)
            self.temps.base_solver.zero_grad()
            for iaug in range(0, self.augnum):
                with autocast(True):
                    '''
                    lfeat = resnet(left_eye_image)
                    rfeat = resnet(right_eye_image)

                    if self.temps.fine_tune_headpose:
                        head_pose = refine_depth(face_image[:,iaug])
                    else:
                        with th.no_grad():
                            head_pose = refine_depth(face_image[:,iaug])
                    '''
                    if self.temps.use_refined_depth:
                        '''
                        with th.no_grad():
                            left_eye_bbox[:, :2] -= face_bbox[:, :2]
                            left_eye_bbox[:, 2:] -= face_bbox[:, :2]
                            right_eye_bbox[:, :2] -= face_bbox[:, :2]
                            right_eye_bbox[:, 2:] -= face_bbox[:, :2]
                            left_eye_bbox = th.clamp(face_factor * left_eye_bbox, min=0, max=223)
                            right_eye_bbox = th.clamp(face_factor * right_eye_bbox, min=0, max=223)
                        

                        for j, lb in enumerate(left_eye_bbox):
                            cur_depth = refined_depth[j, :, int(lb[1]):int(lb[3]), int(lb[0]):int(lb[2])]
                            left_eye_info[j, 2] = th.median(cur_depth).item() * face_factor
                        for j, rb in enumerate(right_eye_bbox):
                            cur_depth = refined_depth[j, :, int(rb[1]):int(rb[3]), int(rb[0]):int(rb[2])]
                            right_eye_info[j, 2] = th.median(cur_depth).item() * face_factor
                        '''
                    if self.targettype == 'gazepoint':
                        lcoord, rcoord = decoder(lfeat, rfeat, head_pose, left_eye_info[:,iaug], right_eye_info[:,iaug])
                    else:
                        lcoord, rcoord = decoder(left_eye_image, right_eye_image, face_image[:,iaug], left_eye_normal, right_eye_normal, left_eye_depth, right_eye_depth, face_depth, face_normal, face_meshattrib)
                        '''
                        left_eye_image_hflip = horizonflip(left_eye_image)
                        right_eye_image_hflip = horizonflip(right_eye_image)
                        face_image_hflip = horizonflip(face_image[:,iaug])
                        left_eye_normal_hflip = horizonflip(left_eye_normal)
                        left_eye_normal_hflip[:,0] = ((- (left_eye_normal_hflip[:,0]*2 - 1))+1)*0.5
                        right_eye_normal_hflip = horizonflip(right_eye_normal)
                        right_eye_normal_hflip[:,0] = ((- (right_eye_normal_hflip[:,0]*2 - 1))+1)*0.5
                        lcoord_hflip, rcoord_hflip = decoder(right_eye_image_hflip, left_eye_image_hflip, face_image_hflip, right_eye_normal_hflip, left_eye_normal_hflip)
                        lcoord_hflip[:,0]*=-1 
                        '''
                        #lcoord = 0.5*(lcoord+lcoord_hflip)
                    if iaug == 0:
                        factor = 0.5
                        if self.augnum == 1:
                            factor = 1.0
                        #if self.multitask:
                        #    loss_depth = depth_loss(refined_depth, face_depth[:,iaug])
                    else:
                        factor = 0.5 / (self.augnum-1)
                    
                    if self.targettype == 'gazepoint' or (self.targettype == 'gazedirection' and self.oneortwotarget == 1):
                        lossl = self._gazeangularloss(lcoord, target[:,iaug], self.iseyediapnotation)*th.pi/180
                        '''
                        lcoord_wrthead = self._convheadpose_torch(lcoord, headposematrix, self.iseyediapnotation, hflip=False)
                        lcoord_hflip_wrthead = self._convheadpose_torch(lcoord_hflip, headposematrix, self.iseyediapnotation, hflip=False)
                        #losslhflip = F.mse_loss(lcoord_hflip, target[:,iaug])
                        lossdiv = F.l1_loss(lcoord_wrthead[:,1], lcoord_hflip_wrthead[:,1])
                        #lossl+=losslhflip
                        #lossl*=0.5
                        lossl+=lossdiv*0.1
                        '''
                        
                        #lossr = F.l1_loss(rcoord, target[:,iaug])
                        loss_coord = factor * lossl
                        loss_misc = loss_coord
                    else:
                        loss_coord = factor * (F.l1_loss(lcoord, target[:,iaug,:2]) + F.l1_loss(rcoord, target[:,iaug,2:])) * 0.5
                        ltheta = lcoord[:,0]
                        lphi = lcoord[:,1]
                        rtheta = rcoord[:,0]
                        rphi = rcoord[:,1]
                        lvector = th.stack([th.sin(ltheta)*th.cos(lphi), th.sin(ltheta)*th.sin(lphi), th.cos(ltheta)]).T
                        rvector = th.stack([th.sin(rtheta)*th.cos(rphi), th.sin(rtheta)*th.sin(rphi), th.cos(rtheta)]).T
                        twoeyevector = left_eye_info[:,iaug] - right_eye_info[:,iaug]
                        distance = th.abs(th.tensordot(th.cross(lvector, rvector), twoeyevector, dims=([1],[1])))
                        loss_noncoplanar = factor * th.mean(distance / th.linalg.norm(twoeyevector, dim=1))
                        loss_misc = loss_coord# + loss_noncoplanar
                    with autocast(enabled=False):
                        loss_coord_cumu+= loss_coord.float()
                        loss_noncoplanar_cumu+= loss_noncoplanar.float()
                        if self.targettype == 'gazepoint':
                            prec_coord+= (self._gazepointprecision(lcoord.data.float(), target[:,iaug].data.float()) + self._gazepointprecision(rcoord.data.float(), target[:,iaug].data.float())) * 0.5 / self.augnum
                        elif self.targettype == 'gazedirection' and self.oneortwotarget == 2:
                            prec_coord+= (self._gazedirectionprecision(lcoord.data.float(), target[:,iaug,:2].data.float(), self.iseyediapnotation) + self._gazedirectionprecision(rcoord.data.float(), target[:,iaug,2:].data.float(), self.iseyediapnotation)) * 0.5 / self.augnum
                        else:
                            prec_coord+= (self._gazedirectionprecision(lcoord.data.float(), target[:,iaug].data.float(), self.iseyediapnotation) + self._gazedirectionprecision(rcoord.data.float(), target[:,iaug].data.float(), self.iseyediapnotation)) * 0.5 / self.augnum
                    coord_cumu.append(lcoord.detach().cpu().numpy())
                    target_cumu.append(target[:,iaug].detach().cpu().numpy())
                    headposematrix_cumu.append(headposematrix.detach().cpu().numpy())
                #if iaug == 0 and self.multitask:
                #    self.scaler.scale(loss_depth).backward(retain_graph=True)                
                #with th.autograd.set_detect_anomaly(True):
                self.scaler.scale(loss_misc).backward()
            '''
            # for visualization
            if self.visualize:
                for ct in range(len(face_image)):
                    self._plot_each_iter_base_ShanghaiTechGaze(batch['face_image_ori'][ct].cpu().detach().numpy(),
                    face_bbox[ct].cpu().detach().numpy(), coord[ct].cpu().detach().numpy(), 
                    target[ct].cpu().detach().numpy(), left_eye_info[ct].cpu().detach().numpy(), 
                    right_eye_info[ct].cpu().detach().numpy(), prec_coord.cpu().detach().numpy())
            '''
            # update resnet & decoder
            self.scaler.step(self.temps.base_solver)
            
            self.scaler.update()

            # record loss & accuracy
            epoch = self.temps.epoch
            self.meters.loss_coord_train[epoch].append(loss_coord_cumu.item())
            self.meters.prec_coord_train[epoch].append(prec_coord.item())
            self.meters.loss_noncoplanar_train[epoch].append(loss_noncoplanar_cumu.item())

            # measure batch time
            self.temps.batch_time = self._timeit()

            # logging
            info = f"[Fold{self.fold_no}][{self.temps.epoch}][{self.temps.iter}/{self.temps.num_train_iters}]\t" \
                   f"data_time: {self.temps.data_time:.2f} batch_time: {self.temps.batch_time:.2f}\t" \
                   f"prec_coord_train: {self.meters.prec_coord_train[self.temps.epoch][-1]:.4f}\t" \
                   f"loss_coord_train: {self.meters.loss_coord_train[self.temps.epoch][-1]:.4f}\t" \
                   f"loss_noncoplanar_train: {self.meters.loss_noncoplanar_train[self.temps.epoch][-1]:.4f}"
            # infodict = dict(
            #     temps=self.temps,
            #     prec_coord_train=self.meters.prec_coord_train[self.temps.epoch][-1],
            #     loss_coord_train=self.meters.loss_coord_train[self.temps.epoch][-1],
            # )
            logger.info(info)
        self.temps.scheduler.step()
        
        ocoord_cumu = np.vstack(coord_cumu)
        otarget_cumu = np.vstack(target_cumu)
        coord_cumu = self._convheadpose_numpy(ocoord_cumu, headposematrix_cumu, self.iseyediapnotation)
        target_cumu = self._convheadpose_numpy(otarget_cumu, headposematrix_cumu, self.iseyediapnotation)
        preco = self._gazedirectionprecision(ocoord_cumu, otarget_cumu, self.iseyediapnotation, usenumpy=True).item()
        precn = self._gazedirectionprecision(coord_cumu, target_cumu, self.iseyediapnotation, usenumpy=True).item()
        precdelta = precn-preco
        logger.info("preco:%f, precn:%f, precdelta:%f"%(preco, precn, precdelta))
        #coord_cumu = ocoord_cumu
        #target_cumu = otarget_cumu
        prec_LSQF = self._fittarget(coord_cumu, target_cumu, -1, 1, 0, self.iseyediapnotation)
        prec_CF = self._fittarget(coord_cumu, target_cumu, -1, 1, 1, self.iseyediapnotation)
        self.meters.loss_noncoplanar_train[epoch] = {'coord':coord_cumu*180/np.pi, 'target':target_cumu*180/np.pi, 'ocoord':ocoord_cumu*180/np.pi, 'otarget':otarget_cumu*180/np.pi, 'prec_LSQF':prec_LSQF[0], 'prec_CF':prec_CF[0], 'qk1':prec_LSQF[1], 'qb1':prec_LSQF[2]*180/np.pi, 'qk2':prec_LSQF[3], 'qb2':prec_LSQF[4]*180/np.pi, 'cb1':prec_CF[2]*180/np.pi, 'cb2':prec_CF[4]*180/np.pi}
        logger.info(json.dumps(self.meters.loss_noncoplanar_train[epoch], default=ignorenumpy))

    def _train_headpose_epoch(self):
        logger = self.temps.headpose_logger.getChild('epoch')
        # prepare models
        refine_depth = self._prepare_model(self.models.refine_depth)
        depth_loss = self.models.depth_loss
        device = th.device("cpu") if not self.is_cuda else th.device("cuda")
        # prepare solvers
        self.temps.headpose_solver = optim.SGD(self._group_weight(self.models.refine_depth, lr=self.temps.lr),
                                               weight_decay=5e-4)

        self.timeit()
        for i, batch in enumerate(self.temps.train_loader):
            self.temps.iter = i
            # prepare data
            face_image = batch['face_image'].to(device)
            face_depth = batch['face_depth'].to(device)

            # measure data loading time
            self.temps.data_time = self._timeit()

            # forward
            head_pose, refined_depth = refine_depth(face_image, face_depth)
            loss_depth = depth_loss(refined_depth, face_depth)

            # update resnet & decoder
            self.temps.headpose_solver.zero_grad()
            loss_depth.backward()
            self.temps.headpose_solver.step()

            # record loss & accuracy
            epoch = self.temps.epoch
            self.meters.loss_depth_train[epoch].append(loss_depth.item())

            # visualize and save results
            face_depth.detach_()
            refined_depth.detach_()
            with th.no_grad():
                depth_grid_gt = make_grid(face_depth).cpu()
                depth_grid_rf = make_grid(refined_depth).cpu()
                if self.visualize:
                    with vplt.set_draw(name='train_depth_groundtruth') as ax:
                        ax.imshow(depth_grid_gt.numpy().transpose((1, 2, 0)))
                    with vplt.set_draw(name='train_depth_refined') as ax:
                        ax.imshow(depth_grid_rf.numpy().transpose((1, 2, 0)))
                #save_image(depth_grid_gt, os.path.join(self.result_root, "train", "depth", f"ep{self.temps.epoch:02d}iter{i:04d}_gt.png"))
                #save_image(depth_grid_rf, os.path.join(self.result_root, "train", "depth", f"ep{self.temps.epoch:02d}iter{i:04d}_rf.png"))

            # measure batch time
            self.temps.batch_time = self._timeit()

            # logging
            infofmt = "[{temps.epoch}][{temps.iter}/{temps.num_iters}]\t" \
                      "data_time: {temps.data_time:.2f} batch_time: {temps.batch_time:.2f}\t" \
                      "loss_depth_train: {loss_depth_train:.4f} "
            infodict = dict(
                temps=self.temps,
                loss_depth_train=self.meters.loss_depth_train[epoch][-1],
            )
            logger.info(infofmt.format(**infodict))

    def _test_base(self, train=False):
        logger = self.temps.base_logger.getChild('val')
        #resnet = self._prepare_model(self.models.resnet, train=False)
        if self.models.valdecoder is None:
            self.models.valdecoder = copy.deepcopy(self.models.decoder)
        decoder = self._prepare_model(self.models.valdecoder, train=train)
        #refine_depth = self._prepare_model(self.models.refine_depth, train=False)
        loss_lcoord, loss_rcoord, loss_coord, loss_noncoplanar, prec_lcoord, prec_rcoord, prec_coord, num_batches = 0, 0, 0, 0, 0, 0, 0, 0
        device = th.device("cpu") if not self.is_cuda else th.device("cuda")
        horizonflip = transforms.RandomHorizontalFlip(p=1)
        self.timeit()
        if train and self.temps.valbase_solver is None:
            self.temps.valbase_solver = optim.Adam([{'params':self.models.valdecoder.parameters(), 'initial_lr':1e-4}], lr=1e-4)
        for i, batch in enumerate(self.temps.val_loader):
            self.temps.iter = i
            # prepare data
            face_image, face_bbox, \
            left_eye_image, left_eye_depth, left_eye_normal, \
            right_eye_image, right_eye_depth, right_eye_normal, \
            target, \
            face_factor, headposematrix, headposevector, face_depth, face_normal, face_meshattrib = \
                batch['face_image'].transpose(1,0)[0].to(device, non_blocking=False), \
                batch["face_bbox"].to(device, non_blocking=False), \
                batch["left_eye_image"].to(device, non_blocking=False), \
                batch["left_eye_depth"].to(device, non_blocking=False), \
                batch["left_eye_normal"].to(device, non_blocking=False), \
                batch["right_eye_image"].to(device, non_blocking=False), \
                batch["right_eye_depth"].to(device, non_blocking=False), \
                batch["right_eye_normal"].to(device, non_blocking=False), \
                batch["gt"].transpose(1,0)[0].to(device, non_blocking=False), \
                batch["face_scale_factor"].to(device, non_blocking=False), \
                batch["headposematrix"].to(device, non_blocking=False), \
                batch["headposevector"].to(device, non_blocking=False),\
                batch["facedepthimg"].to(device, non_blocking=False),\
                batch["facenormalimg"].to(device, non_blocking=False),\
                batch["facemeshattrib"].to(device, non_blocking=False)
            '''
            target = th.FloatTensor(self._convheadpose_numpy(otarget.cpu().numpy(), headposematrix.cpu().numpy(), self.iseyediapnotation))
            target = target.to(device, non_blocking=False)
            '''
            self.temps.val_data_time = self._timeit()

            if train:
                gradclass=th.enable_grad()
            else:
                gradclass=th.no_grad()
            # forward
            with gradclass:
                loss_noncoplanar_iter = th.zeros(1).to(device).float()
                if train:
                    self.temps.valbase_solver.zero_grad()
                '''
                lfeat = resnet(left_eye_image)
                rfeat = resnet(right_eye_image)
                head_pose = refine_depth(face_image)
                '''
                '''
                left_eye_bbox[:, :2] -= face_bbox[:, :2]
                left_eye_bbox[:, 2:] -= face_bbox[:, :2]
                right_eye_bbox[:, :2] -= face_bbox[:, :2]
                right_eye_bbox[:, 2:] -= face_bbox[:, :2]
                left_eye_bbox = th.clamp(face_factor * left_eye_bbox, min=0, max=223)
                right_eye_bbox = th.clamp(face_factor * right_eye_bbox, min=0, max=223)
                
                if self.temps.use_refined_depth:
                    for j, lb in enumerate(left_eye_bbox):
                        cur_depth = refined_depth[j, :, int(lb[1]):int(lb[3]), int(lb[0]):int(lb[2])]
                        left_eye_info[j, 2] = th.median(cur_depth).item() * face_factor
                    for j, rb in enumerate(right_eye_bbox):
                        cur_depth = refined_depth[j, :, int(rb[1]):int(rb[3]), int(rb[0]):int(rb[2])]
                        right_eye_info[j, 2] = th.median(cur_depth).item() * face_factor
                '''
                if self.targettype == 'gazepoint':
                    lcoord, rcoord = decoder(lfeat, rfeat, head_pose, left_eye_info, right_eye_info)
                else:
                    lcoord, rcoord = decoder(left_eye_image, right_eye_image, face_image, left_eye_normal, right_eye_normal, left_eye_depth, right_eye_depth, face_depth, face_normal, face_meshattrib)
                    '''
                    left_eye_image_hflip = horizonflip(left_eye_image)
                    right_eye_image_hflip = horizonflip(right_eye_image)
                    face_image_hflip = horizonflip(face_image)
                    left_eye_normal_hflip = horizonflip(left_eye_normal)
                    left_eye_normal_hflip[:,0] = ((- (left_eye_normal_hflip[:,0]*2 - 1))+1)*0.5
                    right_eye_normal_hflip = horizonflip(right_eye_normal)
                    right_eye_normal_hflip[:,0] = ((- (right_eye_normal_hflip[:,0]*2 - 1))+1)*0.5

                    lcoord_hflip, rcoord_hflip = decoder(right_eye_image_hflip, left_eye_image_hflip, face_image_hflip, right_eye_normal_hflip, left_eye_normal_hflip)
                    lcoord_hflip[:,0]*=-1 
                    '''
                    #lcoord = 0.5*(lcoord+lcoord_hflip)
                th.cuda.synchronize()
                self.temps.val_batch_time = self._timeit()
                if self.targettype == 'gazepoint' or (self.targettype == "gazedirection" and self.oneortwotarget == 1):
                    '''
                    lcoord_wrthead = self._convheadpose_torch(lcoord, headposematrix, self.iseyediapnotation, hflip=False)
                    lcoord_hflip_wrthead = self._convheadpose_torch(lcoord_hflip, headposematrix, self.iseyediapnotation, hflip=False)
                    loss_div = th.var((lcoord_wrthead-lcoord_hflip_wrthead)[:,0], dim=0, keepdim=False) + F.l1_loss(lcoord_wrthead[:,1],lcoord_hflip_wrthead[:,1])
                    loss_coord_iter = loss_div#F.mse_loss(lcoord, target)# + F.l1_loss(rcoord, target)) * 0.5
                    '''
                    loss_coord_iter = self._gazeangularloss(lcoord, target, self.iseyediapnotation)
                else:
                    loss_coord_iter = (F.l1_loss(lcoord, target[:,:2]) + F.l1_loss(rcoord, target[:,2:])) * 0.5
                    ltheta = lcoord[:,0]
                    lphi = lcoord[:,1]
                    rtheta = rcoord[:,0]
                    rphi = rcoord[:,1]
                    lvector = th.stack([th.sin(ltheta)*th.cos(lphi), th.sin(ltheta)*th.sin(lphi), th.cos(ltheta)]).T
                    rvector = th.stack([th.sin(rtheta)*th.cos(rphi), th.sin(rtheta)*th.sin(rphi), th.cos(rtheta)]).T
                    twoeyevector = left_eye_info - right_eye_info
                    distance = th.abs(th.tensordot(th.cross(lvector, rvector), twoeyevector, dims=([1],[1])))
                    loss_noncoplanar_iter = th.mean(distance / th.linalg.norm(twoeyevector, dim=1))
                if self.targettype == 'gazepoint':
                    prec_coord_iter= (self._gazepointprecision(lcoord.data.float(), target.data.float()) + self._gazepointprecision(rcoord.data.float(), target.data.float())) * 0.5
                elif self.targettype == 'gazedirection' and self.oneortwotarget == 2:
                    prec_coord_iter= (self._gazedirectionprecision(lcoord.data.float(), target[:,:2].data.float(), self.iseyediapnotation) + self._gazedirectionprecision(rcoord.data.float(), target[:,2:].data.float(), self.iseyediapnotation)) * 0.5
                else:
                    prec_coord_iter= (self._gazedirectionprecision(lcoord.data.float(), target.data.float(), self.iseyediapnotation) + self._gazedirectionprecision(rcoord.data.float(), target.data.float(), self.iseyediapnotation)) * 0.5
            if train:
                loss_coord_iter.backward()
                self.temps.valbase_solver.step()
                phase='DA'
            else:
                # accumulate meters
                loss_coord += loss_coord_iter.item() * face_image.shape[0]
                prec_coord += prec_coord_iter.item() * face_image.shape[0]
                loss_noncoplanar += loss_noncoplanar_iter.item() * face_image.shape[0]
                num_batches += 1            
                lcoord_cpu = lcoord.detach().cpu().numpy()
                target_cpu = target.detach().cpu().numpy()
                headposematrix_cpu = headposematrix.detach().cpu().numpy()
                if i == 0:
                    lcoord_cumu = [lcoord_cpu]
                    target_cumu = [target_cpu]
                    headposematrix_cumu = [headposematrix_cpu]
                else:
                    lcoord_cumu.append(lcoord_cpu)
                    target_cumu.append(target_cpu)
                    headposematrix_cumu.append(headposematrix_cpu)
                phase='Val'

            # logging
            infofmt = "[Fold{fold_no} "+phase+"][{temps.epoch}][{temps.iter}/{temps.num_val_iters}]\t" \
                      "data_time: {temps.val_data_time: .2f} batch_time: {temps.val_batch_time: .2f}\t" \
                      "prec_coord_val: {prec_coord: .4f}\t" \
                      "loss_coord_val: {loss_coord: .4f}\t" \
                      "loss_noncoplanar_val: {loss_noncoplanar: .4f}"
            infodict = dict(
                fold_no=self.fold_no,
                temps=self.temps,
                loss_coord=loss_coord_iter,
                prec_coord=prec_coord_iter,
                loss_noncoplanar=loss_noncoplanar_iter.item()
            )
            logger.info(infofmt.format(**infodict))

        if not train:
            ocoord_cumu = np.vstack(lcoord_cumu)
            otarget_cumu = np.vstack(target_cumu)
            lcoord_cumu = self._convheadpose_numpy(ocoord_cumu, headposematrix_cumu, self.iseyediapnotation)
            target_cumu = self._convheadpose_numpy(otarget_cumu, headposematrix_cumu, self.iseyediapnotation)
            preco = self._gazedirectionprecision(ocoord_cumu, otarget_cumu, self.iseyediapnotation, usenumpy=True).item()
            precn = self._gazedirectionprecision(lcoord_cumu, target_cumu, self.iseyediapnotation, usenumpy=True).item()
            precdelta = precn-preco
            logger.info("preco:%f, precn:%f, precdelta:%f"%(preco, precn, precdelta))
            #lcoord_cumu = ocoord_cumu
            #target_cumu = otarget_cumu
            prec_LSQF = self._fittarget(lcoord_cumu, target_cumu, -1, 1, 0, self.iseyediapnotation)
            tp = self.meters.loss_noncoplanar_train[self.temps.epoch]
            tlsqf = [tp['qk1'],tp['qb1'],tp['qk2'],tp['qb2']]
            prec_tLSQF = self._fittarget(lcoord_cumu, target_cumu, -1, 1, tlsqf, self.iseyediapnotation)
            qk1 = prec_LSQF[1]
            qb1 = prec_LSQF[2]*180/np.pi
            qk2 = prec_LSQF[3]
            qb2 = prec_LSQF[4]*180/np.pi
            prec_LSQ10 = self._fittarget(lcoord_cumu, target_cumu, 10, 100, 0, self.iseyediapnotation)
            prec_LSQ5 = self._fittarget(lcoord_cumu, target_cumu, 5, 100, 0, self.iseyediapnotation)
            prec_LSQ3 = self._fittarget(lcoord_cumu, target_cumu, 3, 100, 0, self.iseyediapnotation)
            prec_LSQ2 = self._fittarget(lcoord_cumu, target_cumu, 2, 100, 0, self.iseyediapnotation)
            prec_CF = self._fittarget(lcoord_cumu, target_cumu, -1, 1, 1, self.iseyediapnotation)
            tcf = [1,tp['cb1'],1,tp['cb2']]
            prec_tCF = self._fittarget(lcoord_cumu, target_cumu, -1, 1, tcf, self.iseyediapnotation)
            cb1 = prec_CF[2]*180/np.pi
            cb2 = prec_CF[4]*180/np.pi
            prec_C10 = self._fittarget(lcoord_cumu, target_cumu, 10, 100, 1, self.iseyediapnotation)
            prec_C5 = self._fittarget(lcoord_cumu, target_cumu, 5, 100, 1, self.iseyediapnotation)
            prec_C3 = self._fittarget(lcoord_cumu, target_cumu, 3, 100, 1, self.iseyediapnotation)
            prec_C2 = self._fittarget(lcoord_cumu, target_cumu, 2, 100, 1, self.iseyediapnotation)
            prec_C1 = self._fittarget(lcoord_cumu, target_cumu, 1, 100, 1, self.iseyediapnotation)
            resultdict = {
                'coord':lcoord_cumu*180/np.pi, 'target':target_cumu*180/np.pi, 'ocoord':ocoord_cumu*180/np.pi, 'otarget':otarget_cumu*180/np.pi, 
                'prec_LSQF':prec_LSQF[0],
                'prec_LSQ10':prec_LSQ10[0],
                'prec_LSQ5':prec_LSQ5[0],
                'prec_LSQ3':prec_LSQ3[0],
                'prec_LSQ2':prec_LSQ2[0],
                'prec_CF':prec_CF[0],
                'prec_C10':prec_C10[0],
                'prec_C5':prec_C5[0],
                'prec_C3':prec_C3[0],
                'prec_C2':prec_C2[0],
                'prec_C1':prec_C1[0],
                'qk1':qk1,
                'qb1':qb1,
                'qk2':qk2,
                'qb2':qb2,
                'cb1':cb1,
                'cb2':cb2,
                'prec_tLSQF':prec_tLSQF[0],
                'prec_tCF':prec_tCF[0]
            }
            logger.info(json.dumps(resultdict, default=ignorenumpy))
            # record meters
            epoch = self.temps.epoch
            self.meters.loss_coord_val[epoch] = loss_coord / len(self.temps.val_loader.dataset)
            self.meters.prec_coord_val[epoch] = prec_coord / len(self.temps.val_loader.dataset)

            self.meters.loss_noncoplanar_val[epoch] = resultdict#loss_noncoplanar / len(self.temps.val_loader.dataset)

    @staticmethod
    def _convheadpose_numpy(coords, headposematrixs, iseyediapnotation):
        if type(coords) == list:
            coords = np.vstack(coords)
        if type(headposematrixs) == list:
            headposematrixs = np.vstack(headposematrixs)
        newcoords = []
        for i in range(len(coords)):
            coord = coords[i]
            headpose = headposematrixs[i]
            if iseyediapnotation:
                coord3d = GazeTrainer._gazedirectionto3dvec_eyediapnota_numpy(coord)
            else:
                coord3d = GazeTrainer._gazedirectionto3dvec_default_numpy(coord)
            coord3d_wrthead = np.matmul(np.linalg.inv(headpose), coord3d)
            
            if iseyediapnotation:
                coord_wrthead = GazeTrainer._3dvectogazedirection_eyediapnota_numpy(coord3d_wrthead)
            else:
                coord_wrthead = GazeTrainer._3dvectogazedirection_default_numpy(coord3d_wrthead)
            newcoords.append(coord_wrthead)
        newcoords = np.vstack(newcoords)
        return newcoords
    
    @staticmethod
    def _convheadpose_torch(coords, headposematrixs, iseyediapnotation, hflip=False):
        if type(coords) == list:
            coords = th.vstack(coords)
        if type(headposematrixs) == list:
            headposematrixs = th.vstack(headposematrixs)
        newcoords = []
        for i in range(len(coords)):
            coord = coords[i]
            headpose = headposematrixs[i]
            if iseyediapnotation:
                coord3d = GazeTrainer._gazedirectionto3dvec_eyediapnota_torch(coord)
            else:
                coord3d = GazeTrainer._gazedirectionto3dvec_default_torch(coord)
            if hflip:
                coord3d[0]*=-1
            coord3d_wrthead = th.matmul(th.linalg.inv(headpose), coord3d)

            if iseyediapnotation:
                coord_wrthead = GazeTrainer._3dvectogazedirection_eyediapnota_torch(coord3d_wrthead)
            else:
                coord_wrthead = GazeTrainer._3dvectogazedirection_default_torch(coord3d_wrthead)
            newcoords.append(coord_wrthead)
        newcoords = th.vstack(newcoords)
        return newcoords

    @staticmethod
    def _fittarget(coord, target, calibcount, numsamples, method, iseyediapnotation):
        #lcoord_cumu = np.vstack(coord)
        #target_cumu = np.vstack(target)
        if type(coord) == list:
            coord = np.vstack(coord)
        if type(target) == list:
            target = np.vstack(target)
        x1 = coord[:,0]
        x2 = coord[:,1]
        y1 = target[:,0]
        y2 = target[:,1]
        a1 = np.vstack([x1, np.ones(x1.shape[0])]).T
        a2 = np.vstack([x2, np.ones(x2.shape[0])]).T 
        preccumu = 0 
        if calibcount == -1:
            calibcount=x1.shape[0]      
        for _ in range(numsamples):
            idxs = np.random.choice(x1.shape[0], calibcount, replace=False)
            if method == 0:
                k1,b1 = np.linalg.lstsq(a1[idxs], y1[idxs], rcond=None)[0]
                k2,b2 = np.linalg.lstsq(a2[idxs], y2[idxs], rcond=None)[0]
            elif method == 1:
                b1 = np.mean(y1[idxs]-x1[idxs])
                b2 = np.mean(y2[idxs]-x2[idxs])
                k1 = 1
                k2 = 1
            else:
                k1,b1,k2,b2 = method
                b1 = b1*np.pi/180
                b2 = b2*np.pi/180
            pred_y1 = k1*x1 + b1
            pred_y2 = k2*x2 + b2                
            target_pred = np.vstack([pred_y1, pred_y2]).T
            prec = GazeTrainer._gazedirectionprecision(target_pred, target, iseyediapnotation, usenumpy=True).item()
            preccumu+=prec
        return preccumu/numsamples,k1,b1,k2,b2

    def _test_headpose(self):
        logger = self.temps.headpose_logger.getChild('val')
        refine_depth = self._prepare_model(self.models.refine_depth)
        depth_loss = self.models.depth_loss
        loss_depth, num_batchs = 0, 0
        device = th.device("cpu") if not self.is_cuda else th.device("cuda")
        for i, batch in enumerate(self.temps.val_loader):
            self.temps.iter = i
            # prepare data
            face_image = batch['face_image'].to(device)
            face_depth = batch['face_depth'].to(device)

            # measure data loading time
            self.temps.data_time = self._timeit()

            # forward
            with th.no_grad():
                head_pose, refined_depth = refine_depth(face_image, face_depth)
                loss_depth_iter = depth_loss(refined_depth, face_depth)
                depth_grid_gt = make_grid(face_depth).cpu()
                depth_grid_rf = make_grid(refined_depth).cpu()
                if self.visualize:
                    with vplt.set_draw(name='val_depth_groundtruth') as ax:
                        ax.imshow(depth_grid_gt.numpy().transpose((1, 2, 0)))
                    with vplt.set_draw(name='val_train_depth_refined') as ax:
                        ax.imshow(depth_grid_rf.numpy().transpose((1, 2, 0)))
                #save_image(depth_grid_gt, os.path.join(self.result_root, "val", "depth",
                #                                       f"ep{self.temps.epoch:02d}iter{i:04d}_gt.png"))
                #save_image(depth_grid_rf, os.path.join(self.result_root, "val", "depth",
                #                                       f"ep{self.temps.epoch:02d}iter{i:04d}_rf.png"))

            # accumulate meters
            loss_depth += loss_depth_iter.item()
            num_batchs += 1
            # logging
            infofmt = "[{temps.epoch}]\t" \
                      "loss_depth: {loss_depth:.4f}"
            infodict = dict(
                temps=self.temps,
                loss_depth=loss_depth_iter,
            )
            logger.info(infofmt.format(**infodict))

        # record meters
        epoch = self.temps.epoch
        self.meters.loss_depth_val[epoch] = loss_depth / num_batchs

    @staticmethod
    def _gazedirectionto3dvec_eyediapnota_numpy(gaze):
        gaze_gt = np.zeros([3])
        gaze_gt[0] = -np.cos(gaze[1]) * np.sin(gaze[0])
        gaze_gt[1] = -np.sin(gaze[1])
        gaze_gt[2] = -np.cos(gaze[1]) * np.cos(gaze[0])
        return gaze_gt
    
    @staticmethod
    def _3dvectogazedirection_eyediapnota_numpy(gaze):
        yaw = np.arctan2(-gaze[0], -gaze[2])
        pitch = np.arcsin(-gaze[1])
        vector2dof = np.array([yaw, pitch])
        return vector2dof
    
    @staticmethod
    def _3dvectogazedirection_eyediapnota_torch(gaze):
        yaw = th.arctan2(-gaze[0], -gaze[2])
        pitch = th.arcsin(-gaze[1])
        vector2dof = th.cat([yaw.unsqueeze(0), pitch.unsqueeze(0)])
        return vector2dof
    
    @staticmethod
    def _gazedirectionto3dvec_default_numpy(gaze):
        theta, phi = gaze
        vector3d = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
        return vector3d
    
    @staticmethod
    def _gazedirectionto3dvec_default_torch(gaze):
        theta, phi = gaze
        vector3d = th.cat([(th.sin(theta)*th.cos(phi)).unsqueeze(0), (th.sin(theta)*th.sin(phi)).unsqueeze(0), (th.cos(theta)).unsqueeze(0)])
        return vector3d
    
    @staticmethod
    def _3dvectogazedirection_default_numpy(gaze):
        phi = np.arctan2(gaze[1],gaze[0])
        theta = np.arccos(gaze[2]/np.linalg.norm(gaze))
        vector2dof = np.array([theta, phi])
        return vector2dof
    
    @staticmethod
    def _3dvectogazedirection_default_torch(gaze):
        phi = th.arctan2(gaze[1],gaze[0])
        theta = th.arccos(gaze[2]/th.linalg.norm(gaze))
        vector2dof = th.cat([theta.unsqueeze(0), phi.unsqueeze(0)])
        return vector2dof
    
    @staticmethod
    def _gazedirectionto3dvec_eyediapnota_torch(gaze):
        #gaze_gt = th.zeros([3], requires_grad=True)
        x = -th.cos(gaze[1]) * th.sin(gaze[0])
        y = -th.sin(gaze[1])
        z = -th.cos(gaze[1]) * th.cos(gaze[0])
        gaze_gt = th.cat([x.unsqueeze(0),y.unsqueeze(0),z.unsqueeze(0)])
        return gaze_gt

    @staticmethod
    def _3dvecangulardistance_deg_numpy(gaze, label):
        total = np.sum(gaze * label)
        return np.arccos(min(total/(np.linalg.norm(gaze)* np.linalg.norm(label)), 0.9999999))*180/np.pi
    
    @staticmethod
    def _3dvecangulardistance_deg_torch(gaze, label):
        total = th.sum(gaze * label)
        return th.arccos(total/(th.linalg.norm(gaze)* th.linalg.norm(label)))*180/th.pi

    @staticmethod
    # Note that for float32 precision inputs, the numpy method is more accurate than the torch method, especially in train dataset where the target and out have little differences on some data.
    def _gazedirectionprecision(out, target, eyediapnotation=False, expand=False, usenumpy=False):
        if eyediapnotation:
            out = out.transpose(1, 0)
            target = target.transpose(1, 0)
            if not usenumpy:
                out = out.type(th.float64)
                target = target.type(th.float64)
                accarr = th.acos(th.minimum(th.cos(out[1])*th.cos(target[1])*th.cos(target[0]-out[0])+th.sin(out[1])*th.sin(target[1]),th.tensor([0.9999999], device=out.device)))
                if expand:
                    return th.rad2deg(accarr).numpy().astype(np.float32)
                return th.rad2deg(th.nanmean(accarr)).type(th.float32)
            else:
                out = out.astype(np.float64)
                target = target.astype(np.float64)
                accarr = np.arccos(np.minimum(np.cos(out[1])*np.cos(target[1])*np.cos(target[0]-out[0])+np.sin(out[1])*np.sin(target[1]),0.9999999))
                if expand:
                    return np.rad2deg(accarr).astype(np.float32)
                return np.rad2deg(np.nanmean(accarr)).astype(np.float32)
            '''
            accs = 0
            count = 0
            accarr = []
            for k, gaze in enumerate(out):
                gaze = gaze.cpu().detach().numpy()
                count += 1
                acc= GazeTrainer._3dvecangulardistance_deg_numpy(GazeTrainer._gazedirectionto3dvec_eyediapnota_numpy(gaze), GazeTrainer._gazedirectionto3dvec_eyediapnota_numpy(target.cpu().numpy()[k]))
                accarr.append(acc)
            #accs = accs / count
            accarr = np.array(accarr)
            if expand:
                return accarr
            else:
                return accarr.mean()
            '''
        else:
            out = out.transpose(1, 0)
            target = target.transpose(1, 0)            
            if not usenumpy:
                out = out.type(th.float64)
                target = target.type(th.float64)
                accarr = th.acos(th.minimum(th.sin(out[0])*th.sin(target[0])*th.cos(target[1]-out[1])+th.cos(out[0])*th.cos(target[0]),th.tensor([0.9999999], device=out.device)))
                if expand:
                    return th.rad2deg(accarr).numpy().astype(np.float32)
                return th.rad2deg(th.nanmean(accarr)).type(th.float32)
            else:
                out = out.astype(np.float64)
                target = target.astype(np.float64)
                accarr = np.arccos(np.minimum(np.sin(out[0])*np.sin(target[0])*np.cos(target[1]-out[1])+np.cos(out[0])*np.cos(target[0]),0.9999999))
                if expand:
                    return np.rad2deg(accarr).astype(np.float32)
                return np.rad2deg(np.nanmean(accarr)).astype(np.float32)
        
    @staticmethod
    def _gazeangularloss(out, target, eyediapnotation=False):
        if eyediapnotation:
            theloss = 0
            count = 0
            for k, gaze in enumerate(out):
                count += 1
                loss = GazeTrainer._3dvecangulardistance_deg_torch(GazeTrainer._gazedirectionto3dvec_eyediapnota_torch(gaze), GazeTrainer._gazedirectionto3dvec_eyediapnota_torch(target[k]))
                if k==0:
                    theloss = loss
                else:
                    theloss+=loss
            theloss = theloss / count
            return theloss
        else:
            out = out.transpose(1, 0)
            target = target.transpose(1, 0)
            return th.mean(th.acos(th.sin(out[0])*th.sin(target[0])*th.cos(target[1]-out[1])+th.cos(out[0])*th.cos(target[0])))
    
    @staticmethod
    def _gazel2loss(out, target):
        sigma = 3
        miu = 3
        theloss = 0
        count = 0
        for k, gaze in enumerate(out):
            count += 1
            tgt = target[k] + th.normal((miu/180)*th.pi,(sigma/180)*th.pi,(2,),device=out.device)
            loss = F.mse_loss(gaze,tgt)
            #angprecdeg = th.rad2deg(th.acos(th.sin(gaze[0])*th.sin(target[k][0])*th.cos(target[k][1]-gaze[1])+th.cos(gaze[0])*th.cos(target[k][0])))
            if k==0:
                theloss = loss
            else:
                '''
                if angprecdeg > miu:
                    factor = 0
                else:
                    factor = (1/(sigma*np.sqrt(2*th.pi))) * np.power(np.e,-(angprecdeg.item()-miu)*(angprecdeg.item()-miu)/(2*sigma*sigma))
                factor = 1-factor
                '''
                theloss+=loss
        theloss = theloss / count
        return theloss
                         
    @staticmethod
    def _gazepointprecision(out, target):
        return th.mean(th.sqrt(th.sum((out-target)**2,1)))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='<%(name)s:%(levelname)s> %(message)s')
    # trainer = GazeTrainer(
    #     exp_name="gaze_aaai_refine_headpose"
    # )
    # trainer.train_headpose(10, lr=2e-1)
    fire.Fire(GazeTrainer)
