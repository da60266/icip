import torch
#import torchvision.models as models
#from torchsummary import summary

import torch.nn as nn
from .partial.wtconv import WTConv2d
from .partial.dct_modified import LinearDCT, apply_linear_2d
from .partial.wmg import WMG

class GANet(nn.Module):
    def __init__(self):
        super(GANet, self).__init__()
        # 36 * 60
        # 18 * 30
        # 9  * 15
        #eyeconv = models.vgg16(pretrained=True).features
        # self.eyeStreamConv.load_state_dict(vgg16.features[0:9].state_dict())
        # self.faceStreamPretrainedConv.load_state_dict(vgg16.features[0:15].state_dict())
    
        # Eye Stream, is composed of Conv DConv and FC layers. 
        self.eyeStreamConv = nn.Sequential(
            # 64*96
            WTConv2d(1,   64,  3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            WTConv2d(64,  64,  3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # 32*48
            WTConv2d(64,  128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            WTConv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.eyeStreamDConv = nn.Sequential(
            WTConv2d(128, 64, 1, 1), 
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 32*48
            WTConv2d(64, 64, 3, 1,  1,dilation=(3,3)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # dilation-2*2 = 28*44
            WTConv2d(64, 64, 3, 1,  1,dilation=(3,3)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            #nn.MaxPool2d(2,2),# non-dilation output 16*24

            # dilation-2*2 = 24*40
            WTConv2d(64, 128, 3, 1, 1,dilation=(5,5)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # dilation-2*4 = 16*32
            WTConv2d(128, 128, 3, 1, 1,dilation=(5,9)),
            # dilation-2*4,2*8 = 8*16
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            #nn.MaxPool2d(2,2)# non-dilation output 8*12
        )

        self.eyeStreamFC = nn.Sequential(
            nn.Linear(128*8*16, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        self.eyedepthStreamConv = nn.Sequential(
            # 64*96
            WTConv2d(3,   64,  3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            WTConv2d(64,  64,  3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # 32*48
            WTConv2d(64,  128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            WTConv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.eyedepthStreamDConv = nn.Sequential(
            WTConv2d(128, 64, 1, 1), 
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 30*46+2
            WTConv2d(64, 64, 3, 1,  1, dilation=(3,3)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 28*44 26*42+2
            WTConv2d(64, 64, 3, 1,  1, dilation=(3,3)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            #nn.MaxPool2d(2,2),# non-dilation output 16*24

            # 22*38 22*38+2
            WTConv2d(64, 128, 3, 1, 1, dilation=(5,5)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # 14*28 14*30+2
            WTConv2d(128, 128, 3, 1, 1, dilation=(5,9)),
            # 4*6 6*14+2
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            #nn.MaxPool2d(2,2)# non-dilation output 8*12
        )

        self.eyedepthStreamFC = nn.Sequential(
            nn.Linear(128*8*16, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
       
        # Face Stream, is composed of Conv and FC layers.
        #faceconv = models.vgg16(pretrained=True).features
        self.faceStreamPretrainedConv = nn.Sequential(
            # 96*96
            WTConv2d(1, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            WTConv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(2, 2),

            # 48*48
            WTConv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            WTConv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),  
            nn.MaxPool2d(2, 2),
            
            # 24*24
            WTConv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            WTConv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

  
        self.faceStreamConv = nn.Sequential(
            WTConv2d(256, 64, 1),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 12*12
            WTConv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            WTConv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            # 6*6
        )

        self.faceStreamFC = nn.Sequential(
            nn.Linear(128 * 128 * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        self.totalFC = nn.Sequential(
            nn.Linear(256+256+32, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )

        eyeh = 64
        eyew = 96
        faceh = 96
        facew = 96
        eyefactor = eyew/eyeh
        eye_n_freqchans = round(((eyeh**2)+((eyew/eyefactor)**2))**0.5)+1-1#round(((h-1)**2+(w-1)**2)**0.5)+1
        face_n_freqchans = round(((faceh**2)+((facew)**2))**0.5)
        self.eyepercentfreq = round(eye_n_freqchans*0.25)
        self.facepercentfreq = round(face_n_freqchans*0.25)
        self.leftmask = torch.zeros(1,eye_n_freqchans,eyeh,eyew)
        self.rightmask = torch.zeros(1,eye_n_freqchans,eyeh,eyew)
        self.facemask = torch.zeros(1,face_n_freqchans,faceh,facew)
        for i in range(eyeh):
            for j in range(eyew):
                self.leftmask[0,int(round((((i+1)**2)+(((j+1)/eyefactor)**2))**0.5))-1,i,j]=1
                self.rightmask[0,int(round((((i+1)**2)+(((j+1)/eyefactor)**2))**0.5))-1,i,j]=1
        for i in range(faceh):
            for j in range(facew):
                self.facemask[0,int(round((((i+1)**2)+(((j+1))**2))**0.5))-1,i,j]=1

        self.leftwmg = WMG(1, 3, eye_n_freqchans,64,96)
        #self.rightwmg = WMG(1, 3, eye_n_freqchans,64,96)
        #self.facewmg = WMG(1, 1, face_n_freqchans,96,96)
        self.eyelinear_dct_h = LinearDCT(eyeh, 'dct')
        self.eyelinear_dct_w = LinearDCT(eyew, 'dct')
        self.eyelinear_idct_h = LinearDCT(eyeh, 'idct')
        self.eyelinear_idct_w = LinearDCT(eyew, 'idct')
        self.facelinear_dct = LinearDCT(faceh, 'dct')
        self.facelinear_idct = LinearDCT(faceh, 'idct')
        self.eyedepth_eyeappearance_attn = nn.MultiheadAttention(embed_dim=128, num_heads=1, kdim=128, vdim=128, batch_first=True)
        self.eye_face_attn = nn.MultiheadAttention(embed_dim=128*2, num_heads=1, kdim=36, vdim=36, batch_first=True)
        # self.lfi = None
        # self.rfi = None
        # self.lfreq = None
        # self.lficube = None
        # self.lfd = None
        # self.lef = None
        # self.ledf = None
        # self.ref = None
        # self.redf = None
        # self.dctcoef = None
        # self.ff = None
        # self.dctcoefr = None
        # self.rfd = None
        # self.leftwm = None
        # self.leftwmcube = None

    def forward(self, leftimage, rightimage, faceimage, leftnormal, rightnormal, leftdepth, rightdepth, facedepth, facenormal, facemesh):
        # Get face feature
        #print(leftimage.device)
        self.leftmask = self.leftmask.to(leftimage.device)
        self.rightmask = self.rightmask.to(leftimage.device)
        self.facemask = self.facemask.to(leftimage.device)
        #faceimage = faceimage[:,0:1,:,:]
        #ffreqcube = apply_linear_2d(faceimage, self.facelinear_dct, self.facelinear_dct)
        '''
        ffreqcube = self.facelinear_dct(self.facelinear_dct(faceimage).transpose(-1, -2)).transpose(-1,-2)
        #print(ffreqcube.device)
        #print(self.facemask.device)
        ffreqcube = ffreqcube*self.facemask
        #ffreqcube = apply_linear_2d(ffreqcube, self.facelinear_idct, self.facelinear_idct)
        ffreqcube = self.facelinear_idct(self.facelinear_idct(ffreqcube).transpose(-1, -2)).transpose(-1,-2)
        '''
        '''
        if self.training:
            frandidx = torch.randperm(faceimage.shape[0], device=ffreqcube.device, requires_grad=False)
            ffreqcube[:,-self.facepercentfreq:,:,:] = ffreqcube[frandidx,-self.facepercentfreq:,:,:]
        '''
        '''
        #ffiold,ffd,fwmcube = self.facewmg(faceimage, faceimage)
        #ffi = ffreqcube*fwmcube
        ffi = torch.sum(ffreqcube,dim=1,keepdim=True)
        #ffi = faceimage
        '''
        #faceinput = torch.cat([faceimage, facenormal], dim=1)
        faceinput = faceimage
        #print(faceinput.shape)
        faceFeatureMap = self.faceStreamPretrainedConv(faceinput)
        faceFeatureMap = self.faceStreamConv(faceFeatureMap)
        ##self.ff = faceFeatureMap
        faceFeature = torch.flatten(faceFeatureMap, start_dim=2) # batch*64*6*6 -> batch*64*36
        '''
        faceFeature = torch.flatten(faceFeatureMap, start_dim=1)
        faceFeature = self.faceStreamFC(faceFeature)
        '''

        # Get left feature
        #leftimage = leftimage[:,0:1,:,:]
        #lfreqcube = apply_linear_2d(leftimage, self.eyelinear_dct_w, self.eyelinear_dct_h)
        lfreqcube = self.eyelinear_dct_h(self.eyelinear_dct_w(leftimage).transpose(-1, -2)).transpose(-1,-2)
        ##self.dctcoef = lfreqcube
        lfreqcube = lfreqcube*self.leftmask
        #lfreqcube = apply_linear_2d(lfreqcube, self.eyelinear_idct_w, self.eyelinear_idct_h)
        lfreqcube = self.eyelinear_idct_h(self.eyelinear_idct_w(lfreqcube).transpose(-1, -2)).transpose(-1,-2)
        '''
        if self.training:
            lrandidx = torch.randperm(leftimage.shape[0], device=lfreqcube.device, requires_grad=False)
            lfreqcube[:,-self.eyepercentfreq:,:,:] = lfreqcube[lrandidx,-self.eyepercentfreq:,:,:]
        '''
        lfiold,lfd,lwmcube = self.leftwmg(leftimage, leftnormal)
        lfi = lfreqcube*lwmcube
        lfisum = torch.sum(lfi,dim=1,keepdim=True)
        #lfi = leftimage
        #lfd = leftdepth
        ##self.lfi = lfisum
        ##self.lfreq = lfreqcube
        ##self.lfd = lfd
        ##self.lficube = lfi
        ##self.leftwm = self.leftwmg.WM
        ##self.leftwmcube = lwmcube
        leftEyeFeature = self.eyeStreamConv(lfisum)
        leftEyeDepthFeature = self.eyedepthStreamConv(lfd)
        leftEyeFeature = leftEyeFeature + leftEyeDepthFeature
        leftEyeFeature = self.eyeStreamDConv(leftEyeFeature)
        leftEyeDepthFeature = self.eyedepthStreamDConv(leftEyeDepthFeature)
        '''
        leftEyeFeature = leftEyeFeature + leftEyeDepthFeature
        #print(leftEyeFeature.shape)
        leftEyeFeature = torch.flatten(leftEyeFeature, start_dim=1)
        '''
        ##self.lef = leftEyeFeature
        ##self.ledf = leftEyeDepthFeature        
        leftEyeFeature = torch.flatten(leftEyeFeature, start_dim=2)
        leftEyeDepthFeature = torch.flatten(leftEyeDepthFeature, start_dim=2) # batch*128*8*12 -> batch*128*96
        leftEyeFeatureFusion = self.eyedepth_eyeappearance_attn(leftEyeDepthFeature, leftEyeFeature, leftEyeFeature, need_weights=False)
        #print(type(leftEyeFeatureFusion[0]))
        #print(type(leftEyeFeatureFusion[1]))
        leftEyeFeatureFusion = leftEyeFeatureFusion[0] + leftEyeFeature
        '''
        leftEyeFeature = self.eyeStreamFC(leftEyeFeature)
        '''
 
        # Get Right feature
        #rightimage = rightimage[:,0:1,:,:]
        #rfreqcube = apply_linear_2d(rightimage, self.eyelinear_dct_w, self.eyelinear_dct_h)
        rfreqcube = self.eyelinear_dct_h(self.eyelinear_dct_w(rightimage).transpose(-1, -2)).transpose(-1,-2)
        ##self.dctcoefr = rfreqcube
        rfreqcube = rfreqcube*self.rightmask
        #rfreqcube = apply_linear_2d(rfreqcube, self.eyelinear_idct_w, self.eyelinear_idct_h)
        rfreqcube = self.eyelinear_idct_h(self.eyelinear_idct_w(rfreqcube).transpose(-1, -2)).transpose(-1,-2)
        '''
        if self.training:
            rrandidx = torch.randperm(rightimage.shape[0], device=rfreqcube.device, requires_grad=False)
            rfreqcube[:,-self.eyepercentfreq:,:,:] = rfreqcube[rrandidx,-self.eyepercentfreq:,:,:]
        '''
        rfiold,rfd,rwmcube = self.leftwmg(rightimage, rightnormal)
        rfi = rfreqcube*rwmcube
        rfisum = torch.sum(rfi,dim=1,keepdim=True)
        ##self.rfi = rfisum
        ##self.rfd = rfd
        #rfi = rightimage
        #rfd = rightdepth
        rightEyeFeature = self.eyeStreamConv(rfisum)
        rightEyeDepthFeature = self.eyedepthStreamConv(rfd)
        rightEyeFeature = rightEyeFeature + rightEyeDepthFeature
        rightEyeFeature = self.eyeStreamDConv(rightEyeFeature)
        rightEyeDepthFeature = self.eyedepthStreamDConv(rightEyeDepthFeature)
        '''
        rightEyeFeature = rightEyeFeature + rightEyeDepthFeature
        rightEyeFeature = torch.flatten(rightEyeFeature, start_dim=1)
        '''
        ##self.ref = rightEyeFeature
        ##self.redf = rightEyeDepthFeature        
        rightEyeFeature = torch.flatten(rightEyeFeature, start_dim=2)
        rightEyeDepthFeature = torch.flatten(rightEyeDepthFeature, start_dim=2) # batch*128*8*12 -> batch*128*96
        rightEyeFeatureFusion = self.eyedepth_eyeappearance_attn(rightEyeDepthFeature, rightEyeFeature, rightEyeFeature, need_weights=False)
        #print(rightEyeFeatureFusion.shape)
        rightEyeFeatureFusion = rightEyeFeatureFusion[0] + rightEyeFeature
        '''
        rightEyeFeature = self.eyeStreamFC(rightEyeFeature)
        '''
 
        '''
        features = torch.cat((faceFeature, leftEyeFeature, rightEyeFeature), 1)
        '''
        eyeFeature = torch.cat((leftEyeFeatureFusion, rightEyeFeatureFusion), 2)
        miscFeatureFusion = self.eye_face_attn(eyeFeature, faceFeature, faceFeature, need_weights=False) # batch*128*256

        leftEyeFeatureFusionFlatten = torch.flatten(leftEyeFeatureFusion, start_dim=1)
        leftEyeFeatureFusionFlatten = self.eyeStreamFC(leftEyeFeatureFusionFlatten)
        rightEyeFeatureFusionFlatten = torch.flatten(rightEyeFeatureFusion, start_dim=1)
        rightEyeFeatureFusionFlatten = self.eyeStreamFC(rightEyeFeatureFusionFlatten)
        miscFeatureFusionFlatten = torch.flatten(miscFeatureFusion[0], start_dim=1)
        miscFeatureFusionFlatten = self.faceStreamFC(miscFeatureFusionFlatten)
        features = torch.cat([leftEyeFeatureFusionFlatten, rightEyeFeatureFusionFlatten, miscFeatureFusionFlatten], 1)

        gaze = self.totalFC(features)

        return gaze,gaze

if __name__ == '__main__':
    from thop import profile
    from torchsummaryX import summary
    m = GANet()
    #m.to("cuda")
    '''
    feature = {"face":torch.zeros(64, 3, 96, 96).to("cuda"),
                "left":torch.zeros(64, 3, 64,96).to("cuda"),
                "right":torch.zeros(64, 3, 64,96).to("cuda")
              }
    '''
    '''
    a = m(torch.zeros(3, 3, 64,96),torch.zeros(3, 3, 64,96),
    torch.zeros(3, 3, 64,96),torch.zeros(3, 3, 64,96),
    torch.zeros(3, 3, 96,96))
    print(a)
    '''
    #summary(m, [(1, 64,96),(1,64,96),(1,96,96),(3,64,96),(3,64,96)],3,'cpu')
    m = m.eval()
    summary(m, torch.zeros(1,1,64,96),rightimage=torch.zeros(1,1,64,96),faceimage=torch.zeros(1,1,96,96),rightdepth=torch.zeros(1,3,64,96),leftdepth=torch.zeros(1,3,64,96))
    info=profile(m, (torch.zeros(1,1,64,96),torch.zeros(1,1,64,96),torch.zeros(1,1,96,96),torch.zeros(1,3,64,96),torch.zeros(1,3,64,96)),ret_layer_info=True)
    print(info)