import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
import torch
from torch import nn

class crazyleg(nn.Module):
    def __init__(self,  hip0 = -10.0-80+90,knee0 = -80.0 +90, dt_ = 0.002025 ):
        super().__init__()
        dev = "cuda:0" if torch.cuda.is_available() else "cpu"
        device_ = torch.device(dev)
        #torch.autograd.set_detect_anomaly(True)
        self.dev = device_
        self.sigmoid_factor1 = 150#1000
        self.sigmoid_factor2 = 150
        self.g = 9.81
        self.hhip = 0.1#1
        self.lpelv = 0.16

        self.mfem = 9.1
        self.sfem = 0.15
        self.lfem = 0.39
        self.mtib = 3.6
        self.stib = 0.2
        self.ltib = 0.43
        self.sfoot = 0.06
        self.mfoot = 1.5
        self.lfoot = 0.07
        self.rknee = 0.03
        self.sgm = 0.13
        self.betast = math.pi/6
        self.d = torch.sqrt(self.lfem**2 + self.rknee**2 + 2 * self.lfem * self.rknee * torch.cos(torch.tensor(self.betast + 80* 3.14/180))) ##
        self.alpha = torch.acos((-self.rknee**2 + self.d**2 +self.lfem**2)/(2*self.d*self.lfem))
        self.hlow = 0.07
        self.rhip = 0.035
        self.hsup = torch.tensor(0.03,requires_grad=False, device = device_)
        self.sbf = 0.2
        self.betabf = math.pi/6
        self.Fmax = torch.tensor([[1417.0] ,[1176.0], [730.0],[1580.0],[3120.0],[470.0], [1513.0]], device = device_, requires_grad = True)
        self.alphas = torch.tensor([[0.14],[0.0], [0.09],[0.0],[0.05],[0.4], [0.3]], device = device_, requires_grad = True)
        self.Lopt = torch.tensor([0.102,0.158,0.112,0.109,0.104,0.177,0.1], device = device_, requires_grad = True)

        self.theta_hip0 = torch.tensor(30.0,requires_grad=False, device = device_)#np.degrees(np.arcsin(hp/lfem))
        self.hp = self.lfem* torch.sin(torch.deg2rad(self.theta_hip0))
        self.theta_knee0 = -(self.theta_hip0 + torch.rad2deg(torch.arcsin(self.hp/self.ltib)))
        theta0 = torch.tensor([[torch.deg2rad(self.theta_hip0)],[torch.deg2rad(self.theta_knee0)]],requires_grad=False, device = device_)  # these are actually theta, not q
        self.theta0 = theta0
        self.hipstart = hip0 # this is q
        self.kneestart = knee0 # this is q
        #self.theta_ = torch.tensor([[math.radians(hip0)],[math.radians(knee0)]],requires_grad=True, device = device_)
        self.theta_ = torch.tensor([[self.theta0[0][0] + math.pi/2],[torch.abs(self.theta0[0][0]) - torch.abs(self.theta0[1][0])+ math.pi/2 ]],requires_grad=True, device = device_)
        self.q0 = self.theta_
        self.theta_dot = torch.tensor([[0.0],[0.0]],requires_grad=True, device = device_)
        self.A_ = torch.zeros((7, 1),requires_grad=True, device = device_ )
        #self.thetas= torch.tensor([[math.radians(hip0)],[math.radians(knee0)]],requires_grad=True, device = device_)
        self.thetas = torch.tensor([[self.theta0[0][0] + math.pi/2],[torch.abs(self.theta0[0][0]) - torch.abs(self.theta0[1][0])+ math.pi/2 ]],requires_grad=True, device = device_)
        self.theta_dots = torch.tensor([[0.0],[0.0]],requires_grad=True, device = device_)
        self.As = torch.zeros((7, 1),requires_grad=True, device = device_)
        self.dt = dt_
        self.tact = [0.03, 0.12]#[0.01, 0.04]
        #q = np.array([[theta[0][0] + math.pi/2],[np.abs(theta[0][0]) - np.abs(theta[1][0])+ math.pi/2 ]])

    def get_theta0 (self):
        return self.theta0

    def reset(self):
        #self.theta_ = torch.tensor([[math.radians(self.hipstart)],[math.radians(self.kneestart)]],requires_grad=True, device = self.dev)
        self.theta_ = torch.tensor([[self.theta0[0][0] + math.pi/2],[torch.abs(self.theta0[0][0]) - torch.abs(self.theta0[1][0])+ math.pi/2 ]],requires_grad=True, device = self.dev)
        self.theta_dot = torch.tensor([[0.0],[0.0]],requires_grad=True, device = self.dev)
        self.A_ = torch.zeros((7, 1),requires_grad=True, device = self.dev,dtype=torch.float32)
        #self.thetas = torch.tensor([[math.radians(self.hipstart)],[math.radians(self.kneestart)]],requires_grad=True, device = self.dev)
        self.thetas = torch.tensor([[self.theta0[0][0] + math.pi/2],[torch.abs(self.theta0[0][0]) - torch.abs(self.theta0[1][0])+ math.pi/2 ]],requires_grad=True, device = self.dev)
        self.theta_dots = torch.tensor([[0.0],[0.0]],requires_grad=True, device = self.dev)
        self.As = torch.zeros((7, 1),requires_grad=True, device = self.dev,dtype=torch.float32)

    def G(self, q):
        g1 = -(self.mfem*self.sfem + (self.mtib + self.mfoot)*self.lfem)*self.g*torch.sin(q[0][0])
        g2 = -(self.mtib*self.stib+ self.mfoot*self.ltib)*self.g*torch.sin(q[1][0])
        return torch.tensor([[g1], [g2]],requires_grad=True , device =q.device )

    def C(self,q, q_dot):
        c12 = -(self.mtib*self.lfem*self.stib + self.mfoot*self.lfem*self.ltib)*torch.sin(q[0][0] - q[1][0])*(q_dot[1][0])
        c21 = (self.mtib*self.lfem*self.stib + self.mfoot*self.lfem*self.ltib)*torch.sin(q[0][0] - q[1][0])*(q_dot[0][0])
        return torch.tensor([[0 , c12], [c21 , 0]],requires_grad=True , device =q.device )

    def Rtot(self,q):
        RVlat = 0.05*torch.cos(0.8*(q[1][0]-math.radians(100)))#math.radians(80)))
        RBF = -0.038*torch.cos(q[1][0]-math.radians(65))#math.radians(35))
        RRF_hip = 0.053*torch.cos(1*(q[0][0]-math.radians(127)))
        RRF_knee = 0.053*torch.cos(0.55*(q[1][0]-math.radians(100)))#math.radians(70)))
        RST_hip = -0.07*torch.cos(q[0][0]-math.radians(122))
        RST_knee = -0.04*torch.cos(q[1][0]-math.radians(85))#math.radians(55))
        RIl = 0.045*torch.cos(0.6*(q[0][0]-math.radians(140)))
        RGM = -0.06*torch.cos(0.75*(q[0][0]-math.radians(90)))
        RMG = -0.038*torch.cos(0.7*(q[1][0]-math.radians(50)))
        return torch.tensor([[RIl, RGM, RRF_hip, RST_hip,0,0,0], [0,0,RRF_knee, RST_knee,RVlat,RBF,RMG]],requires_grad=True , device =q.device )

    def Rtot_thelen(self,q):
        RVlat = 0.048*torch.cos(0.7*(q[1][0]-math.radians(110)))#math.radians(80)))
        RBF = -0.035*torch.cos(q[1][0]-math.radians(72))#math.radians(35))
        RRF_hip = 0.053*torch.cos(1*(q[0][0]-math.radians(127)))
        RRF_knee = 0.052*torch.cos(0.9*(q[1][0]-math.radians(100)))#math.radians(70)))
        RST_hip = -0.06*torch.cos(q[0][0]-math.radians(122))
        RST_knee = -0.042*torch.cos(0.9*(q[1][0]-math.radians(75)))#math.radians(55))
        RIl = 0.045*torch.cos(0.6*(q[0][0]-math.radians(140)))
        RGM = -0.058*torch.cos(0.65*(q[0][0]-math.radians(70)))
        RMG = -0.038*torch.cos(0.7*(q[1][0]-math.radians(50)))
        return torch.tensor([[RIl, RGM, RRF_hip, RST_hip,0,0,0], [0,0,RRF_knee, RST_knee,RVlat,RBF,RMG]],requires_grad=True , device =q.device )



    def M(self,q):
        m11 = 0.112007 + self.mfem*(self.sfem**2) + (self.mtib + self.mfoot)*self.lfem**2
        m12 = (self.mtib*self.lfem*self.stib + self.mfoot*self.lfem*self.ltib)*torch.cos(q[0][0]-q[1][0])
        m21 = m12
        m22 = 0.0591636 + self.mtib*(self.stib**2) + self.mfoot*(self.ltib**2)
        return torch.tensor([[m11 , m12], [m21 , m22]],requires_grad=True, device = q.device)

    def A_dot(self, A, U):
        #tact =0.05#0.5# 0.01
        #if len(self.tact)== 1:
        return (U-A)/self.tact[1]
        #else:
        #    inter = 0.5+1.5*A
        #    t_act = self.tact[0] ##0.01
        #    tdeact = self.tact[1] ##0.04
        #    tau = t_act*inter + (U<A)*((tdeact /inter) - t_act*inter)
        #return (U-A)/tau

    def sigmoid_lst(self,length):  ##
        return (1-0.05)/(1+torch.exp(-100*(length-0.07)))

    def sigmoid_lstmax(self,lst, factor=100, b=0.125):
        return (1)/(1+torch.exp(-factor*(lst-b)))

    def L_tot(self,q_):
        q =  q_
        LIl = 0.094 - 0.035*(q[0][0] - math.pi/2)
        sigLil = self.sigmoid_lst(LIl)
        LIl = LIl*sigLil + (1-sigLil)*0.05
        LGM = 0.127 + 0.04*(q[0][0]- math.pi/2)
        sigGM = self.sigmoid_lst(LGM)
        LGM = LGM*sigGM + (1-sigGM)*0.05
        #LRF = 0.06 + torch.sqrt(torch.tensor(self.lfem**2 +self.hsup**2 + 2*self.lfem*self.hsup*torch.cos(q[0][0]))) - torch.sqrt(torch.tensor(self.lfem**2 +self.hsup**2)) - self.rknee*(q[1][0]-q[0][0])
        LRF = 0.06 + torch.sqrt(self.lfem**2 +self.hsup**2 + 2*self.lfem*self.hsup*torch.cos(q[0][0]))  - torch.sqrt(torch.tensor(self.lfem**2 +self.hsup**2)) - self.rknee*(q[1][0]-q[0][0])
        sigRF = self.sigmoid_lst(LRF)
        LRF = LRF*sigRF + (1-sigRF)*0.05
        LST = 0.055 + 0.05*(q[0][0]- math.pi/2) + self.rknee * (q[1][0] - q[0][0])
        sigST = self.sigmoid_lst(LST)
        LST = LST*sigST + (1-sigST)*0.05
        Lvlat = 0.046 + self.rknee*(q[0][0] - q[1][0])
        sigVLat = self.sigmoid_lst(Lvlat)
        Lvlat = Lvlat*sigVLat + (1-sigVLat)*0.05
        Lbf =  0.139 + self.rknee * (q[1][0] - q[0][0])
        Lbf = Lbf*self.sigmoid_lst(Lbf) + (1-self.sigmoid_lst(Lbf))*0.05
        LMG = 0.055 + self.rknee * (q[1][0] - q[0][0])
        LMG = LMG*self.sigmoid_lst(LMG) + (1-self.sigmoid_lst(LMG))*0.05
        return torch.tensor([LIl, LGM, LRF, LST, Lvlat, Lbf, LMG],requires_grad=True, device = q_.device)

    def sigmoid_lstmax(self,lst, factor=100, b=0.125):
        return (1)/(1+torch.exp(-factor*(lst-b)))


    def L_tot_thelen(self, q):
        # order : [LIl, LGM, LRF, LST, Lvlat, Lbf, LMG]
        self.Lopt = torch.tensor([0.117,0.156,0.075,0.068,0.097,0.110, 0.053])
        Lnorm_min = 0.5  ## SET BELOW IN L_tot_dot()
        Lnorm_max = 1.7
        factor = 10

        LIl = 0.125 - 0.035*(q[0][0] - math.pi/2)
        LGM = 0.159 + 0.04*(q[0][0]- math.pi/2)
        LRF = 0.069 + torch.sqrt(self.lfem**2 +self.hsup**2 + 2*self.lfem*self.hsup*torch.cos(q[0][0])) - torch.sqrt(self.lfem**2 +self.hsup**2) - 0.045*(q[1][0]-q[0][0])
        LST = 0.073 + 0.05*(q[0][0]- math.pi/2) + 0.035 * (q[1][0] - q[0][0])
        Lvlat = 0.098 + 0.04*(q[0][0] - q[1][0])
        Lbf =  0.152 + 0.03 * (q[1][0] - q[0][0])
        LMG = 0.06 + 0.02 * (q[1][0] - q[0][0])

        # sigmoid for min and max Lnorm
        Ltot = torch.tensor([LIl, LGM, LRF, LST, Lvlat, Lbf, LMG], requires_grad = True)
        vals = torch.zeros(np.shape(Ltot))
        for m in range(len(Ltot)):
            vals[m] = Ltot[m]*(self.sigmoid_lstmax(Ltot[m]/self.Lopt[m], factor=factor, b=Lnorm_min)-self.sigmoid_lstmax(Ltot[m]/self.Lopt[m], factor=factor, b=Lnorm_max)) + (1-self.sigmoid_lstmax(Ltot[m]/self.Lopt[m], factor=factor, b=Lnorm_min))*Lnorm_min*self.Lopt[m] + self.sigmoid_lstmax(Ltot[m]/self.Lopt[m], factor=factor, b=Lnorm_max)*Lnorm_max*self.Lopt[m]

        return vals.clone().requires_grad_(True)#.detach().requires_grad_(True)


    def L_tot_dot(self, q , q_dot,L):
        sigL = self.sigmoid_lst(L)
        LIldot = -self.rhip*q_dot[0][0]
        LIldot = sigL[0]*LIldot
        LGMdot = 0.04*q_dot[0][0]
        LGMdot = sigL[1]*LGMdot
        LRFdot = -2*self.lfem*self.hsup*torch.sin(q[0][0])*q_dot[0][0]/(2*torch.sqrt(self.lfem**2 + self.hsup**2 + 2*self.lfem*self.hsup*torch.cos(q[0][0]))) - self.rknee*(q_dot[1][0] - q_dot[0][0])
        LRFdot = sigL[2]*LRFdot
        LSTdot = 0.05*q_dot[0][0] + self.rknee * (q_dot[1][0] - q_dot[0][0])
        LSTdot = sigL[3]*LSTdot
        Lvlatdot = self.rknee * q_dot[0][0] - self.rknee * q_dot[1][0]
        Lvlatdot = sigL[4]*Lvlatdot
        Lbfdot = self.rknee * (q_dot[1][0] - q_dot[0][0])
        Lbfdot = sigL[5]*Lbfdot
        LMGdot = self.rknee * (q_dot[1][0] - q_dot[0][0])
        LMGdot = sigL[6]*LMGdot
        return torch.tensor([LIldot, LGMdot, LRFdot, LSTdot, Lvlatdot,Lbfdot,LMGdot],requires_grad=True , device = q.device)

    def L_tot_dot_thelen(self, q, q_dot, L):
        # order : [LIl, LGM, LRF, LST, Lvlat, Lbf, LMG]
        Lnorm_min = 0.5
        Lnorm_max = 1.7
        factor = 10

        LIldot = 0.035*(q_dot[0][0])
        LGMdot = 0.04*(q_dot[0][0])
        LRFdot = -2*self.lfem*self.hsup*torch.sin(q[0][0])*q_dot[0][0]/(2*torch.sqrt(self.lfem**2 + self.hsup**2 + 2*self.lfem*self.hsup*torch.cos(q[0][0]))) - 0.045*(q_dot[1][0] - q_dot[0][0])
        LSTdot = 0.05*q_dot[0][0] + 0.035 * (q_dot[1][0] - q_dot[0][0])
        Lvlatdot = 0.04 * q_dot[0][0] -self.rknee * q_dot[1][0]
        Lbfdot = 0.03 * (q_dot[1][0] - q_dot[0][0])
        LMGdot = 0.02 * (q_dot[1][0] - q_dot[0][0])

        # sigmoid for min and max Lnorm
        Ldot = torch.tensor([LIldot, LGMdot, LRFdot, LSTdot, Lvlatdot, Lbfdot, LMGdot], requires_grad = True)
        vals = torch.zeros(np.shape(Ldot))
        for m in range(len(Ldot)):
            vals[m] = Ldot[m]*(self.sigmoid_lstmax(L[m]/self.Lopt[m], factor=factor, b=Lnorm_min)-self.sigmoid_lstmax(L[m]/self.Lopt[m], factor=factor, b=Lnorm_max))

        return vals.clone().requires_grad_(True)#.detach().requires_grad_(True)


    def fl(self,l):
        return torch.unsqueeze(torch.exp(-((l-1)**2)/0.45),1) #.to(l.device)

    def fp(self , l):
        res = torch.zeros((len(l), 1), requires_grad = False, device = l.device)
        for k in range((len(l))):
            res[k][0] = torch.exp((5*(l[k]-1)/0.7)-1)/(torch.exp(torch.tensor(5.0))-1)
        return res.requires_grad_(True)

    def fvl_tot(self,l,ldot):
        result = torch.ones((len(l),1), device = l.device)
        for i in range (len(l)):
            if ldot[i] <=0:
                val = (ldot[i] + 0.25*10) / (0.25*10 - ldot[i]/0.25)
            else :
                val = (0.25*10*(1.4 - 1) + (2 + 2/0.25)*1.4*ldot[i]) / ( (2 + 2/0.25)*ldot[i] + 0.25*10*(1.4 - 1) )
            result[i][0]=val
        return result.requires_grad_(True)

    def F_tot(self,A, L, L_dot):
        return A*self.Fmax*self.fvl_tot(L/self.Lopt, L_dot/self.Lopt)*self.fl(L/self.Lopt)*torch.cos(self.alphas) + self.Fmax*self.fp(L/self.Lopt)*torch.cos(self.alphas)

    def F_tot_thelen(self,A, L, L_dot):
        self.Fmax = torch.tensor([[1417.0] ,[1086.0], [577.0], [2566.0],[2647.0],[233.0], [1468.0]], requires_grad = True)  #Il T 827 - ST M 1580
        self.alphas = torch.tensor([[0.14],[0.0], [0.09], [0.26],[0.05],[0.4], [0.3]], requires_grad = True) # unit ==
        self.Lopt = torch.tensor([0.117,0.156,0.075,0.068,0.097,0.110, 0.053], requires_grad = True)
        return A*self.Fmax*self.fvl_tot(L/self.Lopt, L_dot/self.Lopt)*self.fl(L/self.Lopt)*torch.cos(self.alphas) + self.Fmax*self.fp(L/self.Lopt)*torch.cos(self.alphas)


    def sigmoid_hip(self,theta ,factor1): # THE ARGUMENT PROVIDED TO THIS FUNCTION IS INDEED THETA (NOT Q) IN DEGREES
        # first compute theta hip to define min theta knee
        min_hip = self.theta_hip0
        max_hip = 100
        sig = theta[0][0]*(1/(1+torch.exp(-factor1*(theta[0][0] -(min_hip))))-(1/(1+torch.exp(-factor1*(theta[0][0]-(max_hip))))))
        sig_min = (min_hip)*(1-1/(1+torch.exp(-factor1*(theta[0][0] -(min_hip)))))
        sig_max = (max_hip)*(1/(1+torch.exp(-factor1*(theta[0][0]-(max_hip)))))
        return torch.tensor([[sig + sig_min + sig_max], [theta[1][0]]], device = self.dev, requires_grad = True)

    def sigmoid_min(self,theta, factor1 ,factor2 ): # THE ARGUMENT PROVIDED TO THIS FUNCTION IS INDEED THETA (NOT Q) IN DEGREES
        # sigmoid to balance gravity and ground reaction forces (knee on pillow + feet on table)
        min_hip = self.theta_hip0
        theta_hip = self.sigmoid_hip(theta, factor1)[0][0] ## theta a passer en radians no ?
        hmax = torch.sin(torch.deg2rad(theta_hip))*self.lfem
        min_knee = -torch.minimum(theta_hip+torch.rad2deg(torch.arcsin(hmax/self.ltib)),torch.tensor(120.0))
        sig = torch.tensor([[(1/(1+torch.exp(-factor1*(theta[0][0] -(min_hip)))))],[(1/(1+torch.exp(-factor2*(theta[1][0]-(min_knee)))))]], device = self.dev,  requires_grad = True)
        return sig

    def sigmoid_theta(self,theta,factor1 , factor2): # THE ARGUMENT PROVIDED TO THIS FUNCTION IS INDEED THETA (NOT Q) IN DEGREES

        theta = self.sigmoid_hip(theta,factor1)  # first compute theta hip to define min theta knee
        hmax = torch.sin(torch.deg2rad(theta[0][0]))*self.lfem
        min_knee = -torch.minimum(theta[0][0]+torch.rad2deg(torch.arcsin(hmax/self.ltib)),torch.tensor(120.0))
        max_knee = 0
        sig = theta[1][0]*(1/(1+torch.exp(-factor2*(theta[1][0] -(min_knee))))-(1/(1+torch.exp(-factor2*(theta[1][0]-(max_knee))))))
        sig_min = (min_knee)*(1-1/(1+torch.exp(-factor2*(theta[1][0] -(min_knee)))))
        sig_max = (max_knee)*(1/(1+torch.exp(-factor2*(theta[1][0]-(max_knee)))))
        return torch.tensor([[theta[0][0]], [sig + sig_min + sig_max]], device = self.dev ,  requires_grad = True)


    def sigmoid(self,theta,factor1 , factor2):
        theta_hip = self.sigmoid_hip(theta,factor1)[0][0]
        hmax = torch.sin(torch.deg2rad(theta_hip))*self.lfem
        min_hip = self.theta_hip0
        max_hip = 100
        min_knee = -torch.minimum(theta_hip+torch.rad2deg(torch.arcsin(hmax/self.ltib)),torch.tensor(120.0))
        max_knee = 0
        return torch.tensor([[(1/(1+torch.exp(-factor1*(theta[0][0] -(min_hip)))))-(1/(1+torch.exp(-factor1*(theta[0][0]-(max_hip)))))],[(1/(1+torch.exp(-factor2*(theta[1][0]-(min_knee)))))-(1/(1+torch.exp(-factor2*(theta[1][0]-(max_knee)))))]], device = self.dev, requires_grad = True)  ### -(-15)
        #return torch.tensor([[(1/(1+torch.exp(-self.sigmoid_factor1*(theta[0][0] -(self.theta_hip0+5)))))-(1/(1+torch.exp(-self.sigmoid_factor1*(theta[0][0]-115))))],[(1/(1+torch.exp(-self.sigmoid_factor2*(theta[1][0]+--(torch.minimum(theta[0][0]+torch.rad2deg(torch.arcsin(self.hp/self.ltib)),torch.tensor(120))+5)))))-(1/(1+torch.exp(-self.sigmoid_factor2*(theta[1][0]+5))))]])#-(-15)))))]])

    def update(self, U ):
        old_thet =  self.theta_
        #self.theta_ = self.theta_+ self.dt*self.theta_dot
        #if (self.hhip - self.lfem*torch.cos(self.theta_[0][0]))<(self.hhip+self.hp) : # knee height
        #    self.theta_[0][0]= torch.arccos(-self.hp/self.lfem)

        #if (self.hhip - self.lfem*torch.cos(self.theta_[0][0]) - self.ltib*torch.cos(self.theta_[1][0]))< self.hhip : # ankle height  ## (hhip)
        #    hmax = torch.sin(self.theta_[0][0]-torch.tensor(math.pi/2))*self.lfem
        #    self.theta_[1][0]= -torch.arcsin(hmax/self.ltib)+torch.tensor(math.pi/2 ) ### np.arccos((-lfem*np.cos(theta_[0][0]))/ltib)
        ##########ALICE ##################
        ### theta degrees
        #theta_deg = torch.tensor([[0.0],[0.0]], device  = self.dev, requires_grad = True)
        #theta_deg[0][0] = torch.rad2deg(old_thet[0][0]-math.pi/2)
        #theta_deg[1][0] = torch.rad2deg(old_thet[1][0]-old_thet[0][0])
        theta_deg = torch.tensor([[torch.rad2deg(old_thet[0][0]-math.pi/2)],[torch.rad2deg(old_thet[1][0]-old_thet[0][0])]], device  = self.dev, requires_grad = True)
        ### Sigmoid theta
        #theta_deg_new = torch.tensor([[0.0],[0.0]], device  = self.dev)
        #theta_deg_new[0][0] = torch.rad2deg(self.theta_[0][0]-math.pi/2)
        #theta_deg_new[1][0] = torch.rad2deg(self.theta_[1][0]-self.theta_[0][0])
        theta_deg_new = torch.tensor([[torch.rad2deg(self.theta_[0][0]-math.pi/2)],[torch.rad2deg(self.theta_[1][0]-self.theta_[0][0])]], device  = self.dev, requires_grad = True)
        self.sigmoid_factor1 = 10
        self.sigmoid_factor2 = 10
        theta_deg_sig = self.sigmoid_theta(theta_deg_new,self.sigmoid_factor1,self.sigmoid_factor2)
        self.theta_[0][0] = torch.deg2rad(theta_deg_sig[0][0]) + math.pi/2
        self.theta_[1][0] = self.theta_[0][0] + torch.deg2rad(theta_deg_sig[1][0])
###     ##################################
        #self.A_ = U
        L =  self.L_tot_thelen(self.theta_.clone())#.detach())
        L_dot = self.L_tot_dot_thelen(self.theta_.clone(), self.theta_dot.clone(),L)#.detach(), self.theta_dot.clone().detach(),L)
        self.A_ = self.A_ + self.dt*self.A_dot(self.A_,U)

        ################################3
        #L = self.L_tot(self.theta_.clone().detach())#.to(U.device)
        #L_dot = self.L_tot_dot(self.theta_.clone().detach(), self.theta_dot.clone().detach(),L)#.to(U.device)
        #self.A_ = self.A_ + self.dt*self.A_dot(self.A_,U)
        M_inv = torch.linalg.inv(self.M(self.theta_.clone()))#.to(U.device)

        ##################################
        ctheta_g = self.C(self.theta_, self.theta_dot)@self.theta_dot + self.G(self.theta_)
        self.theta_dot = self.theta_dot + self.dt*M_inv@( self.Rtot_thelen(self.theta_)@self.F_tot_thelen(self.A_,L,L_dot)  + ctheta_g +ctheta_g*(self.sigmoid_min(theta_deg, self.sigmoid_factor1, self.sigmoid_factor2)-1)) +self.theta_dot*(self.sigmoid(theta_deg, self.sigmoid_factor1, self.sigmoid_factor2)-1)
        ###################################33
        self.theta_ = self.theta_+ self.dt*self.theta_dot
        #ctheta_g = self.C(self.theta_, self.theta_dot)@self.theta_dot + self.G(self.theta_)

        #sig = (self.sigmoid(torch.rad2deg(theta_deg)-90)-1) # Alice ISN''T IT THETA CURRENT
        #self.theta_dot = self.theta_dot + self.dt*M_inv@( self.Rtot(self.theta_)@self.F_tot(self.A_,L,L_dot)  + self.C(self.theta_, self.theta_dot)@self.theta_dot + self.G(self.theta_) +(self.C(self.theta_, self.theta_dot)@self.theta_dot + self.G(self.theta_))*(self.sigmoid(torch.rad2deg(self.theta_)-90)-1)) +self.theta_dot*(self.sigmoid(torch.rad2deg(self.theta_)-90)-1)
        #self.theta_dot = self.theta_dot + self.dt*M_inv@( self.Rtot(self.theta_)@self.F_tot(self.A_,L,L_dot)  + ctheta_g + (ctheta_g)*sig) +self.theta_dot*sig


    def forward(self, U):
        #[6, 4, 584])
        for k in range(np.shape(U)[1]):
            self.reset()
            for i in range(U.shape[2]):
                #print(' U ' , U[:,i:i+1])
                self.update(U[:,k,i:i+1])
                #self.thetas = torch.cat((self.thetas, self.theta_-torch.tensor([[math.pi/2],[self.theta_[0][0]]] , device = U.device)), dim=1) # convert q to theta
                self.thetas = torch.cat((self.thetas, self.theta_ ), dim=1)
                self.theta_dots = torch.cat((self.theta_dots, self.theta_dot), dim=1)#.to(U.device)
                self.As = torch.cat((self.As, self.A_), dim=1)#.to(U.device)
                #print(' A ' , self.A_)
            if k ==0 :
                all_thetas = self.thetas[:,1:].unsqueeze(dim= 1)
            else :
                all_thetas = torch.cat((all_thetas,self.thetas[:,1:].unsqueeze(dim= 1)), dim = 1)
        #[2, 4, 584])
        #return self.thetas[:,1:].unsqueeze(dim= 1), self.As[:,1:]#.to(U.device)
        return all_thetas, self.As[:,1:]



class crazyleg2(nn.Module):
    def __init__(self,  hip0 = -10.0-80+90,knee0 = -80.0 +90, dt_ = 0.002025 ):
        super().__init__()
        dev = "cuda:0" if torch.cuda.is_available() else "cpu"
        device_ = torch.device(dev)
        #torch.autograd.set_detect_anomaly(True)
        self.dev = device_
        self.sigmoid_factor1 = 10#1000
        self.sigmoid_factor2 = 10
        self.g = 9.81
        self.hhip = 0.1#1
        self.lpelv = 0.16

        self.mfem = 9.1
        self.sfem = 0.15
        self.lfem = 0.39
        self.mtib = 3.6
        self.stib = 0.2
        self.ltib = 0.43
        self.sfoot = 0.06
        self.mfoot = 1.5
        self.lfoot = 0.07
        self.rknee = 0.03
        self.sgm = 0.13
        self.betast = math.pi/6
        self.d = torch.sqrt(self.lfem**2 + self.rknee**2 + 2 * self.lfem * self.rknee * torch.cos(torch.tensor(self.betast + 80* 3.14/180))) ##
        self.alpha = torch.acos((-self.rknee**2 + self.d**2 +self.lfem**2)/(2*self.d*self.lfem))
        self.hlow = 0.07
        self.rhip = 0.035
        self.hsup = torch.tensor(0.03, device = device_)
        self.sbf = 0.2
        self.betabf = math.pi/6
        self.Fmax = torch.tensor([[1417.0] ,[1176.0], [730.0],[1580.0],[3120.0],[470.0], [1513.0]], device = device_)
        self.alphas = torch.tensor([[0.14],[0.0], [0.09],[0.0],[0.05],[0.4], [0.3]], device = device_)
        self.Lopt = torch.tensor([0.102,0.158,0.112,0.109,0.104,0.177,0.1], device = device_)

        self.theta_hip0 = torch.tensor(30.0, device = device_)#np.degrees(np.arcsin(hp/lfem))
        self.hp = self.lfem* torch.sin(torch.deg2rad(self.theta_hip0))
        self.theta_knee0 = -(self.theta_hip0 + torch.rad2deg(torch.arcsin(self.hp/self.ltib)))
        theta0 = torch.tensor([[torch.deg2rad(self.theta_hip0)],[torch.deg2rad(self.theta_knee0)]], device = device_)  # these are actually theta, not q
        self.theta0 = theta0
        print(torch.rad2deg(theta0))
        self.hipstart = hip0 # this is q
        self.kneestart = knee0 # this is q
        #self.theta_ = torch.tensor([[math.radians(hip0)],[math.radians(knee0)]],requires_grad=True, device = device_)
        self.theta_ = torch.tensor([[self.theta0[0][0] + math.pi/2],[torch.abs(self.theta0[0][0]) - torch.abs(self.theta0[1][0])+ math.pi/2 ]], device = device_)
        self.q0 = self.theta_
        self.theta_dot = torch.tensor([[0.0],[0.0]], device = device_)
        self.A_ = torch.zeros((7, 1), device = device_ )
        #self.thetas= torch.tensor([[math.radians(hip0)],[math.radians(knee0)]],requires_grad=True, device = device_)
        self.thetas = torch.tensor([[self.theta0[0][0] + math.pi/2],[torch.abs(self.theta0[0][0]) - torch.abs(self.theta0[1][0])+ math.pi/2 ]], device = device_,requires_grad=True)
        self.theta_dots = torch.tensor([[0.0],[0.0]], device = device_)
        self.As = torch.zeros((7, 1), device = device_)
        self.dt = dt_
        self.tact = [0.01, 0.04]
        #q = np.array([[theta[0][0] + math.pi/2],[np.abs(theta[0][0]) - np.abs(theta[1][0])+ math.pi/2 ]])

    def get_theta0 (self):
        return self.theta0

    def reset(self):
        #self.theta_ = torch.tensor([[math.radians(self.hipstart)],[math.radians(self.kneestart)]],requires_grad=True, device = self.dev)
        self.theta_ = torch.tensor([[self.theta0[0][0] + math.pi/2],[torch.abs(self.theta0[0][0]) - torch.abs(self.theta0[1][0])+ math.pi/2 ]], device = self.dev)
        self.theta_dot = torch.tensor([[0.0],[0.0]], device = self.dev)
        self.A_ = torch.zeros((7, 1), device = self.dev,dtype=torch.float32)
        #self.thetas = torch.tensor([[math.radians(self.hipstart)],[math.radians(self.kneestart)]],requires_grad=True, device = self.dev)
        self.thetas = torch.tensor([[self.theta0[0][0] + math.pi/2],[torch.abs(self.theta0[0][0]) - torch.abs(self.theta0[1][0])+ math.pi/2 ]], device = self.dev,requires_grad=True)
        self.theta_dots = torch.tensor([[0.0],[0.0]], device = self.dev)
        self.As = torch.zeros((7, 1), device = self.dev,dtype=torch.float32)
    def G(self, q):
        g1 = -(self.mfem*self.sfem + (self.mtib + self.mfoot)*self.lfem)*self.g*torch.sin(q[0][0])
        g2 = -(self.mtib*self.stib+ self.mfoot*self.ltib)*self.g*torch.sin(q[1][0])
        return torch.tensor([[g1], [g2]] , device =q.device )

    def C(self,q, q_dot):
        c12 = -(self.mtib*self.lfem*self.stib + self.mfoot*self.lfem*self.ltib)*torch.sin(q[0][0] - q[1][0])*(q_dot[1][0])
        c21 = (self.mtib*self.lfem*self.stib + self.mfoot*self.lfem*self.ltib)*torch.sin(q[0][0] - q[1][0])*(q_dot[0][0])
        return torch.tensor([[0 , c12], [c21 , 0]] , device =q.device )

    def Rtot(self,q):
        RVlat = 0.05*torch.cos(0.8*(q[1][0]-math.radians(100)))#math.radians(80)))
        RBF = -0.038*torch.cos(q[1][0]-math.radians(65))#math.radians(35))
        RRF_hip = 0.053*torch.cos(1*(q[0][0]-math.radians(127)))
        RRF_knee = 0.053*torch.cos(0.55*(q[1][0]-math.radians(100)))#math.radians(70)))
        RST_hip = -0.07*torch.cos(q[0][0]-math.radians(122))
        RST_knee = -0.04*torch.cos(q[1][0]-math.radians(85))#math.radians(55))
        RIl = 0.045*torch.cos(0.6*(q[0][0]-math.radians(140)))
        RGM = -0.06*torch.cos(0.75*(q[0][0]-math.radians(90)))
        RMG = -0.038*torch.cos(0.7*(q[1][0]-math.radians(50)))
        return torch.tensor([[RIl, RGM, RRF_hip, RST_hip,0,0,0], [0,0,RRF_knee, RST_knee,RVlat,RBF,RMG]] , device =q.device )

    def Rtot_thelen(self,q):
        RVlat = 0.048*torch.cos(0.7*(q[1][0]-math.radians(110)))#math.radians(80)))
        RBF = -0.035*torch.cos(q[1][0]-math.radians(72))#math.radians(35))
        RRF_hip = 0.053*torch.cos(1*(q[0][0]-math.radians(127)))
        RRF_knee = 0.052*torch.cos(0.9*(q[1][0]-math.radians(100)))#math.radians(70)))
        RST_hip = -0.06*torch.cos(q[0][0]-math.radians(122))
        RST_knee = -0.042*torch.cos(0.9*(q[1][0]-math.radians(75)))#math.radians(55))
        RIl = 0.045*torch.cos(0.6*(q[0][0]-math.radians(140)))
        RGM = -0.058*torch.cos(0.65*(q[0][0]-math.radians(70)))
        RMG = -0.038*torch.cos(0.7*(q[1][0]-math.radians(50)))
        return torch.tensor([[RIl, RGM, RRF_hip, RST_hip,0,0,0], [0,0,RRF_knee, RST_knee,RVlat,RBF,RMG]], device =q.device )



    def M(self,q):
        m11 = 0.112007 + self.mfem*(self.sfem**2) + (self.mtib + self.mfoot)*self.lfem**2
        m12 = (self.mtib*self.lfem*self.stib + self.mfoot*self.lfem*self.ltib)*torch.cos(q[0][0]-q[1][0])
        m21 = m12
        m22 = 0.0591636 + self.mtib*(self.stib**2) + self.mfoot*(self.ltib**2)
        return torch.tensor([[m11 , m12], [m21 , m22]], device = q.device)

    def A_dot(self, A, U):
        #tact =0.05#0.5# 0.01
        if len(self.tact)== 1:
            return (U-A)/self.tact[1]
        else:
            inter = 0.5+1.5*A
            t_act = self.tact[0] ##0.01
            tdeact = self.tact[1] ##0.04
            tau = t_act*inter + (U<A)*((tdeact /inter) - t_act*inter)
        return (U-A)/tau

    def sigmoid_lst(self,length):  ##
        return (1-0.05)/(1+torch.exp(-100*(length-0.07)))

    def sigmoid_lstmax(self,lst, factor=100, b=0.125):
        return (1)/(1+torch.exp(-factor*(lst-b)))

    def L_tot(self,q_):
        q =  q_
        LIl = 0.094 - 0.035*(q[0][0] - math.pi/2)
        sigLil = self.sigmoid_lst(LIl.clone())
        LIl = LIl.clone()*sigLil + (1-sigLil)*0.05
        LGM = 0.127 + 0.04*(q[0][0]- math.pi/2)
        sigGM = self.sigmoid_lst(LGM.clone())
        LGM = LGM.clone()*sigGM + (1-sigGM)*0.05
        #LRF = 0.06 + torch.sqrt(torch.tensor(self.lfem**2 +self.hsup**2 + 2*self.lfem*self.hsup*torch.cos(q[0][0]))) - torch.sqrt(torch.tensor(self.lfem**2 +self.hsup**2)) - self.rknee*(q[1][0]-q[0][0])
        LRF = 0.06 + torch.sqrt(self.lfem**2 +self.hsup**2 + 2*self.lfem*self.hsup*torch.cos(q[0][0]))  - torch.sqrt(torch.tensor(self.lfem**2 +self.hsup**2)) - self.rknee*(q[1][0]-q[0][0])
        sigRF = self.sigmoid_lst(LRF.clone())
        LRF = LRF.clone()*sigRF + (1-sigRF)*0.05
        LST = 0.055 + 0.05*(q[0][0]- math.pi/2) + self.rknee * (q[1][0] - q[0][0])
        sigST = self.sigmoid_lst(LST.clone())
        LST = LST.clone()*sigST + (1-sigST)*0.05
        Lvlat = 0.046 + self.rknee*(q[0][0] - q[1][0])
        sigVLat = self.sigmoid_lst(Lvlat.clone())
        Lvlat = Lvlat.clone()*sigVLat + (1-sigVLat)*0.05
        Lbf =  0.139 + self.rknee * (q[1][0] - q[0][0])
        Lbf = Lbf.clone()*self.sigmoid_lst(Lbf.clone()) + (1-self.sigmoid_lst(Lbf.clone()))*0.05
        LMG = 0.055 + self.rknee * (q[1][0] - q[0][0])
        LMG = LMG.clone()*self.sigmoid_lst(LMG.clone()) + (1-self.sigmoid_lst(LMG.clone()))*0.05
        return torch.tensor([LIl, LGM, LRF, LST, Lvlat, Lbf, LMG], device = q_.device, requires_grad = False)

    def sigmoid_lstmax(self,lst, factor=100, b=0.125):
        return (1)/(1+torch.exp(-factor*(lst-b)))


    def L_tot_thelen(self, q):
        # order : [LIl, LGM, LRF, LST, Lvlat, Lbf, LMG]
        self.Lopt = torch.tensor([0.117,0.156,0.075,0.068,0.097,0.110, 0.053],device = q.device)
        Lnorm_min = 0.5  ## SET BELOW IN L_tot_dot()
        Lnorm_max = 1.7
        factor = 10

        LIl = 0.125 - 0.035*(q[0][0] - math.pi/2)
        LGM = 0.159 + 0.04*(q[0][0]- math.pi/2)
        LRF = 0.069 + torch.sqrt(self.lfem**2 +self.hsup**2 + 2*self.lfem*self.hsup*torch.cos(q[0][0])) - torch.sqrt(self.lfem**2 +self.hsup**2) - 0.045*(q[1][0]-q[0][0])
        LST = 0.073 + 0.05*(q[0][0]- math.pi/2) + 0.035 * (q[1][0] - q[0][0])
        Lvlat = 0.098 + 0.04*(q[0][0] - q[1][0])
        Lbf =  0.152 + 0.03 * (q[1][0] - q[0][0])
        LMG = 0.06 + 0.02 * (q[1][0] - q[0][0])

        # sigmoid for min and max Lnorm
        Ltot = torch.tensor([LIl, LGM, LRF, LST, Lvlat, Lbf, LMG],device = q.device)
        vals = torch.zeros(np.shape(Ltot),device = q.device)
        for m in range(len(Ltot)):
            vals[m] = Ltot[m]*(self.sigmoid_lstmax(Ltot[m]/self.Lopt[m], factor=factor, b=Lnorm_min)-self.sigmoid_lstmax(Ltot[m]/self.Lopt[m], factor=factor, b=Lnorm_max)) + (1-self.sigmoid_lstmax(Ltot[m]/self.Lopt[m], factor=factor, b=Lnorm_min))*Lnorm_min*self.Lopt[m] + self.sigmoid_lstmax(Ltot[m]/self.Lopt[m], factor=factor, b=Lnorm_max)*Lnorm_max*self.Lopt[m]

        return vals#.detach().requires_grad_(True)


    def L_tot_dot(self, q , q_dot,L):
        sigL = self.sigmoid_lst(L)
        LIldot = -self.rhip*q_dot[0][0]
        LIldot = sigL[0]*LIldot.clone()
        LGMdot = 0.04*q_dot[0][0]
        LGMdot = sigL[1]*LGMdot.clone()
        LRFdot = -2*self.lfem*self.hsup*torch.sin(q[0][0])*q_dot[0][0]/(2*torch.sqrt(self.lfem**2 + self.hsup**2 + 2*self.lfem*self.hsup*torch.cos(q[0][0]))) - self.rknee*(q_dot[1][0] - q_dot[0][0])
        LRFdot = sigL[2]*LRFdot.clone()
        LSTdot = 0.05*q_dot[0][0] + self.rknee * (q_dot[1][0] - q_dot[0][0])
        LSTdot = sigL[3]*LSTdot.clone()
        Lvlatdot = self.rknee * q_dot[0][0] - self.rknee * q_dot[1][0]
        Lvlatdot = sigL[4]*Lvlatdot.clone()
        Lbfdot = self.rknee * (q_dot[1][0] - q_dot[0][0])
        Lbfdot = sigL[5]*Lbfdot.clone()
        LMGdot = self.rknee * (q_dot[1][0] - q_dot[0][0])
        LMGdot = sigL[6]*LMGdot.clone()
        return torch.tensor([LIldot, LGMdot, LRFdot, LSTdot, Lvlatdot,Lbfdot,LMGdot], device = q.device, requires_grad = False)

    def L_tot_dot_thelen(self, q, q_dot, L):
        # order : [LIl, LGM, LRF, LST, Lvlat, Lbf, LMG]
        Lnorm_min = 0.5
        Lnorm_max = 1.7
        factor = 10

        LIldot = 0.035*(q_dot[0][0])
        LGMdot = 0.04*(q_dot[0][0])
        LRFdot = -2*self.lfem*self.hsup*torch.sin(q[0][0])*q_dot[0][0]/(2*torch.sqrt(self.lfem**2 + self.hsup**2 + 2*self.lfem*self.hsup*torch.cos(q[0][0]))) - 0.045*(q_dot[1][0] - q_dot[0][0])
        LSTdot = 0.05*q_dot[0][0] + 0.035 * (q_dot[1][0] - q_dot[0][0])
        Lvlatdot = 0.04 * q_dot[0][0] -self.rknee * q_dot[1][0]
        Lbfdot = 0.03 * (q_dot[1][0] - q_dot[0][0])
        LMGdot = 0.02 * (q_dot[1][0] - q_dot[0][0])

        # sigmoid for min and max Lnorm
        Ldot = torch.tensor([LIldot, LGMdot, LRFdot, LSTdot, Lvlatdot, Lbfdot, LMGdot],device = q.device)
        vals = torch.zeros(np.shape(Ldot),device = q.device)
        for m in range(len(Ldot)):
            vals[m] = Ldot[m]*(self.sigmoid_lstmax(L[m]/self.Lopt[m], factor=factor, b=Lnorm_min)-self.sigmoid_lstmax(L[m]/self.Lopt[m], factor=factor, b=Lnorm_max))

        return vals#.detach().requires_grad_(True)


    def fl(self,l):
        return torch.unsqueeze(torch.exp(-((l-1)**2)/0.45),1) #.to(l.device)

    def fp(self , l):
        res = torch.zeros((len(l), 1), device = l.device)
        for k in range((len(l))):
            res[k][0] = torch.exp((5*(l[k]-1)/0.7)-1)/(torch.exp(torch.tensor(5.0))-1)
        return res

    def fvl_tot(self,l,ldot):
        result = torch.ones((len(l),1), device = l.device)
        for i in range (len(l)):
            if ldot[i] <=0:
                val = (ldot[i] + 0.25*10) / (0.25*10 - ldot[i]/0.25)
            else :
                val = (0.25*10*(1.4 - 1) + (2 + 2/0.25)*1.4*ldot[i]) / ( (2 + 2/0.25)*ldot[i] + 0.25*10*(1.4 - 1) )
            result[i][0]=val
        return result

    def F_tot(self,A, L, L_dot):
        return A*self.Fmax*self.fvl_tot(L/self.Lopt, L_dot/self.Lopt)*self.fl(L/self.Lopt)*torch.cos(self.alphas) + self.Fmax*self.fp(L/self.Lopt)*torch.cos(self.alphas)

    def F_tot_thelen(self,A, L, L_dot):
        self.Fmax = torch.tensor([[1417.0] ,[1086.0], [577.0], [2566.0],[2647.0],[233.0], [1468.0]],device = A.device)  #Il T 827 - ST M 1580
        self.alphas = torch.tensor([[0.14],[0.0], [0.09], [0.26],[0.05],[0.4], [0.3]],device = A.device) # unit ==
        self.Lopt = torch.tensor([0.117,0.156,0.075,0.068,0.097,0.110, 0.053],device = A.device)
        return A*self.Fmax*self.fvl_tot(L/self.Lopt, L_dot/self.Lopt)*self.fl(L/self.Lopt)*torch.cos(self.alphas) + self.Fmax*self.fp(L/self.Lopt)*torch.cos(self.alphas)


    def sigmoid_hip(self,theta_ ,factor1): # THE ARGUMENT PROVIDED TO THIS FUNCTION IS INDEED THETA (NOT Q) IN DEGREES
        # first compute theta hip to define min theta knee
        theta= theta_.clone()
        theta0 = theta.clone()[0][0]
        theta1 = theta.clone()[1][0]

        min_hip = self.theta_hip0
        max_hip = 100
        sig = theta0*(1/(1+torch.exp(-factor1*(theta0 -(min_hip))))-(1/(1+torch.exp(-factor1*(theta0-(max_hip))))))
        sig_min = (min_hip)*(1-1/(1+torch.exp(-factor1*(theta0 -(min_hip)))))
        sig_max = (max_hip)*(1/(1+torch.exp(-factor1*(theta0-(max_hip)))))
        theta[0][0] = sig + sig_min+ sig_max
        return theta#torch.tensor([[sig + sig_min + sig_max], [theta[1][0]]], device = self.dev)

    def sigmoid_min(self,theta_, factor1 ,factor2 ): # THE ARGUMENT PROVIDED TO THIS FUNCTION IS INDEED THETA (NOT Q) IN DEGREES
        # sigmoid to balance gravity and ground reaction forces (knee on pillow + feet on table)
        theta= theta_.clone()
        theta0 = theta.clone()[0][0]
        theta1 = theta.clone()[1][0]
        min_hip = self.theta_hip0
        theta_hip = self.sigmoid_hip(theta, factor1)[0][0] ## theta a passer en radians no ?
        hmax = torch.sin(torch.deg2rad(theta_hip))*self.lfem
        min_knee = -torch.minimum(theta_hip+torch.rad2deg(torch.arcsin(hmax/self.ltib)),torch.tensor(120.0))
        theta[0][0] = (1/(1+torch.exp(-factor1*(theta0 -(min_hip)))))
        theta[1][0] = (1/(1+torch.exp(-factor2*(theta1-(min_knee)))))
        return theta#sig

    def sigmoid_theta(self,theta_,factor1 , factor2): # THE ARGUMENT PROVIDED TO THIS FUNCTION IS INDEED THETA (NOT Q) IN DEGREES
        theta= theta_.clone()
        theta = self.sigmoid_hip(theta_,factor1).clone()  # first compute theta hip to define min theta knee
        theta0 = theta.clone()[0][0]
        theta1 = theta.clone()[1][0]
        hmax = torch.sin(torch.deg2rad(theta0))*self.lfem
        min_knee = -torch.minimum(theta[0][0]+torch.rad2deg(torch.arcsin(hmax/self.ltib)),torch.tensor(120.0))
        max_knee = 0
        sig = theta1*(1/(1+torch.exp(-factor2*(theta1 -(min_knee))))-(1/(1+torch.exp(-factor2*(theta1-(max_knee))))))
        sig_min = (min_knee)*(1-1/(1+torch.exp(-factor2*(theta1 -(min_knee)))))
        sig_max = (max_knee)*(1/(1+torch.exp(-factor2*(theta1-(max_knee)))))
        theta[1][0] = sig + sig_min+ sig_max
        #print(sig , sig_min , sig_max)
        return theta#torch.tensor([[theta.clone()[0][0]], [theta.clone()[1][0]]], device = self.dev)

    def a(self):
        return self.A_

    def sigmoid(self,theta_,factor1 , factor2):
        theta= theta_.clone()
        theta0 = theta.clone()[0][0]
        theta1 = theta.clone()[1][0]
        theta_hip = self.sigmoid_hip(theta,factor1)[0][0]
        hmax = torch.sin(torch.deg2rad(theta_hip))*self.lfem
        min_hip = self.theta_hip0
        max_hip = 100
        min_knee = -torch.minimum(theta_hip+torch.rad2deg(torch.arcsin(hmax/self.ltib)),torch.tensor(120.0))
        max_knee = 0
        theta[0][0] = (1/(1+torch.exp(-factor1*(theta0 -(min_hip)))))-(1/(1+torch.exp(-factor1*(theta0-(max_hip)))))
        theta[1][0] = (1/(1+torch.exp(-factor2*(theta1-(min_knee)))))-(1/(1+torch.exp(-factor2*(theta1-(max_knee))))) ### -(-15)
        return theta#torch.tensor([[(1/(1+torch.exp(-factor1*(theta0 -(min_hip)))))-(1/(1+torch.exp(-factor1*(theta0-(max_hip)))))],[(1/(1+torch.exp(-factor2*(theta1-(min_knee)))))-(1/(1+torch.exp(-factor2*(theta1-(max_knee)))))]], device = self.dev)  ### -(-15)
        #return torch.tensor([[(1/(1+torch.exp(-self.sigmoid_factor1*(theta[0][0] -(self.theta_hip0+5)))))-(1/(1+torch.exp(-self.sigmoid_factor1*(theta[0][0]-115))))],[(1/(1+torch.exp(-self.sigmoid_factor2*(theta[1][0]+--(torch.minimum(theta[0][0]+torch.rad2deg(torch.arcsin(self.hp/self.ltib)),torch.tensor(120))+5)))))-(1/(1+torch.exp(-self.sigmoid_factor2*(theta[1][0]+5))))]])#-(-15)))))]])

    def update(self, U ):
        old_thet =  self.theta_.clone()
        #self.theta_ = old_thet+ self.dt*self.theta_dot

        #if (self.hhip - self.lfem*torch.cos(self.theta_[0][0]))<(self.hhip+self.hp) : # knee height
        #    self.theta_[0][0]= torch.arccos(-self.hp/self.lfem)

        #if (self.hhip - self.lfem*torch.cos(self.theta_[0][0]) - self.ltib*torch.cos(self.theta_[1][0]))< self.hhip : # ankle height  ## (hhip)
        #    hmax = torch.sin(self.theta_[0][0]-torch.tensor(math.pi/2))*self.lfem
        #    self.theta_[1][0]= -torch.arcsin(hmax/self.ltib)+torch.tensor(math.pi/2 ) ### np.arccos((-lfem*np.cos(theta_[0][0]))/ltib)
        ##########ALICE ##################
        ### theta degrees
        #theta_deg = torch.tensor([[0.0],[0.0]], device  = self.dev, requires_grad = True)
        #theta_deg[0][0] = torch.rad2deg(old_thet[0][0]-math.pi/2)
        #theta_deg[1][0] = torch.rad2deg(old_thet[1][0]-old_thet[0][0])
        theta_deg = torch.tensor([[torch.rad2deg(old_thet[0][0]-math.pi/2)],[torch.rad2deg(old_thet[1][0]-old_thet[0][0])]], device  = self.dev)
        ### Sigmoid theta
        #theta_deg_new = torch.tensor([[0.0],[0.0]], device  = self.dev)
        #theta_deg_new[0][0] = torch.rad2deg(self.theta_[0][0]-math.pi/2)
        #theta_deg_new[1][0] = torch.rad2deg(self.theta_[1][0]-self.theta_[0][0])
        theta_deg_new = torch.tensor([[torch.rad2deg(self.theta_[0][0]-math.pi/2)],[torch.rad2deg(self.theta_[1][0]-self.theta_[0][0])]], device  = self.dev)
        self.sigmoid_factor1 = 10
        self.sigmoid_factor2 = 10
        theta_deg_sig = self.sigmoid_theta(theta_deg_new,self.sigmoid_factor1,self.sigmoid_factor2)
        self.theta_[0][0] = torch.deg2rad(theta_deg_sig[0][0]) + math.pi/2
        self.theta_[1][0] = self.theta_.clone()[0][0] + torch.deg2rad(theta_deg_sig[1][0])
###     ##################################
        #self.A_ = U
        L =  self.L_tot_thelen(self.theta_)#.detach())
        L_dot = self.L_tot_dot_thelen(self.theta_, self.theta_dot,L)#.detach(), self.theta_dot.clone().detach(),L)
        A__ = self.A_.clone()
        #print(U.is_leaf)
        #self.A_ = self.A_ + self.dt*self.A_dot(self.A_,U)
        self.A_ = A__ + self.dt*self.A_dot(A__,U)
        ################################3
        #L = self.L_tot(self.theta_.clone().detach())#.to(U.device)
        #L_dot = self.L_tot_dot(self.theta_.clone().detach(), self.theta_dot.clone().detach(),L)#.to(U.device)
        #self.A_ = self.A_ + self.dt*self.A_dot(self.A_,U)
        M_inv = torch.linalg.inv(self.M(self.theta_))#.to(U.device)
        ##################################
        ctheta_g = self.C(self.theta_, self.theta_dot)@self.theta_dot + self.G(self.theta_)
        self.theta_dot = self.theta_dot.clone() + self.dt*M_inv@( self.Rtot_thelen(self.theta_)@self.F_tot_thelen(self.A_,L,L_dot)  + ctheta_g +ctheta_g*(self.sigmoid_min(theta_deg, self.sigmoid_factor1, self.sigmoid_factor2)-1)) +self.theta_dot*(self.sigmoid(theta_deg, self.sigmoid_factor1, self.sigmoid_factor2)-1)
        self.theta_ = old_thet+ self.dt*self.theta_dot
        ###################################33
        #ctheta_g = self.C(self.theta_, self.theta_dot)@self.theta_dot + self.G(self.theta_)

        #sig = (self.sigmoid(torch.rad2deg(theta_deg)-90)-1) # Alice ISN''T IT THETA CURRENT
        #self.theta_dot = self.theta_dot + self.dt*M_inv@( self.Rtot(self.theta_)@self.F_tot(self.A_,L,L_dot)  + self.C(self.theta_, self.theta_dot)@self.theta_dot + self.G(self.theta_) +(self.C(self.theta_, self.theta_dot)@self.theta_dot + self.G(self.theta_))*(self.sigmoid(torch.rad2deg(self.theta_)-90)-1)) +self.theta_dot*(self.sigmoid(torch.rad2deg(self.theta_)-90)-1)
        #self.theta_dot = self.theta_dot + self.dt*M_inv@( self.Rtot(self.theta_)@self.F_tot(self.A_,L,L_dot)  + ctheta_g + (ctheta_g)*sig) +self.theta_dot*sig


    def forward(self, U):
        #[6, 4, 584])
        for k in range(np.shape(U)[1]):

            self.reset()
            for i in range(U.shape[2]):

                #print(' U ' , U[:,i:i+1])
                self.update(U[:,k,i:i+1])
                #self.thetas = torch.cat((self.thetas, self.theta_-torch.tensor([[math.pi/2],[self.theta_[0][0]]] , device = U.device)), dim=1) # convert q to theta
                self.thetas = torch.cat((self.thetas.clone(), self.theta_.clone() ), dim=1)
                self.theta_dots = torch.cat((self.theta_dots.clone(), self.theta_dot.clone()), dim=1)#.to(U.device)
                self.As = torch.cat((self.As.clone(), self.A_.clone()), dim=1)#.to(U.device)
                #print(' A ' , self.A_)
            if k ==0 :
                all_thetas = self.thetas[:,1:].clone().unsqueeze(dim= 1)
            else :
                all_thetas = torch.cat((all_thetas.clone(),self.thetas[:,1:].clone().unsqueeze(dim= 1)), dim = 1)

        #[2, 4, 584])
        #return self.thetas[:,1:].unsqueeze(dim= 1), self.As[:,1:]#.to(U.device)
        return all_thetas, self.As[:,1:]
