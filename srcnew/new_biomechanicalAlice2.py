import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from scipy import signal
import os


### simulation ###

# data
#with open('leftleg.npy', 'rb') as f:
with open('leftlegs180.npy', 'rb') as f:
    ang = np.load(f, allow_pickle=True)
    ang = ang[0]
print('angles ', np.shape(ang))

#with open('emg54.npy', 'rb') as f:
with open('emgs180.npy', 'rb') as f:
    emg = np.load(f, allow_pickle=True)
    emg = emg[0]
print('emg ', np.shape(emg))

test_plot = False  # True to plot the muscle length and determinant of M
simulation = True  # Trknee0ue to run the simulation
emg_env = False  # True to compute and plot emg envelop
result_folder = "biomechresults/"
if not os.path.exists(result_folder):
    os.makedirs(result_folder)

# initial position in degrees and simulation parameters
#hip0 = -10
#knee0 = -80

dt = 0.002 #0.01*0.2 # [s]# not more than 0.01
nbpt = len(emg[:,0])
time = np.linspace(0, 999, num=nbpt) #1000)
sigmoid_factor1 = 1000
sigmoid_factor2 = 1000
# muscle input right leg
muscle_names = ["Il", "GM", "RF", "ST", "VLat", 'BF']
corresponding_emg = [0, None, 1, 3, 2, 3]
scaling_factor = [1, 1, 1, 1, 1, 1]

# Plot EMG
fig, ax = plt.subplots(3, 2, figsize=(13, 10))
for m in range(len(muscle_names)):
    if corresponding_emg[m] is not None:
        ax[m%3][m//3].plot(time, emg[:, corresponding_emg[m]])
        ax[m%3][m//3].set_title('EMG '+muscle_names[m])
plt.tight_layout()
f = result_folder + "EMG.png"
fig.savefig(f)

# Process EMG env
if emg_env:
    for m in range(len(emg[0])):
        emg[:,m] = np.abs(emg[:,m])/np.max(np.abs(emg))  # MAX OVER ALL DATA SET

    # Plot EMG
    fig, ax = plt.subplots(3, 2, figsize=(13, 10))
    for m in range(len(muscle_names)):
        if corresponding_emg[m] is not None:
            ax[m%3][m//3].plot(time, emg[:, corresponding_emg[m]])
            ax[m%3][m//3].set_title('EMG '+muscle_names[m])
    plt.tight_layout()
    f = result_folder + "processed_EMG.png"
    fig.savefig(f)


###optimal values :
### order : [LIl, LGM, LRF, LST, Lvlat,Lbf]

### For your information :
# knee flexion
#U[5, 0:300] = 0.2
# hip flexion
#U[0, 0:300] = 0.7
# knee extension
#U[2, 0:300] = 0.7
    #or
#U[4, 0:300] = 0.1
# hip flexion + knee extension
#U[0, 0:300] = 0.48
#U[2, 0:300] = 0.44
# hip flexion + knee flexion
#U[0, 0:300] = 0.7
#U[5, 0:300] = 0.2

### test plateau1 :
#U[0, 0:200] = 0.55
#U[0, 400:600] = 0.65
#U[0, 800:970] = 0.75
### test plateau1 :
#U[2, 0:200] = 0.4
#U[2, 400:600] = 0.5
#U[2, 800:970] = 0.6

### test trial 54 :
#U[4, 170:] = 0.04#0.03
#U[4, 170:170+200] = 0.04*np.sqrt(np.linspace(0, 1, num=200))
#U[2, :100] =0.01
#U[2, 100:] = 0.2#0.2 #wiggle$
#U[2, 100:100+100] = 0.19*np.sqrt(np.linspace(0, 1, num=100))+0.01
#U[1, 0:100] = 0.1*np.sqrt(np.linspace(0, 1, num=100))#0.2 #wiggle$
#U[1, :] = 0.2*(1-np.sqrt(np.linspace(0.02, 1, num=nbpt)))#1000)))
#U[5, 0:150] = 0.08
#U[5, 150:250] = 0.08*(1-np.sqrt(np.linspace(0, 1, num=100)))


U = np.zeros((6, nbpt))
U[4, int(0.17*nbpt)] = 0.04#0.03
U[4,  int(0.17*nbpt): int(0.17*nbpt)+ int(0.20*nbpt)] = 0.04*np.sqrt(np.linspace(0, 1, num= int(0.20*nbpt)))
U[2, : int(0.10*nbpt)] =0.01
U[2,  int(0.10*nbpt):] = 0.2#0.2 #wiggle$
U[2,  int(0.10*nbpt): int(0.10*nbpt)+int(0.20*nbpt)] = 0.19*np.sqrt(np.linspace(0, 1, num= int(0.20*nbpt)))+0.01
U[1, 0: int(0.10*nbpt)] = 0.1*np.sqrt(np.linspace(0, 1, num= int(0.10*nbpt)))#0.2 #wiggle$
U[1, :] = 0.2*(1-np.sqrt(np.linspace(0.1, 1, num=nbpt)))#1000)))
U[5] = 0.02
U[5, 0: int(0.15*nbpt)] = 0.08
#U[5,  int(0.15*nbpt): int(0.25*nbpt)] = 0.06*(1-np.sqrt(np.linspace(0, 1, num= int(0.10*nbpt))))+0.02

# try with peaks - KNEE EXTENSION
from scipy.signal import hilbert
def sigmoid(x):
    #return 1.0 / (1.0 + np.exp(-(1.3*x-60)/9))
    #return 1.0 / (1.0 + np.exp(-(15*x-180)/20))
    #return 1.0 / (1.0 + np.exp(-(x-50)/10))
    return -1 + 2.0 / (1.0 + np.exp(-(x)/3))

low_pass = 3
sfreq = 1/dt
low_pass = low_pass/(sfreq/2)
print('lowpass   ' , low_pass)
b2, a2 = signal.butter(4, low_pass, btype='lowpass')
#emg_envelope = signal.filtfilt(b2, a2, emg_rectified)


U = np.zeros((6, nbpt))

U[0] = 1.82*hilbert(signal.filtfilt(b2, a2, np.abs(sigmoid(emg[:,0]))))
U[2] = 0.72*  hilbert(signal.filtfilt(b2, a2, np.abs(sigmoid(emg[:,1]))))
U[3] =0.82 *  hilbert(signal.filtfilt(b2, a2, np.abs(sigmoid(emg[:,0]))))
U[4] = 0.2*  hilbert(signal.filtfilt(b2, a2, np.abs(sigmoid(emg[:,0]))))
U[5] = 0.7*  hilbert(signal.filtfilt(b2, a2, np.abs(sigmoid(emg[:,0]))))
test = None#"knee_flex"  # in "knee_flex", "hip_flex_ext", "full_hip_flex", "full_hip_flex_ext"

# TEST KNEE EXTENSION-FLEXION
if test == "knee_flex":
    U = np.zeros((6, nbpt))
    U[2, 100:500] = 0.1
    U[4, 100:500] = 0.1
# TEST HIP AND KNEE FLEXION-EXTENSION
elif test == "hip_flex_ext":
    U[0, 100:500] = 0.5
    U[2, 100:500] = 0.2
# TEST FULL HIP AND KNEE FLEXION
elif test == "full_hip_flex":
    U = np.zeros((6, nbpt))
    U[0, 100:500] = 0.6
    U[2, 100:500] = 0.2
# TEST FULL HIP AND KNEE FLEXION THEN EXTENSION (full hip and knee flexion is a stable position)
elif test == "full_hip_flex_ext":
    U[0, 100:500] = 0.6
    U[2, 100:500] = 0.2
    U[3, 550:700] = 0.6


#for i in range (20):
#    U[0, i*50:i*50+40] = 0.8
#U[0, 100:] = 0.5
### Constants ###
g = 9.81
mfem = 9.1
sfem = 0.15
lfem = 0.39
mtib = 3.6
stib = 0.2
ltib = 0.43
sfoot = 0.06
mfoot = 1.5
lfoot = 0.07
rknee = 0.03
sgm = 0.13
betast = math.pi/6 ##
d = np.sqrt(lfem**2 + rknee**2 + 2 * lfem * rknee * np.cos(betast + 80* 3.14/180)) ##
alpha = np.arccos((-rknee**2 + d**2 +lfem**2)/(2*d*lfem))
hlow = 0.07 ##
rhip = 0.035
hsup = 0.03 ##
sbf = 0.2
betabf = math.pi/6
#knee0 = -30# tbc en degrees
#L = np.sqrt(lfem**2 + ltib**2 + 2 * lfem * ltib*np.cos(math.radians(knee0)))
hip0 = 30 #np.degrees(np.arcsin(hp/lfem))
hp = lfem* np.sin(math.radians(hip0))
print("hp: ", hp)
knee0 = -(hip0 + np.degrees(np.arcsin(hp/ltib)))
#hip0 = np.degrees(np.arccos((lfem**2 + L**2 - ltib**2)/(2*L*lfem))) #degrees
print(' initial angle of knee ', knee0)
theta0 = np.array([[math.radians(hip0)],[math.radians(knee0)]])  # these are actually theta, not q
thetadot0 = np.array([[0.0], [0.0]])   ### ATTENTION [0],[0] int array (useless as defined in simulation again)
A0 = np.array([[0.0],[0.0]])  ### idem
#################

def G(q):
# Theta should be in degrees
    g1 = -(mfem*sfem + (mtib + mfoot)*lfem)*g*np.sin(q[0][0])
    g2 = -(mtib*stib+ mfoot*ltib)*g*np.sin(q[1][0])
    return np.array([[g1], [g2]])

def C(q, q_dot):
# Theta should be in degrees
    c12 = -(mtib*lfem*stib + mfoot*lfem*ltib)*np.sin(q[0][0] - q[1][0])*(q_dot[1][0])
    c21 = (mtib*lfem*stib + mfoot*lfem*ltib)*np.sin(q[0][0] - q[1][0])*(q_dot[0][0])
    return np.array([[0 , c12], [c21 , 0]])

def Rtot(q):
    RVlat = 0.05*np.cos(0.8*(q[1][0]-math.radians(80)))
    RBF = -0.038*np.cos(q[1][0]-math.radians(35))
    RRF_hip = 0.053*np.cos(0.8*(q[0][0]-math.radians(127)))
    RRF_knee = 0.053*np.cos(0.55*(q[1][0]-math.radians(70)))
    RST_hip = -0.07*np.cos(q[0][0]-math.radians(122))
    RST_knee = -0.04*np.cos(q[1][0]-math.radians(55))
    RIl = 0.045*np.cos(0.6*(q[0][0]-math.radians(140)))
    RGM = -0.06*np.cos(0.75*(q[0][0]-math.radians(90)))
    return np.array([[RIl, RGM, RRF_hip, RST_hip,0,0], [0,0,RRF_knee, RST_knee,RVlat,RBF]])

def M(q):
    Mfem = np.array([[0.112007,0,0], [0,0.106216,0], [0,0,0.0278432]])
    Mtib = np.array([[0.0591636,0,0], [0,0.0583531,0], [0,0,0.00590478]])
    m11 = Mfem[0,0] + mfem*(sfem**2) + (mtib + mfoot)*lfem**2
    m12 = (mtib*lfem*stib+mfoot*lfem*ltib)*np.cos(q[0][0]-q[1][0])
    m21 = m12
    m22 = Mtib[0,0] + mtib*(stib**2) + mfoot*(ltib**2)
    return np.array([[m11 , m12], [m21 , m22]])


def A_dot(A, U):
    #inter = 0.5+1.5*A
    tact = 0.05#0.01
    #tdeact = 0.05
    #tau = tact*inter + (U<A)*((tdeact /inter) - tact*inter)
    tau = tact
    return (U-A)/tau

def L_tot(q):
    LIl = 0.094 - 0.035*(q[0][0] - math.pi/2)
    LIl = LIl*sigmoid_lst(LIl) + (1-sigmoid_lst(LIl))*0.05
    #LGM = 0.127 + np.sqrt(sgm**2 +hlow**2 - 2*sgm*hlow*np.cos(q[0][0])) - np.sqrt(sgm**2 +hlow**2)
    LGM = 0.127 + 0.04*(q[0][0]- math.pi/2)
    LGM = LGM*sigmoid_lst(LGM) + (1-sigmoid_lst(LGM))*0.05#*0.01
    LRF = 0.06 + np.sqrt(lfem**2 +hsup**2 + 2*lfem*hsup*np.cos(q[0][0])) - np.sqrt(lfem**2 +hsup**2) - rknee*(q[1][0]-q[0][0])
    LRF = LRF*sigmoid_lst(LRF) + (1-sigmoid_lst(LRF))*0.05
    #LST = 0.055 + np.sqrt(d**2 +hlow**2 + 2*d*hlow*np.cos(alpha-q[0][0])) - np.sqrt(d**2 +hlow**2 + 2*d*hlow*np.sin(alpha)) + rknee*(q[1][0]-q[0][0])  ##
    LST = 0.055 + 0.05*(q[0][0]- math.pi/2) + rknee * (q[1][0] - q[0][0])
    LST = LST*sigmoid_lst(LST) + (1-sigmoid_lst(LST))*0.05##
    Lvlat = 0.046 + rknee*(q[0][0] - q[1][0])
    Lvlat = Lvlat*sigmoid_lst(Lvlat) + (1-sigmoid_lst(Lvlat))*0.05
    #Lbf = np.sqrt(sbf**2 + rknee**2 + 2*sbf*rknee*np.cos(betabf + q[0][0] - q[1][0])) - 0.091
    Lbf =  0.139 + rknee * (q[1][0] - q[0][0])
    Lbf = Lbf*sigmoid_lst(Lbf) +0.1# + 0.05
    return np.array([LIl, LGM, LRF, LST, Lvlat, Lbf])

def L_tot_dot(q , q_dot,L):
    LIldot = -rhip*q_dot[0][0]
    LIldot = sigmoid_lst(L[0])*LIldot
    #LGMdot = 2*sgm*hlow*np.sin(q[0][0])*q_dot[0][0] / (2*np.sqrt(sgm**2 +hlow**2 -2*sgm*hlow*np.cos(q[0][0])))
    LGMdot = 0.04*q_dot[0][0]
    LGMdot = sigmoid_lst(L[1])*LGMdot
    LRFdot = -2*lfem*hsup*np.sin(q[0][0])*q_dot[0][0]/(2*np.sqrt(lfem**2 + hsup**2 + 2*lfem*hsup*np.cos(q[0][0]))) - rknee*(q_dot[1][0] - q_dot[0][0])
    LRFdot = sigmoid_lst(L[2])*LRFdot
    #LSTdot = 2*d*hlow*np.sin(alpha-q[0][0])*q_dot[0][0]/(2*np.sqrt(d**2 +hlow**2 + 2*d*hlow*np.cos(alpha-q[0][0]))) + rknee*(q_dot[1][0]-q_dot[0][0])  ##
    LSTdot = 0.05*q_dot[0][0] + rknee * (q_dot[1][0] - q_dot[0][0])
    LSTdot = sigmoid_lst(L[3])*LSTdot
    Lvlatdot = rknee * q_dot[0][0] -rknee * q_dot[1][0]
    Lvlatdot = sigmoid_lst(L[4])*Lvlatdot
    #Lbfdot = -sbf*rknee*np.sin(betabf + q[0][0] - q[1][0])*(q_dot[0][0]-q_dot[1][0])/np.sqrt(sbf**2 + rknee**2 + 2*sbf*rknee*np.cos(betabf + q[0][0]- q[1][0]))
    Lbfdot = rknee * (q_dot[1][0] - q_dot[0][0])
    Lbfdot = sigmoid_lst(L[5])*Lbfdot
    return np.array([LIldot, LGMdot, LRFdot, LSTdot, Lvlatdot,Lbfdot])

def L_tot_theta(theta):  ##
    LIl = 0.094 - 0.035*(theta[0][0])
    LGM = 0.127 + np.sqrt(sgm**2 +hlow**2 + 2*sgm*hlow*np.sin(theta[0][0])) - np.sqrt(sgm**2 +hlow**2)
    LRF = 0.06 + np.sqrt(lfem**2 +hsup**2 - 2*lfem*hsup*np.sin(theta[0][0]))- np.sqrt(lfem**2 +hsup**2) - rknee*(theta[1][0])
    LST = 0.055 + np.sqrt(d**2 +hlow**2 + 2*d*hlow*np.sin(alpha-theta[0][0])) - np.sqrt(d**2 +hlow**2 + 2*d*hlow*np.sin(alpha)) + rknee*(theta[1][0])
    LST = LST*sigmoid_lst(LST) + 0.05
    Lvlat = 0.046 - rknee*(theta[1][0])
    Lbf = np.sqrt(sbf**2 + rknee**2 + 2*sbf*rknee*np.cos(betabf - theta[1][0])) - 0.091
    return np.array([LIl, LGM, LRF, LST, Lvlat, Lbf])

def fl(l):
    #return np.expand_dims(np.exp(np.abs((-1+np.float_power(l,1.55))/0.81)),1)
    return np.expand_dims(np.exp(-((l-1)**2)/0.45),1)


def fp(l):
    res = np.zeros((6, 1))
    for k in range (6):#[1,3]:
        res[k][0] = np.exp((5*(l[k]-1)/0.7)-1)/(np.exp(5)-1)
    return res


def fvl_tot(l,ldot):
    result = np.ones((len(l),1))
    vmax = 10
    Flenm = 1.4
    Af = 0.25
    for i in range (len(l)):
        if ldot[i] <=0:
            #val = (-ldot[i]-7.39)/(-7.39+(4.17*l[i]-3.21)*ldot[i])
            val = (ldot[i] + 0.25*vmax) / (0.25*vmax - ldot[i]/Af)
        else :
            #val = (0.62-(-3.12+4.21*l[i]-2.67*l[i]**2)*ldot[i])/(ldot[i]+0.62)
            val = (0.25*vmax*(Flenm - 1) + (2 + 2/Af)*Flenm*ldot[i]) / ( (2 + 2/Af)*ldot[i] + 0.25*vmax*(Flenm - 1) )
        result[i][0]=val
    return result

def F_tot(A, L, L_dot):
    # order : [LIl, LGM, LRF, LST, Lvlat,Lbf]
    Fmax = np.array([[1417] ,[1176], [730],[1580],[3120],[470]])
    alpha_ = np.array([[0.14],[0.0], [0.087],[0.0],[0.052],[0.4]]) # unit ==
    Lopt = np.array([0.102,0.158,0.112,0.109,0.104,0.177])
    return A*Fmax*fvl_tot(L/Lopt, L_dot/Lopt)*fl(L/Lopt)*np.cos(alpha_) + Fmax*fp(L/Lopt)*np.cos(alpha_) #???


def sigmoid(theta, factor1,factor2): # THE ARGUMENT PROVIDED TO THIS FUNCTION IS INDEED THETA (NOT Q) IN DEGREES
    # sigmoid to constraint theta_dot at theta  min and max
    theta_hip = sigmoid_hip(theta, factor1)[0][0]
    hmax = np.sin(np.radians(theta_hip))*lfem
    min_hip = hip0
    max_hip = 120
    min_knee = -np.minimum(theta_hip+np.degrees(np.arcsin(hmax/ltib)),120)
    max_knee = 10
    sig = np.array([[(1/(1+np.exp(-factor1*(theta[0][0] -(min_hip)))))-(1/(1+np.exp(-factor1*(theta[0][0]-(max_hip)))))],[(1/(1+np.exp(-factor2*(theta[1][0]-(min_knee)))))-(1/(1+np.exp(-factor2*(theta[1][0]-(max_knee)))))]])  ### -(-15)
    return sig


def sigmoid_min(theta, factor1,factor2): # THE ARGUMENT PROVIDED TO THIS FUNCTION IS INDEED THETA (NOT Q) IN DEGREES
    # sigmoid to balance gravity and ground reaction forces (knee on pillow + feet on table)
    theta_hip = sigmoid_hip(theta, factor1)[0][0]
    hmax = np.sin(np.radians(theta_hip))*lfem
    min_hip = hip0
    min_knee = -np.minimum(theta_hip+np.degrees(np.arcsin(hmax/ltib)),120)
    sig = np.array([[(1/(1+np.exp(-factor1*(theta[0][0] -(min_hip)))))],[(1/(1+np.exp(-factor2*(theta[1][0]-(min_knee)))))]])  ### -(-15)
    return sig


def sigmoid_hip(theta, factor1): # THE ARGUMENT PROVIDED TO THIS FUNCTION IS INDEED THETA (NOT Q) IN DEGREES
    # first compute theta hip to define min theta knee
    min_hip = hip0
    max_hip = 120
    sig = theta[0][0]*(1/(1+np.exp(-factor1*(theta[0][0] -(min_hip))))-(1/(1+np.exp(-factor1*(theta[0][0]-(max_hip))))))
    sig_min = (min_hip)*(1-1/(1+np.exp(-factor1*(theta[0][0] -(min_hip)))))
    sig_max = (max_hip)*(1/(1+np.exp(-factor1*(theta[0][0]-(max_hip)))))
    return np.array([[sig + sig_min + sig_max], [theta[1][0]]])


def sigmoid_theta(theta, factor1, factor2): # THE ARGUMENT PROVIDED TO THIS FUNCTION IS INDEED THETA (NOT Q) IN DEGREES

    theta = sigmoid_hip(theta, factor1)  # first compute theta hip to define min theta knee
    hmax = np.sin(np.radians(theta[0][0]))*lfem
    min_knee = -np.minimum(theta[0][0]+np.degrees(np.arcsin(hmax/ltib)),120)
    max_knee = 10
    sig = theta[1][0]*(1/(1+np.exp(-factor2*(theta[1][0] -(min_knee))))-(1/(1+np.exp(-factor2*(theta[1][0]-(max_knee))))))
    sig_min = (min_knee)*(1-1/(1+np.exp(-factor2*(theta[1][0] -(min_knee)))))
    sig_max = (max_knee)*(1/(1+np.exp(-factor2*(theta[1][0]-(max_knee)))))
    return np.array([[theta[0][0]], [sig + sig_min + sig_max]])


def sigmoid_lst(lst):
    return (1-0.05)/(1+np.exp(-100*(lst-0.07)))


def update_tot(theta_, theta_dot, A_, U, dt):
    hhip = 0.1 ### 1
    lpelv = 0.16
    A_ = U
    """## discontinuity
    if (hhip - lfem*np.cos(theta_[0][0]))<(hhip + hp):  # knee height  ## (hhip)
        theta_[0][0]= np.arccos(-hp/lfem)  ## np.arcsin(hp/lfem) + np.pi/2 same

    if (hhip - lfem*np.cos(theta_[0][0]) - ltib*np.cos(theta_[1][0]))< hhip : # ankle height  ## (hhip)
        hmax = np.sin(theta_[0][0]-np.pi/2)*lfem
        print("knee update ", np.degrees(theta_[1][0]-theta[0][0]))
        theta_[1][0]= -np.arcsin(hmax/ltib)+np.pi/2  ### np.arccos((-lfem*np.cos(theta_[0][0]))/ltib)
        print(np.degrees(theta_[1][0]-theta[0][0]))"""

    theta_new = theta_+ dt*theta_dot

    ### theta degrees
    theta_deg = np.array([[0.0],[0.0]])
    theta_deg[0][0] = np.degrees(theta_[0][0]-np.pi/2)
    theta_deg[1][0] = np.degrees(theta_[1][0]-theta_[0][0])

    ### Sigmoid theta0
    theta_deg_new = np.array([[0.0],[0.0]])
    theta_deg_new[0][0] = np.degrees(theta_new[0][0]-np.pi/2)
    theta_deg_new[1][0] = np.degrees(theta_new[1][0]-theta_new[0][0])
    factor1 = 1000
    factor2 = 1000
    theta_deg_sig = sigmoid_theta(theta_deg_new, factor1, factor2)
    theta_new[0][0] = math.radians(theta_deg_sig[0][0]) + np.pi/2
    theta_new[1][0] = theta_new[0][0] + math.radians(theta_deg_sig[1][0])

    #check Z

    #if (hhip - lfem*np.cos(theta_new[0][0]))<(hhip+hp) : # knee height
    #    theta_new[0][0]= np.arccos(-hp/lfem)

    #if (  hhip - lfem*np.cos(theta_new[0][0]) - ltib*np.cos(theta_new[1][0]))<hhip : # ankle height
    #    theta_new[1][0]= np.arccos((-lfem*np.cos(theta_new[0][0]))/ltib)

    #theta_ = theta_new
    L = L_tot(theta_)
    L_dot = L_tot_dot(theta_, theta_dot,L)
    A_new = A_ #+ dt*A_dot(A_,U)
    #theta_new = theta_+ dt*theta_dot
    M_inv = np.linalg.inv(M(theta_))

    hihi = Rtot(theta_)[0] *F_tot(A_,L,L_dot).T
    #phi = (M_inv@( Rtot(theta_)@F_tot(A_,L,L_dot)  + C(theta_, theta_dot)@theta_dot + G(theta_) ))
    #theta_dot_new = theta_dot + dt*(M_inv@( Rtot(theta)@F_tot(A_,L,L_dot)  + C(theta, theta_dot)@theta_dot + G(theta) ))#)*sigmoid(np.degrees(theta)-90))+ theta_dot*(sigmoid(np.degrees(theta)-90)+np.array([[-1],[0]]))
    #theta_dot_new = theta_dot + dt*phi+ dt*(1-sigmoid(np.degrees(theta)-90))*(-phi-theta_dot/dt)
    #good one
    theta_dot_new = theta_dot + dt*M_inv@( Rtot(theta_)@F_tot(A_,L,L_dot)  + C(theta_, theta_dot)@theta_dot + G(theta_) +(C(theta_, theta_dot)@theta_dot + G(theta_))*(sigmoid_min(theta_deg, sigmoid_factor1, sigmoid_factor2)-1)) +theta_dot*(sigmoid(theta_deg, sigmoid_factor1, sigmoid_factor2)-1)

    #no coriolis
    #theta_dot_new = theta_dot + dt*M_inv@( Rtot(theta_)@F_tot(A_,L,L_dot)  + G(theta_) +( G(theta_))*(sigmoid(np.degrees(theta_)-90, sigmoid_factor1, sigmoid_factor2)-1)) +theta_dot*(sigmoid(np.degrees(theta_)-90,sigmoid_factor1, sigmoid_factor2)-1)

    #theta_dot_new[0][0]=0
    return A_new, theta_new, theta_dot_new, hihi, F_tot(A_,L,L_dot).T


if test_plot:  ##
    # test muscle length and det M
    # thetat
    hip_range = [-20, 120]
    knee_range = [-120, 10]
    hip_angle = np.linspace(hip_range[0], hip_range[1], 100)*3.14/180
    knee_angle = np.linspace(knee_range[0], knee_range[1], 100)*3.14/180
    # q
    qhip_angle = hip_angle + math.pi/2
    qknee_angle = np.zeros((100, 100))
    for h in range(len(qhip_angle)):
        qknee_angle[h, :] = qhip_angle[h] + knee_angle
    Ltheta = np.zeros((6, 100, 100))
    Lq = np.zeros((6, 100, 100))
    detM = np.zeros((100, 100))
    for h in range(len(qhip_angle)):
        for k in range(len(qknee_angle)):
            Ltheta[:, k, h] = L_tot_theta([[hip_angle[h]], [knee_angle[k]]])
            Lq[:, k, h] = L_tot([[qhip_angle[h]], [qknee_angle[h,k]]])
            detM[k, h] = np.linalg.det(np.array(M([[qhip_angle[h]], [qknee_angle[h,k]]])))

    # plots
    Xt, Yt = np.meshgrid(hip_angle, knee_angle)
    Xq, Yq = np.meshgrid(qhip_angle, knee_angle)
    for m in range(len(muscle_names)):
        # theta
        """fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot_surface(Xt*180/3.14, Yt*180/3.14, Ltheta[m])
        ax.set_title(muscle_names[m] + " - Theta")
        ax.set_xlabel('Theta hip (°)')
        ax.set_ylabel('Theta knee (°)')"""
        # q
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot_surface(Xq*180/3.14, Yq*180/3.14, Lq[m])
        ax.set_title(muscle_names[m] + " - q")
        ax.set_xlabel('q hip (°)')
        ax.set_ylabel('Theta knee (°)')

    # det M
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(Xq*180/3.14, Yq*180/3.14, detM)
    ax.set_title("det M")
    ax.set_xlabel('q hip (°)')
    ax.set_ylabel('Theta knee (°)')

    plt.show()

if simulation:
    thetadot0 = np.array([[0.0],[0.0]])  ###[0],[0]
    # order : [LIl, LGM, LRF, LST, Lvlat,Lbf]
    A0 = np.array([[0.0],[0.0],[0.0],[0.0],[0.0],[0.0]])  ### idem
    A0[:,0] = U[:,0]

    joint = []
    joint_deriv = []
    joint_derivhip = []
    A1_list = []
    A2_list = []
    c = []
    g_= []
    hipp= []
    f = []
    big_term = []
    lesL = []
    lesLdot = []
    lesF = []
    theta = theta0 # np.array([[math.radians(hip0)],[math.radians(knee0)]])
    theta_dot = thetadot0
    A = A0
    q = np.array([[theta[0][0] + math.pi/2],[theta[0][0] + theta[1][0]+ math.pi/2 ]])   ### no abs(theta[.][0]) and + instead of -
    #q = np.array([[theta[0][0] + math.pi/2],[ math.pi/2- np.arcsin(hp/ltib)]])
    print("qknee: ", np.degrees(q[1][0]))
    q_dot = np.array([[theta_dot[0][0]],[theta_dot[0][0] + theta_dot[1][0]]])
    muscle_length =L_tot(q)
    muscle_force =muscle_length
    simpleforce = muscle_length

    A_array = np.zeros((6, len(time)))  ##
    for i in range(U.shape[1]):
        #print(i)
        #q = np.maximum(q, [[math.radians(80)],[math.radians(-20)]])
        #q = np.minimum(q, [[math.radians(210)],[math.radians(220)]])
        muscle_length = np.vstack((muscle_length, L_tot(q)))
        A , q , q_dot, hihi , f = update_tot(q, q_dot, A, U[:,i:i+1], dt)
        muscle_force =np.vstack((muscle_force, hihi))
        simpleforce =np.vstack((simpleforce, f))

        joint.append(q[1][0])
        #big_term.append(a[1][0])
        hipp.append(q[0][0])
        joint_deriv.append(q_dot[1][0])
        joint_derivhip.append(q_dot[0][0])
        print(np.shape(A))
        for m in range(len(muscle_names)):  ##
            A_array[m, i] = A[m][0]

        c.append((C(q, q_dot)@q_dot)[0][0])
        g_.append((G(q))[0][0])
    Lopt = np.array([0.102,0.158,0.112,0.109,0.104,0.177])
    # plot joints (theta)
    """fig, ax = plt.subplots(2, 1, figsize= (10,6))
    ax[0].plot(time,np.rad2deg(np.array(joint)-np.array(hipp))-(knee0), label = 'knee')  ##
    ax[0].plot(time,np.rad2deg(np.array(hipp))-90-(hip0), label = 'hip')  ##
    ax[0].set_title('Joint position')
    ax[0].set_ylim([-60,60])
    #ax[1].set_title('Joint velocity')
    ax[1].plot(time, joint_deriv)
    ax[1].plot(time, joint_derivhip)
    #ax[1].plot(time, c,color = 'r')
    #ax[1].plot(time, g_, linewidth= 3,color = 'g')
    #ax[1].plot(time,muscle_length[1:,:2]/Lopt[:2])
    #ax[1].plot(time, big_term, linewidth= 1)
    #ax[1].legend(['Il', 'GM'])#, 'RF', 'ST', 'Vlat','Bf'])
    #ax[3].plot(time,muscle_force[1:,:2])
    #ax[1].plot(time, big_term, linewidth= 1)
    #ax[2].plot(time,simpleforce[1:,:2])
    plt.legend()
    plt.tight_layout()
    f = result_folder +"theta.png"
    fig.savefig(f)"""

    # Plot JointsU.shap
    ax_col = '#585759'
    fig, ax = plt.subplots(1, figsize=(13, 4),frameon = False)
    plt.plot(np.linspace(0,1000,len(joint)) , np.rad2deg(np.array(joint)-np.array(hipp))-(knee0),linewidth=3, color = '#19D3C5', label ='Prediction - Knee')
    plt.plot(np.linspace(0,1000,len(joint)),ang[:,1], color = '#19D3C5', linewidth=6, alpha = 0.3, label ='Ground truth - Knee')
    plt.plot(np.linspace(0,1000,len(joint)) , np.rad2deg(np.array(hipp))-90-(hip0),linewidth=3, color = '#FA525B', label ='Prediction - Hip')
    plt.plot(np.linspace(0,1000,len(joint)),ang[:,0], color = '#FA525B', linewidth=6, alpha = 0.3, label ='Ground truth - Hip')
    #plt.plot(np.rad2deg(np.array(hipp))-90-(hip0),color = 'mediumaquamarine' , linewidth=1)
    #plt.plot(ang[:,0],color = 'mediumaquamarine', linewidth=3, alpha = 0.3)
    for pos in ['right', 'top']:
        plt.gca().spines[pos].set_visible(False)
    plt.gca().spines['left'].set_color(ax_col)
    plt.gca().spines['bottom'].set_color(ax_col)
    plt.tick_params(axis='both', colors = ax_col)
    plt.ylabel('Angle [degree]', color = ax_col)
    #plt.title('Knee joint angle', color = ax_col)
    plt.xlabel('Time [ms]',color = ax_col)
    plt.ylim([-100,100])
    plt.legend()
    plt.tight_layout()
    f = result_folder + "reportjoint.png"
    print(f)
    fig.savefig(f , transparent = True)
    # Plot Activation
    fig, ax = plt.subplots(1, 6, figsize=(13, 2.5),frameon = False)  #plt.subplots(1, figsize=(13, 1.5))
    for m in range(len(muscle_names)):
        ax[m].set_title(muscle_names[m]  , color = ax_col, fontsize = "x-large")
        ax[m].plot(np.linspace(0,1000,np.shape(U)[1]), U[m, :], color='#7E7B7D', linewidth=1)
        for pos in ['right', 'top']:
            ax[m].spines[pos].set_visible(False)
        ax[m].spines['left'].set_color(ax_col)
        ax[m].spines['bottom'].set_color(ax_col)
        ax[m].set_ylabel(' ', color = ax_col)
        ax[m].set_xlabel('Time [ms]',color = ax_col)
        ax[m].set_ylim([-0.05,1])
        ax[m].tick_params(axis='both', colors = ax_col)
    ax[0].set_ylabel('Activation [a.u.]', color = ax_col)

    plt.tight_layout()
    f = result_folder + "stimreport.png"
    print(f)
    fig.savefig(f , transparent = True)


    fig, ax = plt.subplots(1, figsize=(13, 2))
    #plt.plot(np.rad2deg(np.array(joint)-np.array(hipp))-(-80), color = 'orchid', linewidth=1)
    #plt.plot(np.rad2deg(np.array(hipp))-90-(-10),color = 'mediumaquamarine' , linewidth=1)
    plt.plot(np.rad2deg(np.array(joint)-np.array(hipp))-(knee0), color = 'orchid', linewidth=1)
    plt.plot(ang[:,1], color = 'orchid', linewidth=3, alpha = 0.3)
    plt.plot(np.rad2deg(np.array(hipp))-90-(hip0),color = 'mediumaquamarine' , linewidth=1)
    plt.plot(ang[:,0],color = 'mediumaquamarine', linewidth=3, alpha = 0.3)
    #plt.ylim([-80,80])
    plt.tight_layout()
    f = result_folder + "joint.png"
    print(f)
    fig.savefig(f) #, transparent = True)

    # Plot Activation
    fig, ax = plt.subplots(3, 2, figsize=(13, 10))  #plt.subplots(1, figsize=(13, 1.5))
    for m in range(len(muscle_names)):
        ax[m%3][m//3].set_title("Stim "+muscle_names[m])
        ax[m%3][m//3].plot(U[m, :], color='#fa525b', linewidth=1)
    """plt.plot(np.linspace(1.1, 1.7,nbpt), U[1, :], color='#fa525b', linewidth=1)
    plt.plot(np.linspace(2.2, 2.8,nbpt), U[2, :], color='#fa525b', linewidth=1)
    plt.plot(np.linspace(3.3, 3.9,nbpt), U[3, :], color='#fa525b', linewidth=1)
    plt.plot(np.linspace(4.4, 5,nbpt), U[4, :], color='#fa525b', linewidth=1)
    plt.plot(np.linspace(5.5, 6.1,nbpt), U[5, :], color='#fa525b', linewidth=1)
    plt.ylim([-0.05,0.25])"""
    plt.tight_layout()
    f = result_folder + "stim.png"
    print(f)
    fig.savefig(f) #, transparent = True)

    fig, ax = plt.subplots(3, 2, figsize=(13, 10))
    for m in range(len(muscle_names)):
        ax[m%3][m//3].plot(time, A_array[m,:])
        ax[m%3][m//3].set_title('Activation '+muscle_names[m])
    plt.tight_layout()
    f = result_folder + "activation.png"
    fig.savefig(f)

    def compute_cool(knee, hip):
        knee = np.array(knee)
        hip = np.array(hip)
        hhip = 1
        lpelv = 0.16
        hip_coo = lpelv*np.ones((len(knee), 3))/2
        knee_coo = lpelv*np.ones((len(knee), 3))/2
        ank_coo = lpelv*np.ones((len(knee), 3))/2
        hip_coo[:,1] = 0
        knee_coo[:,1] = lfem*np.sin(hip)
        ank_coo[:,1] = lfem*np.sin(hip) + ltib*np.sin(knee)
        hip_coo[:,2] = hhip
        knee_coo[:,2] = hhip - lfem*np.cos(hip)
        ank_coo[:,2] = hhip - lfem*np.cos(hip) - ltib*np.cos(knee)
        return hip_coo, knee_coo , ank_coo

    hip_coo, knee_coo , ank_coo = compute_cool(joint, hipp)
    """fig, ax = plt.subplots(1,1)
    plt.plot(np.linspace(1,np.shape(ank_coo[:,2])[0],np.shape(ank_coo[:,2])[0]),hip_coo[:,2])
    plt.plot(np.linspace(1,np.shape(ank_coo[:,2])[0],np.shape(ank_coo[:,2])[0]),knee_coo[:,2])
    plt.plot(np.linspace(1,np.shape(ank_coo[:,2])[0],np.sha sig + sig_min + sig_maxpe(ank_coo[:,2])[0]),ank_coo[:,2])
    fig.savefig(result_folder+'elsa.png')"""

    fig, ax = plt.subplots(1,1)
    ax = plt.subplot(111,projection='3d',computed_zorder=False)
    numDataPoints = len(hipp)-1
    ax.set_xlim3d([-0.3, 0.5])
    ax.set_ylim3d([-0.3, 0.5])
    ax.set_zlim3d([0.7, 1.5])
    #ax.view_init(20, -25)
    ax.view_init(0, -25)


    xxx = np.concatenate((np.expand_dims(hip_coo,axis = 0 ) , np.expand_dims(knee_coo,axis = 0 ), np.expand_dims(ank_coo,axis = 0 )), axis=0)
    print(np.shape (xxx))
    ang = np.concatenate((np.expand_dims(np.array(hipp),axis = 0 ), np.expand_dims(np.array(joint),axis = 0 )), axis=0)
    print(np.shape(ang))

    with open('Leftlegang.npy', 'wb') as f:
        np.save(f,ang, allow_pickle=False)
    with open('Leftlegcoo.npy', 'wb') as f:
        np.save(f,xxx, allow_pickle=False)

    def animate_func(num):
        ax.clear()
        ax.plot3D([hip_coo[num,0],knee_coo[num,0],ank_coo[num,0]],[hip_coo[num,1],knee_coo[num,1],ank_coo[num,1]], [hip_coo[num,2],knee_coo[num,2],ank_coo[num,2]], color='#fa525b',antialiased=True, linewidth=12,fillstyle='full')
        ax.set_xlim3d([-0.3, 0.5])
        ax.set_ylim3d([-0.3, 0.5])
        ax.set_zlim3d([0.7, 1.5])
    line_ani = animation.FuncAnimation(fig, animate_func, interval=3,frames=int(numDataPoints))
    plt.tight_layout()
    plt.show()
    f = result_folder + "animation.gif"
    writergif = animation.PillowWriter(fps=30)
    line_ani.save(f, writer=writergif)
