#data files are numbered on the server.
#for exmaple imuRaw1.mat, imuRaw2.mat and so on.
#write a function that takes in an input number (1 through 6)
#reads in the corresponding imu Data, and estimates 
#roll pitch and yaw using an extended kalman filter

from scipy import io
# from scipy import integrate
import os
import matplotlib.pyplot as plt
import numpy as np
import math
# from scipy.spatial.transform import Rotation as R
from ukf import UKF
import time

def plot_time_delta_imu(list_name, max_range = 100):
    delta_t_imu = []
    for i in range(100):
        t1 = list_name[0][:,i][0]
        t2 = list_name[0][:,i+1][0]
        delta_t_imu.append(t2-t1)
    plt.plot(delta_t_imu, linewidth=1.0)  
    plt.savefig('time_delta_imu_1.png')

def plot_angles(ls, fname):
    plt.plot(ls, linewidth = 1.0)
    plt.savefig(fname)
    plt.cla()

def plot_angles_multiple(ls2, ls3, fname):
    # plt.plot(ls1, linewidth = 1.0, color='green')
    plt.plot(ls2, linewidth = 1.0, color='blue')  
    plt.plot(ls3, linewidth = 1.0, color='red')  
    plt.savefig(fname)
    plt.cla()

def is_rotation(rot_mat):
    n = rot_mat.shape[0]
    identity = np.identity(n)    
    if np.abs(np.linalg.norm((rot_mat.T @ rot_mat - identity))) < 1e-6:
        return True
    return False

# https://www.learnopencv.com/rotation-matrix-to-euler-angles/
def rot_to_euler(rot_mat):
    assert(is_rotation(rot_mat))
    sy = math.sqrt(rot_mat[0,0] * rot_mat[0,0] +  rot_mat[1,0] * rot_mat[1,0])
    singular = sy < 1e-6
    if  not singular :
        x = math.atan2(rot_mat[2,1] , rot_mat[2,2])
        y = math.atan2(-rot_mat[2,0], sy)
        z = math.atan2(rot_mat[1,0], rot_mat[0,0])
    else :
        x = math.atan2(-rot_mat[1,2], rot_mat[1,1])
        y = math.atan2(-rot_mat[2,0], sy)
        z = 0
    return np.array([x, y, z])

# def scipy_rot_eular(rot_mat):
#     r = R.from_matrix(rot_mat)
#     e = r.as_euler('xyz')
#     return e
 
# https://theccontinuum.com/2012/09/24/arduino-imu-pitch-roll-from-accelerometer/
def pitch_calc(acc):
    # return math.atan2(-acc[1], math.sqrt(acc[0]**2 + acc[2]**2))
    return math.atan2(acc[0], math.sqrt(acc[1]**2 + acc[2]**2))

def roll_calc(acc):
    return math.atan2(-acc[1],acc[2]) #used -acc[0], acc[2] before

def gyro_to_eular(angles, delta):
    return 1

# def digital_to_analog():
def clean_data_imu(readings):
    data = readings
    data = data.astype(float)
    #sensitivty for imu from data sheet 330
    imu_scale_factor = 3300/(34.7*1023)
    #sensitivity non amplified
    gyro_scale_factor = (3300*math.pi)/(1023*180*3.7)
    # gyro_scale_factor = (3300)/(1023*390)
    bias_imu = np.array([[511, 501, 500]])
    bias_gyro = np.array([373.7, 375.8, 369.8])
    print('Imu bias - ', bias_imu)
    print('Gyro Bias - ', bias_gyro)
    data[:, 0:3] = (data[:, 0:3] - bias_imu) * imu_scale_factor
    data[:, 3:6] = (data[:, 3:6] - bias_gyro) * gyro_scale_factor
    return data


def estimate_rot(data_num=1):
	#your code goes here
    vals_data = []
    ts_data = []
    rot_mat = []
    vicon_ts = []
    number  = data_num
    filename = os.path.join(os.path.dirname(__file__), "imu/imuRaw" + str(number) + ".mat")
    imuRaw = io.loadmat(filename)
    trim_imu = {k:imuRaw[k] for k in ['vals', 'ts']}
    trim_imu['vals'] = np.rollaxis(trim_imu['vals'], 1)
    trim_imu['ts'] =  np.rollaxis(trim_imu['ts'], 1)
    trim_imu['vals'][:, [3, 5]] = trim_imu['vals'][:, [5, 3]]
    trim_imu['vals'][:, [3, 4]] = trim_imu['vals'][:, [4, 3]]
    trim_imu['vals'] = trim_imu['vals'].astype(float)
    vals_data.append(trim_imu['vals'])
    ts_data.append(trim_imu['ts'])
    
    file1 = os.path.join(os.path.dirname(__file__), "vicon/ViconRot" + str(number) + ".mat")
    viconRaw = io.loadmat(file1)

    viconRaw['rots'] = np.rollaxis(viconRaw['rots'],2)
    viconRaw['ts'] = np.rollaxis(viconRaw['ts'],1)
    rot_mat.append(viconRaw['rots'])
    vicon_ts.append(viconRaw['ts'])
    
    imu_data_new = []
    for data in vals_data:
        cleaned_data = clean_data_imu(data)
        imu_data_new.append(cleaned_data)
    
    # print(imu_data_new[0][:20, :3])
    imu_data_new[0][:,0] = -imu_data_new[0][:,0]
    imu_data_new[0][:,1] = -imu_data_new[0][:,1]
    # print(imu_data_new[0][:20, :3])
    delta_ts = []
    for i, timer in enumerate(ts_data):
        ts = np.diff(timer, axis = 0)
        ts = np.concatenate([np.array([0]).reshape(1,1), ts])
        delta_ts.append(ts)
    
    # #plot roll and pitch
    # for num, (data, rot) in enumerate(zip(imu_data_new, rot_mat)):
    #     pitch_val = []
    #     roll_val = []
    #     gyrox, gyroy = [0], [0]
    #     for i in range(data.shape[0]):
    #         acc = data[i][0:3]
    #         roll_val.append(roll_calc(acc))
    #         pitch_val.append(pitch_calc(acc))
    #         n = gyrox[i] + imu_data_new[0][i,3]*delta_ts[0][i]
    #         gyrox.append(n)
            
    #     # gyrox = integrate.cumtrapz(imu_data_new[0][:,3].reshape(delta_ts[0].shape), delta_ts[0], axis = 0)
    #     # gyrox = integrate.cumtrapz(imu_data_new[0][:,4].reshape(delta_ts[0].shape), delta_ts[0], axis = 0)   
        
    #     eular = np.zeros((rot.shape[0], 3))
    #     for i in range(rot.shape[0]):
    #         angles = rot_to_euler(rot[i])
    #         eular[i] = angles
         
        # gyrox = np.cumsum(imu_data_new[0][:,3], axis = 0)
        # print(imu_data_new[0][:,3], eular[:-1,0])   
        # plot_angles_multiple(gyrox, eular[:,0], 'roll_%d_gyro.png'%num)
        # plot_angles_multiple(imu_data_new[0][:,4], eular[:,1], 'pitch_%d_gyro.png'%num)
        # plot_angles_multiple(roll_val, eular[:,0], 'roll_%d_acc.png'%num)
        # plot_angles_multiple(pitch_val, eular[:,1], 'pitch_%d_acc.png'%num)
        
    roll, pitch, yaw =  UKF(data, delta_ts[0])
    roll, pitch, yaw = np.array(roll), np.array(pitch), np.array(yaw)
    
    
    rot = rot_mat[0]
    eular = np.zeros((rot.shape[0], 3))
    for i in range(rot.shape[0]):
        angles = rot_to_euler(rot[i])
        eular[i] = angles
    
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(roll, linewidth = 1.0, color='green')
    plt.plot(eular[:,0], linewidth = 1.0, color='blue')
    
    plt.subplot(3,1,2) 
    plt.plot(pitch, linewidth = 1.0, color='green')
    plt.plot(eular[:,1], linewidth = 1.0, color='blue')
    
    plt.subplot(3,1,3)
    plt.plot(yaw, linewidth = 1.0, color='green')
    plt.plot(eular[:,2], linewidth = 1.0, color='blue')
    plt.savefig('final_answer_100andavg0.001.png')
    # roll, pitch, yaw = [],[],[]
    return roll,pitch,yaw

start = time.time()	
estimate_rot()
end = time.time()
print('time to execute: ', end - start)