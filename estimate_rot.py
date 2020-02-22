from scipy import io
import os
import matplotlib.pyplot as plt
import numpy as np
import math
from ukf import UKF
import time
 
def clean_data_imu(data):
    # data = data.astype(float)
    imu_scale_factor = 3300/(34.5*1023)
    gyro_scale_factor = (3300*math.pi)/(1023*180*3.7)
    bias_imu = np.array([[511, 501, 500]])
    bias_gyro = np.array([373.7, 375.8, 369.8])
    data[:, 0:3] = (data[:, 0:3] - bias_imu) * imu_scale_factor
    data[:, 3:6] = (data[:, 3:6] - bias_gyro) * gyro_scale_factor
    return data

def estimate_rot(data_num=1):
    vals_data = []
    ts_data = []
    filename = os.path.join(os.path.dirname(__file__), "imu/imuRaw" + str(data_num) + ".mat")
    imuRaw = io.loadmat(filename)
    trim_imu = {k:imuRaw[k] for k in ['vals', 'ts']}
    trim_imu['vals'] = np.rollaxis(trim_imu['vals'], 1)
    trim_imu['ts'] =  np.rollaxis(trim_imu['ts'], 1)
    trim_imu['vals'][:, [3, 5]] = trim_imu['vals'][:, [5, 3]]
    trim_imu['vals'][:, [3, 4]] = trim_imu['vals'][:, [4, 3]]
    trim_imu['vals'] = trim_imu['vals'].astype(float)
    vals_data.append(trim_imu['vals'])
    ts_data.append(trim_imu['ts'])
    
    imu_data_new = []
    for data in vals_data:
        cleaned_data = clean_data_imu(data)
        imu_data_new.append(cleaned_data)
    
    imu_data_new[0][:,0] = -imu_data_new[0][:,0]
    imu_data_new[0][:,1] = -imu_data_new[0][:,1]
    delta_ts = []
    ts = np.diff(ts_data[0], axis = 0)
    delta_ts = np.concatenate([np.array([0]).reshape(1,1), ts])
    
    ###UKF
    roll, pitch, yaw =  UKF(imu_data_new[0], delta_ts)
    return roll,pitch,yaw