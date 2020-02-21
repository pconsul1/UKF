#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 14:48:36 2020

@author: poojaconsul
"""
import math
import numpy as np
from  quaternion import Quaternion
import matplotlib.pyplot as plt
    
def rot_to_quat(w): 
    w = np.array(w)
    norm = np.linalg.norm(w)
    if(norm == 0):
        # print('norm is 0 in rot_to_quat function')
        return Quaternion(0, [0,0,0], "value")
        
    axis = w/norm
    q1 = Quaternion(norm, axis, "angle")
    q1 = q1.unit_quaternion()
    return q1
    
#verified
def quat_to_rot(q):
    q = Quaternion(q[0], q[1:4], "value")
    
    if(q.norm() == 0):
        return np.array([0,0,0])
    
    q = q.unit_quaternion()
    y = np.array(q.quaternion)
    q0 = y[0]
    angle = math.acos(q0)
    
    if(angle == 0):
        return np.array([0,0,0])
        # return np.array([y[1],y[2],y[3]])
    
    sin_angle = math.sin(angle)
    angular = (2*angle*y[1:4])/sin_angle #unit vector of angular velocity
    return angular


def quat_average(quaternions, current_state_quaternion, thresh = 0.005): #verified
    estimated_mean = current_state_quaternion
    counter = 0   
    while(True):
        counter = counter+1
        ei_s = np.zeros((3,len(quaternions)))
        # ei_s = np.zeros((len(quaternions),3))
        for i in range(len(quaternions)):
         
            ei = quaternions[i].multiply(estimated_mean.inverse())
            # print("\n\nei ",ei.quaternion)
            ei_rotvec = quat_to_rot(ei.quaternion)
            # ei_s[i] = ei_rotvec
            ei_s[:,i] = ei_rotvec
            # print(ei_s[:,i])
            # print(ei_rotvec)
            
        # e = np.mean(ei_s,axis=0)
        e = np.mean(ei_s,axis=1)
        
        # print("e ", e)
        
        e_quaternion = rot_to_quat(e)
        
        # print(e_quaternion.quaternion)
        
        estimated_mean = e_quaternion.multiply(estimated_mean)  
        
        # print('estimate mean \n',estimated_mean.quaternion)
        
        
        if(np.linalg.norm(e)< thresh or counter>10):
            break
    # print('steps to converge ',counter)
    
    # return estimated_mean, ei_s, counter
    return estimated_mean, ei_s.T, counter

# def plot_angles(ls, fname):
#     plt.plot(ls, linewidth = 1.0)
#     plt.savefig(fname)
#     plt.cla()

#verified
def sigma_pts(P, Q):
    p_plus_q = P+Q
    
    ###PSD fix
    eigs = np.linalg.eigvals(p_plus_q)
    if not(np.all(eigs) > 0):
        p_plus_q = p_plus_q - np.min(eigs)*np.identity(6)
        p_plus_q/=p_plus_q[1,1]
        
    n = P.shape[0]
    l = np.linalg.cholesky(p_plus_q)
    # l = l/np.linalg.norm(l)
    left_vec = l*np.sqrt(2*n)
    right_vec = -l*np.sqrt(2*n)
    w_i = np.hstack((left_vec, right_vec)).T
    return w_i
        
#verified
def covariance(data):
    #data is n*6 col vectors rvs
    data = data.T
    mean = np.average(data.T, axis = 0)
    cov = ((data - mean[:, None]) @ (data - mean[:, None]).T)/(data.shape[1])
    return cov

def process_model(w_i, x_i_prev, delta_ts):
    w_prev_est = x_i_prev[4:7]
    quat_prev_est = Quaternion(x_i_prev[0], x_i_prev[1:4], "value")
    
    # print('w prev est ', w_prev_est)
    y_i = np.zeros((w_i.shape[0], w_i.shape[1]+1))
    for i in range(w_i.shape[0]):
        w_i_quat = rot_to_quat(w_i[i][0:3])

        x_i_quat = quat_prev_est.multiply(w_i_quat)
        x_i_w = w_prev_est + w_i[i][3:6]
        x_i_w_norm = np.linalg.norm(x_i_w)
        angle = x_i_w_norm * delta_ts
        axis = x_i_w/x_i_w_norm
        q_delta = Quaternion(angle, axis, "angle")
        y_i_quat = x_i_quat.multiply(q_delta)
        y_i[i] = np.concatenate([np.array(y_i_quat.quaternion), x_i_w], axis = 0)
    return y_i
    
def measure_acc(prev_q_wt, delta_t):
    """
    Attributes
    -------
    data(array) : a 1*6 array with accelerometer and gyroscope readings in order [ax, ay, az, wz, wx, wy]
    prev_q_wt(Quaternion) : previous computed quaternion for angular velocity
    delta_t(int) : time interval for angular rotation
    Returns
    -------
    acc(Quaternion) : a quaternion for acceleration

    """
    g_quat = Quaternion(0, [0, 0, 9.8], "value")
    prev_q_wt_inverse = prev_q_wt.inverse()
    next_q_wt = prev_q_wt_inverse.multiply(g_quat)
    q_acc_body = next_q_wt.multiply(prev_q_wt)
    return q_acc_body

def measurement_model(y_i, w_i_dash, delta_t, data, tuneR = 10):
    num = y_i.shape[0]
    z_acc, z_gyro = np.zeros((num, 3)), np.zeros((num, 3))
    z_gyro = y_i[:,4:7]
    g_quat = Quaternion(0, [0, 0, 9.8], "value")
    R = np.identity(6)*tuneR

    for i in range(y_i.shape[0]):        
        prev_q_wt = Quaternion(y_i[i][0], y_i[i][1:4], "value")
        prev_q_wt_inverse = prev_q_wt.inverse()
        next_q_wt = prev_q_wt_inverse.multiply(g_quat)
        q_acc_body = next_q_wt.multiply(prev_q_wt)
        z_acc[i] = q_acc_body.quaternion[1:4] #stores quaternion values
        
    #Z = 12*6 z_gyro = 12*3
    Z = np.hstack((z_acc, z_gyro))
    cov_z = covariance(Z)
    mean_z = np.average(Z, axis = 0)
    innov = data - mean_z  
    innov = innov.reshape((innov.shape[0], 1))  ##shape becomes 6*1
    Pvv = cov_z + R

    #12*6 w_i_dash, Pxz = 6*6, innov = 6*1
    num = w_i_dash.shape[0]
    Pxz = (1/num) * (w_i_dash.T @ (Z - mean_z))

    #kalman_gyro 6*6
    try:
        kalman = Pxz @ np.linalg.inv(Pvv)

    except(ValueError, ArithmeticError):
        print("Error: trying to find inverse of singular matrix")
    
    #6*1
    update = kalman @ innov
    #1*6
    return update.T[0], kalman, Pvv

    
#verified       
def transform_7d_to_6d(y):
    # print('shape of y in transform 7d to 6d', y.shape)
    #type (n,1)
    if len(y.shape) == 1:
        w_i = np.zeros((y.shape[0]-1,))
        w_i[0:3] = quat_to_rot(y)
        w_i[3:6] = y[4:7]
        
    else:
        w_i = np.zeros((y.shape[0], y.shape[1]-1))
        for i in range(y.shape[0]):
            w_i[i][0:3] = quat_to_rot(y[i])
            w_i[i][3:6] = y[i][4:7]
    return w_i

#verified
def make_quat_list(y):
    quats = []
    for i in range(y.shape[0]):
        quats.append(Quaternion(y[i][0], y[i][1:4], "value"))
    return quats
       
def UKF(data, delta_ts, tuneQ = 10):
    # print(data.shape, len(delta_ts))
    steps_logger = []
    roll, pitch, yaw = [], [], []
    P = np.identity(6)*0.01
    Q = np.identity(6)*tuneQ
    w_xyz = np.array([0.004, 0.003, 0.003])
    prev_qk = Quaternion(1, [0,0,0]).unit_quaternion()
    x_i_prev = np.concatenate([np.array(prev_qk.quaternion), w_xyz])
    print('x_i_prev ', x_i_prev)
    # for i in range(1,5):
    for i in range(1,data.shape[0]):
        #### PREIDCTION #####
        w_i = sigma_pts(P, Q)

        y_i = process_model(w_i, x_i_prev, delta_ts[i])

        y_i_quats = make_quat_list(y_i)
        mean_y_quat, e, steps = quat_average(y_i_quats, Quaternion(x_i_prev[0], x_i_prev[1:4], "value"))
        steps_logger.append(steps)
        
        mean_y_w = np.mean(y_i[:, 4:7], axis = 0)
        w_sub = y_i[:, 4:7] - mean_y_w
        w_i_dash = np.hstack([e, w_sub])

        cov_x = (1/w_i_dash.shape[0])*(w_i_dash.T @ w_i_dash)
        
        #### PREIDCTION #####
        
        #### MEASUREMENT MODEL ####
        update, kalman, Pvv = measurement_model(y_i, w_i_dash, delta_ts[i], data[i])

        update_7d_quat = rot_to_quat(update[0:3])
        x_i_quat = update_7d_quat.multiply(mean_y_quat)
        x_i_ang = mean_y_w + update[3:6]  #bug fixed
        x_i_prev = np.concatenate([x_i_quat.quaternion, x_i_ang])
        
        # print('new estimate is ', x_i_prev)
        
        r, p, y = x_i_quat.quat_to_eular()
        roll.append(r)
        pitch.append(p)
        yaw.append(y)
        # 6*6, 6*6, 6*6
        P = cov_x - kalman @ Pvv @ kalman.T
        #### MEASUREMENT MODEL ####
        # plot_angles(steps_logger, 'steps_to_converge_avg.png')
    return roll, pitch, yaw

