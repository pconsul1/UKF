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
from quatop import rot_to_quat, quat_to_rot, quat_average
#accelerometer values ->quternions
#rotation values remain same

def plot_angles(ls, fname):
    plt.plot(ls, linewidth = 1.0)
    plt.savefig(fname)
    plt.cla()
    
#verified
def init_covar():
    mat = np.identity(6)*0.01
    cov_x = np.cov(mat)
    return cov_x

#verified
def sigma_pts(P, Q):
    p_plus_q = P+Q
    
    ###PSD fix
    eigs = np.linalg.eigvals(p_plus_q)
    if not(np.all(eigs) > 0):
        # print('negative eigen values in covariance matrix ', np.min(eigs), eigs)
        p_plus_q = p_plus_q - np.min(eigs)*np.identity(6)
        p_plus_q/=p_plus_q[1,1]
        
    n = P.shape[0]
    # l = np.linalg.cholesky(p_plus_q + pow(10, -8))
    l = np.linalg.cholesky(p_plus_q)
    # l = l/np.linalg.norm(l)
    left_vec = l*np.sqrt(2*n)
    right_vec = -l*np.sqrt(2*n)
    w_i = np.hstack((left_vec, right_vec)).T
    # print('w_i shape', w_i.shape)
    # l_t = l
    # norm = np.linalg.norm(l_t)
    # # l_t/=norm
    # # print(l.shape, l_t.shape)
    # w_i = np.zeros((2*n, n))
    # factor = math.sqrt(2*n)
    # # w_i[0] = mean
    # # print('sigma points factor ', factor)
    # for i in range(0,n):
    #     ##verified row-col
    #     # print('l_t row %d' %i, l_t[i])
    #     # print('l col %d'%i, l[:,i])
    #     w_i[i] = l_t[i]*factor
    #     w_i[i+n] = -l_t[i]*factor
    #     # print('cross check neg multiplication ',-l_t[i]*factor)
    # print('mean of sigma points is ', np.mean(w_i, axis = 0))
    return w_i


def get_state_est(w_i, x_i_prev, qk_prev):
    """
    Parameters
    ----------
    w_i : array of size (12,6)
        sigma points.
    x_i_prev : array of size (7,)
        previous state estimate consisting of quaternion values, angular velocities.
    qk_prev : Quaternion
        quaternion of previous state.

    Returns
    -------
    x_i : array of size (7,)
        state estimate consisting of quaternion values, angular velocities.

    """
    x_i = np.zeros((w_i.shape[0], w_i.shape[1]+1))
    for i in range(w_i.shape[0]):
        
        qw = Quaternion(w_i[i][0], w_i[i][1:4], "value")
        # print(qw.quaternion, qk_prev.quaternion)
        qk = qk_prev.multiply(qw)
        # print(w_i[i][3:6], x_i_prev[4:7])
        angular_sum = np.array(w_i[i][3:6]) + np.array(x_i_prev[4:7])
        x_i[i] = np.concatenate([qk.quaternion, angular_sum], axis = 0)
    return x_i
        
#verified
#https://stackoverflow.com/questions/38177799/numpy-covariance-and-covariance-matrix-by-formula-is-producing-different-results
def covariance(data):
    #data is n*6 col vectors rvs
    data = data.T
    mean = np.average(data.T, axis = 0)
    cov = ((data - mean[:, None]) @ (data - mean[:, None]).T)/(data.shape[1])
    return cov

def process_model(w_i, x_i_prev, delta_ts):
    # print('shape of sigma points variable in process model ', sigma_pts.shape)
    # print('sigma point in process_model ', sigma_pts)
    #implementing y_i = A(x_i, 0)
    w_prev_est = x_i_prev[4:7]
    quat_prev_est = Quaternion(x_i_prev[0], x_i_prev[1:4], "value")
    
    # print('w prev est ', w_prev_est)
    y_i = np.zeros((w_i.shape[0], w_i.shape[1]+1))
    for i in range(w_i.shape[0]):
        # print('w_i %d '%i, w_i[i])
        # print('norm of sigmal points - ', np.linalg.norm(w_i[i][0:3]))
        w_i_quat = rot_to_quat(w_i[i][0:3])
        
        #make x_i point
        x_i_quat = quat_prev_est.multiply(w_i_quat)
        x_i_w = w_prev_est + w_i[i][3:6]
        # print('adding angular velocities ', w_prev_est, w_i[i][3:6])
        # print('y %d w part  \n'%i, x_i_w)
        x_i_w_norm = np.linalg.norm(x_i_w)
        angle = x_i_w_norm * delta_ts
        axis = x_i_w/x_i_w_norm
        q_delta = Quaternion(angle, axis, "angle")
        # print('q_delta ',q_delta.quaternion)
        y_i_quat = x_i_quat.multiply(q_delta)
        y_i[i] = np.concatenate([np.array(y_i_quat.quaternion), x_i_w], axis = 0)
        # print('y%d'%i, ' is ', y_i[i])
    # print('shape of y_i after process_model execites ', y_i.shape)
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
    # g_quat = g_quat.unit_quaternion()
    
    prev_q_wt_inverse = prev_q_wt.inverse()
    # print('prev_q_wt ', prev_q_wt_inverse.quaternion)
    next_q_wt = prev_q_wt_inverse.multiply(g_quat)
    # print('next_q_wt ', next_q_wt.quaternion)
    # temp_quat = next_q_wt.unit_quaternion()
    # print('temp quat ', temp_quat.quaternion)
    
    # print('inverse quat ', next_q_wt_inverse.quaternion)
    q_acc_body = next_q_wt.multiply(prev_q_wt)
    # print('quat inverse check ', prev_q_wt_inverse.multiply(prev_q_wt).quaternion) ##verified
    # print('q_acc_body ', q_acc_body.quaternion[1:4])
    return q_acc_body

def measurement_model(y_i, w_i_dash, delta_t, data, tuneR = 10):
    num = y_i.shape[0]
    z_acc, z_gyro = np.zeros((num, 3)), np.zeros((num, 3))
    z_gyro = y_i[:,4:7]
    # print('z_gyro shape ',z_gyro.shape,)
    # print(z_gyro, y_i[:,4:7])
    # R = np.ones((6,6))*tuneR
    R = np.identity(6)*tuneR
    # print('check input values in measurement model ',y_i.shape, w_i_dash.shape, delta_t, data)
    for i in range(y_i.shape[0]):        
        prev_q_wt = Quaternion(y_i[i][0], y_i[i][1:4], "value")
        # print('norm of y_%d quaternion '%i, prev_q_wt.norm())
        acc = measure_acc(prev_q_wt, delta_t)
        # print('acceleration after measurement for %d '%i, acc.quaternion)
        # z_acc_quats[i] = acc #stores quaternions
        z_acc[i] = acc.quaternion[1:4] #stores quaternion values
        # print('z acc values check ', z_acc[i]) ##verified
        
    #Z = 12*6 z_gyro = 12*3
    Z = np.hstack((z_acc, z_gyro))
    # print('value of Z', Z, Z.shape)
    # print('acc and gyro first row', z_acc[0], z_gyro[0])
    cov_z = covariance(Z)
    # print('cov check', np.linalg.norm(np.cov(Z.T) - cov_z)) ##verified with n-1 exact
    mean_z = np.average(Z, axis = 0)
    # print('shape of Z gyro', z_gyro)
    # print('Z mean ', mean_z)
    #innov_gyro (6,)

    # print('compare acc value ')
    # print(data[0:3])
    # print(mean_z[0:3])
    innov = data - mean_z  
    innov = innov.reshape((innov.shape[0], 1))  ##shape becomes 6*1
    # print('innovation ', innov.shape, innov)
    #Pvv = (6,6)
    Pvv = cov_z + R

    #12*6 w_i_dash, Pxz = 6*6, innov = 6*1
    # print('w_i_dash shape and innovation shape ', w_i_dash.shape, innov.shape)
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
def transform_6d_to_7d(y, delta_ts):
    rot = y[:3]
    norm = np.linalg.norm(rot)
    angle = norm * delta_ts
    try:
        axis = rot/norm
    except (ValueError, ArithmeticError):
        print("Error: 0 norm in transform_6d_to_7d for vector {}".format(y))
    quat =  Quaternion(angle, axis, "angle")
    vec7d = np.concatenate([quat.quaternion, y[3:6]])
    return vec7d

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
    q_pts = np.ones((P.shape[0], P.shape[1]))*tuneQ
    Q = covariance(q_pts)
    Q = np.identity(6)*tuneQ
    # cov_x = covariance((data[:init_pts] + 10).T)
    # mean = np.mean(data[:init_pts], axis = 0)
    # w_i = sigma_pts(P, Q)
    # print(w_i.shape)
    ### initializing the state quaternion
    # w_xyz = np.array([0.2,0.2,0.2])
    w_xyz = np.array([0.004, 0.003, 0.003])
    w_xyz_norm = np.linalg.norm(w_xyz)
    angle_w = delta_ts[0] * w_xyz_norm
    axis_w = w_xyz/w_xyz_norm
    prev_qk = Quaternion(angle_w, axis_w, "angle")
    # print(prev_qk.quaternion)
    ### initializing the state quaternion
    prev_qk = Quaternion(1, [0,0,0]).unit_quaternion()
    x_i_prev = np.concatenate([np.array(prev_qk.quaternion), w_xyz])
    print('x_i_prev ', x_i_prev)
    # for i in range(1,5):
    for i in range(1,data.shape[0]):
        #### PREIDCTION #####
        w_i = sigma_pts(P, Q)
        # print('shape of w_i', w_i.shape)
        ### computing q_delta with x_i_prev
        # verified 
        # print('input to process model wishp pre est delta',w_i.shape, x_i_prev,  delta_ts[i])
        y_i = process_model(w_i, x_i_prev, delta_ts[i])
        # print('value of y_i ', y_i)
        # mean_y = np.avergae(y_i, axis = 0)
        y_i_quats = make_quat_list(y_i)
        mean_y_quat, e, steps = quat_average(y_i_quats, Quaternion(x_i_prev[0], x_i_prev[1:4], "value"))
        steps_logger.append(steps)
        # cov_y = covariance(y_i.T)
        
        mean_y_w = np.mean(y_i[:, 4:7], axis = 0)
        w_sub = y_i[:, 4:7] - mean_y_w
        # print('y_i shape and mean', y_i.shape, mean_y_w)
        w_i_dash = np.hstack([e, w_sub])
        # ##verified
        # print(w_i_dash.shape)
        # print('cross check stacking')
        # print('stack first row ', w_i_dash[0])
        # print('e and w_sub first row ', e[0], w_sub[0])
        # print('mean check ', mean_y_w, np.average(w_i_dash, axis = 0))
        # y_sub_xk = y_i - mean_y
        # # print('y_sub_xk values ', y_sub_xk)
        # w_i_dash = transform_7d_to_6d(y_sub_xk)
        # print('y_i[:, 4:7]', y_i[:, 4:7])
        # print('y i w mean value ', mean_y_w)
        # print('w_sub ', w_sub)
        # x_i_prev = x_i
        # x_i = update 
        # prev_qk = Quaternion(x_i_prev[0], x_i_prev[1:4], "value")
        cov_x = (1/w_i_dash.shape[0])*(w_i_dash.T @ w_i_dash)
        # print('div = ',w_i_dash.shape[0])
        # print('w_i_dash ', w_i_dash)
        #### PREIDCTION #####
        
        #### MEASUREMENT MODEL ####
        # The set {Yi} is thus transformed further by the measurement model H , resulting in a set {Zi } of projected measurement vectors.
        update, kalman, Pvv = measurement_model(y_i, w_i_dash, delta_ts[i], data[i])
        # print('PVV ', Pvv)
        # print('kalman ', kalman)
        #convert update to 7d and perform quternion multiplication and angular velocity addition separately
        #update 6*1
        # mean_y_6d = transform_7d_to_6d(mean_y)
        # print('mean_y   ',mean_y)
        # print('update[0:3] ',update, update[0:3])
        update_7d_quat = rot_to_quat(update[0:3])
        # x_i_quat = mean_y_quat.multiply(update_7d_quat)
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
        # print('covariance cov_x  ', cov_x)
        #### MEASUREMENT MODEL ####
        plot_angles(steps_logger, 'steps_to_converge_avg.png')
    return roll, pitch, yaw

# data = np.array([1.2,0.9,0,0,0,0])
# q1 = Quaternion(0, [0,0.98,0], "angle")
# a, b = acceleration_to_quaternion(data, q1, 0.01)
# print(a, b)