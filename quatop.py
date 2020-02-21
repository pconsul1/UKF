#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 13:40:37 2020

@author: poojaconsul
"""
import math
from  quaternion import Quaternion
import numpy as np
    
def rot_to_quat(w): 
    w = np.array(w)
    # yaw, pitch, roll = w[2], w[1], w[0]
    # qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    # qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    # qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    # qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    # q1 = Quaternion(qw, [qx, qy, qz], "value")
    #print("Quaternion  ", qw, qx, qy, qz)
    norm = np.linalg.norm(w)
    if(norm == 0):
        # print('norm is 0 in rot_to_quat function')
        return Quaternion(0, [0,0,0], "value")
        
    axis = w/norm
    q1 = Quaternion(norm, axis, "angle")
    q1 = q1.unit_quaternion()
    return q1

def check_q0_0(quat):
    if quat[0] == 0:
        return True
    return False
    
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


def quat_average(quaternions, current_state_quaternion): #verified

    '''
    0.92707 +   0.02149i +   0.19191j +   0.32132k
    0.90361 + 0.0025836i +  0.097279j +   0.41716k
    0.75868 -   0.21289i +   0.53263j +   0.30884k
    '''
    '''
    quaternions = []
    quaternions.append(Quaternion(0.92707, [0.02149,   0.19191,   0.32132], "qu"))
    quaternions.append(Quaternion(0.90361, [0.0025836,  0.097279,   0.41716], "qu"))
    quaternions.append(Quaternion(0.75868, [-0.21289,   0.53263,   0.30884], "qu"))
    '''
    #Declare a random unit qauternion
    #estimated_mean = Quaternion(0.75868, [0.21289,   0.53263,   0.30884], "qu")
    estimated_mean = current_state_quaternion
    counter = 0   
    while(True):
        counter = counter+1
        ei_s = np.zeros((3,len(quaternions)))
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
        
        
        if(np.linalg.norm(e)< 0.05 or counter>10):
            break
    print(counter)
    
    return estimated_mean, ei_s.T, counter

# def quat_average(quats, quat_mean, thresh = 0.001):
#     # quat_mean = Quaternion(0, [1,0,0], "value")
#     quat_inverse = quat_mean.inverse()
#     err_agg = pow(10,4)
#     num = len(quats)
#     steps = 0
#     e = np.zeros((num, 3))
#     # print('number of quaternions ', num)
#     # print(err_agg, thresh)
#     while(err_agg > thresh or steps < 100):
#         e = np.zeros((num, 3))
#         # print('step number ', steps)
#         for i in range(num):
#             e_i = quats[i].multiply(quat_inverse)
#             # print('quat * mean inverse result ', e_i.quaternion)
#             e[i] = quat_to_rot(np.array(e_i.quaternion))
#             # print('e[%d] value '%i, e[i])
#         e_avg = np.mean(e, axis = 0)
#         err_agg = np.linalg.norm(e_avg)
#         e_avg_quat = rot_to_quat(e_avg)
#         quat_mean = e_avg_quat.multiply(quat_mean)
#         quat_inverse = quat_mean.inverse()
#         steps += 1
#         # print('e avg', e_avg)
#         # print('e avg quat ', e_avg_quat.quaternion)
#         # print('quat mean at step %d'%i, quat_mean.quaternion)
#         # print('err_agg at step %d'%steps, err_agg)
#         # print(err_agg, quat_mean.quaternion)
#     print('steps taken to converge ', steps)
#     quat_mean = quat_mean.inverse()
#     return quat_mean, e, steps