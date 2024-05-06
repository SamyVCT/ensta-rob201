""" A set of robotics control functions """

import random
import numpy as np


def reactive_obst_avoid(lidar):
    """
    Simple obstacle avoidance
    lidar : placebot object with lidar data
    """
    # TODO for TP1
    
    telemetre = lidar.get_sensor_values()
    angles = lidar.get_ray_angles()
    # print("telem",len(telemetre),telemetre, "angle", len(angles),angles)
    if any(value < 100 for value in telemetre[170:190]):
        speed = 0.05
        rotation_speed = 0.3*random.choice([-1, 1])
        # print("rotation_speed",rotation_speed)
    else : 
        speed = 0.05
        rotation_speed = 0


    command = {"forward": speed,
               "rotation": rotation_speed}

    return command


def potential_field_control(lidar, current_pose, goal_pose):
    """
    Control using potential field for goal reaching and obstacle avoidance
    lidar : placebot object with lidar data
    current_pose : [x, y, theta] nparray, current pose in odom or world frame
    goal_pose : [x, y, theta] nparray, target pose in odom or world frame
    Notes: As lidar and odom are local only data, goal and gradient will be defined either in
    robot (x,y) frame (centered on robot, x forward, y on left) or in odom (centered / aligned
    on initial pose, x forward, y on left)
    """
    speed = 0.05
    rotation_speed = 0

    dsafe = 60  # security distance

    #potentiel répulsif
    repulsive = np.zeros(2)
    telemetre = lidar.get_sensor_values()
    angles = lidar.get_ray_angles()
    ouverture_angle = 25
    pose_resized = np.resize(current_pose[0:2], (2, len(telemetre)))
    telemetre_cartesien = np.add(pose_resized,telemetre * np.array([np.cos(angles+current_pose[2]), np.sin(angles+current_pose[2])]))
    K = 5000
    i = np.argmin(telemetre)
    d_obs =  telemetre[i]    # d(q - q_obs)
    angle_obs =  angles[i]   # par rapport à l'avant du robot
     # Find the index of the closest point
    if any(telemetre[i] < dsafe for i in range(180-ouverture_angle,180+ouverture_angle)):
        for k in range(180-ouverture_angle,180+ouverture_angle):
            repulsive = K*(1/d_obs - 1/dsafe)*d_obs*np.array([np.cos(angle_obs+current_pose[2]), np.sin(angle_obs+current_pose[2])])/(d_obs**3)
            
            # repulsive += np.array([(telemetre_cartesien[0][k]- current_pose[0])*np.cos(angles[k]), (telemetre_cartesien[1][k] - current_pose[1])*np.sin(angles[k])]) * (1 / telemetre[k] - 1 / dsafe) / telemetre[k] ** 3
    # print("repulsive",repulsive)

    #potentiel attractif
    if any(goal_pose[i] != current_pose[i] for i in [0,1]):
        attractif = [a-b for a,b in zip(goal_pose[0:2],current_pose[0:2])]
        attractif = attractif/ np.linalg.norm(attractif)
    else: 
        print("goal reached")
        return {"forward": 0,
            "rotation": 0}

    # Orientation désirée vers l'objectif
    desired_orientation = np.arctan2(goal_pose[1] - current_pose[1], goal_pose[0] - current_pose[0])

    # Difference d'orientation
    rotation_speed = (desired_orientation - current_pose[2])

    # Normalize the rotation speed to [-pi, pi]
    rotation_speed = (rotation_speed + np.pi) % (2 * np.pi) - np.pi

    # print("attractif, repulsif",np.linalg.norm(attractif),np.linalg.norm(repulsive))
    grad_tot = np.linalg.norm(attractif) +  np.linalg.norm(repulsive)
    # grad_tot = attractif + repulsive
    # print("grad_tot",grad_tot, np.linalg.norm(grad_tot))
    eps = 0.1
    speed += eps*grad_tot
    speed = np.clip(speed, -1, 0.15)

    # si obstacle devant, rotation
    if any(telemetre[i] < dsafe for i in range(180-ouverture_angle,180+ouverture_angle)):
        count_left = sum(1 for i in range(180-ouverture_angle,180) if telemetre[i] < dsafe)
        print("Nombre d'obstacles à gauche :", count_left)

        count_right = sum(1 for i in range(180,180+ouverture_angle) if telemetre[i] < dsafe)
        print("Nombre d'obstacles à droite :", count_right)

        if count_left > count_right:
            speed *= 0.01
            rotation_speed = 0.5
        else:
            speed *= 0.01
            rotation_speed = -0.5

    # d = np.linalg.norm(current_pose[0:2] - goal_pose[0:2]) #distance euclidienne
    # print("pose,goal,d",current_pose[0:2],goal_pose[0:2],d)
    
    rotation_speed = np.clip(rotation_speed, -0.5, 0.5)
    if abs(rotation_speed) < 0.01:
        rotation_speed = 0

    # print("speed,rotation_speed",speed,rotation_speed)
    command = {"forward": speed,
            "rotation": rotation_speed}

    return command
