""" A simple robotics navigation code including SLAM, exploration, planning"""

import cv2
import numpy as np

from occupancy_grid import OccupancyGrid


class TinySlam:
    """Simple occupancy grid SLAM"""

    def __init__(self, occupancy_grid: OccupancyGrid):
        self.grid = occupancy_grid
        
        # Origin of the odom frame in the map frame
        self.odom_pose_ref = np.array([0, 0, 0])

    def _score(self, lidar, pose):
        """
        Computes the sum of log probabilities of laser end points in the map
        lidar : placebot object with lidar data
        pose : [x, y, theta] nparray, position of the robot to evaluate, in world coordinates
        """
        # TODO for TP4
        telemetre = lidar.get_sensor_values()
        angles = lidar.get_ray_angles()
      
        # Supprimer les points a la distance maximale du laser???
        # max_distance = np.max(telemetre)
        # angles = angles[telemetre < max_distance]
        # telemetre = telemetre[telemetre < max_distance]
        angles = angles[telemetre < telemetre.max()]
        telemetre = telemetre[telemetre < telemetre.max()]

        pose_resized = np.resize(pose[0:2], (2, len(telemetre)))

        x = pose_resized[0] + telemetre * np.cos(angles + pose[2])
        y = pose_resized[1] + telemetre * np.sin(angles + pose[2])
        points = np.array([x, y]).T
        # print(points.shape)
        points = points[(points[:, 0] >= self.grid.x_min_world) & (points[:, 0] < self.grid.x_max_world) & (points[:, 1] >= self.grid.y_min_world) & (points[:, 1] < self.grid.y_max_world)]
        indices_x, indices_y = self.grid.conv_world_to_map(points[:,0], points[:,1])
        # indices = np.array([indices_x, indices_y])
        # print("indices",len(indices_x))
        score = np.sum(np.exp(self.grid.occupancy_map[indices_x, indices_y]))
        # print("score",score)
        return score

    def get_corrected_pose(self, odom_pose, odom_pose_ref=None):
        """
        Compute corrected pose in map frame from raw odom pose and odom frame pose,
        either given as second param or using the ref from the object
        odom : raw odometry position
        odom_pose_ref : optional, origin of the odom frame if given,
                        use self.odom_pose_ref if not given
        """
        # TODO for TP4
        if odom_pose_ref is None:
            odom_pose_ref = self.odom_pose_ref
       
        corrected_x = odom_pose[0] + odom_pose_ref[0]
        corrected_y = odom_pose[1]  + odom_pose_ref[1]

        # Compute the corrected rotation
        corrected_theta = (odom_pose[2] + odom_pose_ref[2]) % (2 * np.pi) 

        return [corrected_x, corrected_y, corrected_theta]

    def localise(self, lidar, raw_odom_pose):
        """
        Compute the robot position wrt the map, and updates the odometry reference
        lidar : placebot object with lidar data
        odom : [x, y, theta] nparray, raw odometry position
        """
        # TODO for TP4
        N = 200
        sigma = 0.01
        best_score = self._score(lidar, self.get_corrected_pose(raw_odom_pose,self.odom_pose_ref))
        best_pose = self.odom_pose_ref 
        for i in range(N):
            new_pose = np.random.normal(0, sigma, 3)
            # print(np.size(new_pose))
            pose_world = self.get_corrected_pose(self.odom_pose_ref, new_pose)
            score = self._score(lidar, pose_world)
            if score > best_score:
                best_score = score
                best_pose = new_pose
        self.odom_pose_ref = best_pose 
        
        return best_score

    def update_map(self, lidar, pose):
        """
        Bayesian map update with new observation
        lidar : placebot object with lidar data
        pose : [x, y, theta] nparray, corrected pose in world coordinates
        """
        # TODO for TP3
        
        telemetre = lidar.get_sensor_values()
        angles = lidar.get_ray_angles()
        #delete points at the maximum distance of the laser
        # max_distance = np.max(telemetre)
        # angles = angles[telemetre < max_distance]
        # telemetre = telemetre[telemetre < max_distance]
        # angles = angles[telemetre < 600]
        # telemetre = telemetre[telemetre < 600]

        #Conversion des coordonnées polaires provenant du lidar en coordonnées cartésiennes
        pose_resized = np.resize(pose[0:2], (2, len(telemetre)))
        telemetre_cartesien = np.add(pose_resized,telemetre * np.array([np.cos(angles+pose[2]), np.sin(angles+pose[2])]))

        # p_faible = np.log(0.02/(0.98))
        # p_forte = np.log(0.98/(0.02))   
        # p_moyen = 0 #np.log(0.6/(0.4))
        buffer_zone = 20  # Définir le ratio de la zone tampon
        # for i in range (len(telemetre_cartesien[0])):
        #     # if i%2 == 0 :
        #         x_avant = pose[0] + buffer_zone_ratio * (telemetre_cartesien[0][i] - pose[0])
        #         y_avant = pose[1] + buffer_zone_ratio * (telemetre_cartesien[1][i] - pose[1])
        #         self.grid.add_map_line(pose[0], pose[1], x_avant, y_avant, p_faible)
        #         self.grid.add_map_line(x_avant, y_avant, telemetre_cartesien[0][i], telemetre_cartesien[1][i], p_moyen)
       
        # self.grid.add_map_points(telemetre_cartesien[0], telemetre_cartesien[1], p_forte)

        p_faible = np.log(0.02/(0.98))
        p_forte = np.log(0.98/(0.02))
        x_avant = pose[0] + (telemetre - buffer_zone) * np.cos(angles + pose[2])
        y_avant = pose[1] + (telemetre - buffer_zone) * np.sin(angles + pose[2])
        for i in range (len(telemetre)):
            self.grid.add_map_line(pose[0], pose[1], x_avant[i], y_avant[i], p_faible)

        self.grid.add_map_points(pose[0] + telemetre * np.cos(angles + pose[2]), pose[1] + telemetre * np.sin(angles + pose[2]), p_forte)
      
        self.grid.occupancy_map[self.grid.occupancy_map > 5] = 5
        self.grid.occupancy_map[self.grid.occupancy_map < -5] = -5

        goal = [-400,50,0]

    def compute(self):
        """ Useless function, just for the exercise on using the profiler """
        # Remove after TP1

        ranges = np.random.rand(3600)
        ray_angles = np.arange(-np.pi, np.pi, np.pi / 1800)

        # Poor implementation of polar to cartesian conversion
        points = []
        for i in range(3600):
            pt_x = ranges[i] * np.cos(ray_angles[i])
            pt_y = ranges[i] * np.sin(ray_angles[i])
            points.append([pt_x, pt_y])
