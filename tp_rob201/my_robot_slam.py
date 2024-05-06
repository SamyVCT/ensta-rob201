"""
Robot controller definition
Complete controller including SLAM, planning, path following
"""
import numpy as np

from place_bot.entities.robot_abstract import RobotAbstract
from place_bot.entities.odometer import OdometerParams
from place_bot.entities.lidar import LidarParams

from tiny_slam import TinySlam

from control import potential_field_control, reactive_obst_avoid
from occupancy_grid import OccupancyGrid
from planner import Planner


# Definition of our robot controller
class MyRobotSlam(RobotAbstract):
    """A robot controller including SLAM, path planning and path following"""

    def __init__(self,
                 lidar_params: LidarParams = LidarParams(),
                 odometer_params: OdometerParams = OdometerParams()):
        # Passing parameter to parent class
        super().__init__(should_display_lidar=False,
                         lidar_params=lidar_params,
                         odometer_params=odometer_params)

        # step counter to deal with init and display
        self.counter = 0

        # Init SLAM object
        self._size_area = (800, 800)
        self.occupancy_grid = OccupancyGrid(x_min=- self._size_area[0],
                                            x_max=self._size_area[0],
                                            y_min=- self._size_area[1],
                                            y_max=self._size_area[1],
                                            resolution=2)
        self.tiny_slam = TinySlam(self.occupancy_grid)
        self.planner = Planner(self.occupancy_grid)

        # storage for pose after localization
        self.corrected_pose = np.array([0, 0, 0])
    
    def transform_list(self):
        return [[x[0], x[1]] for x in self.planner.path]

    def control(self):
        """
        Main control function executed at each time step
        """

        """ PLANIFICATION DE TRAJECTOIRE"""
        counter_planif = 600
        self.counter+=1
        goal = [-500, 130, 0]#[30,500,0]#
        pose = self.odometer_values()
        if self.counter==1:
            print("Initialisation of the map")
            self.tiny_slam.update_map(self.lidar(), self.odometer_values())
       
        score = self.tiny_slam.localise(self.lidar(), self.odometer_values())
        if score > 1000:
            # print("score",score)
            self.corrected_pose = self.tiny_slam.get_corrected_pose(self.odometer_values())
            self.tiny_slam.update_map(self.lidar(), self.corrected_pose)
            if self.counter < counter_planif:
                self.tiny_slam.grid.display_cv(pose, goal, traj=None)
            
        if self.counter < counter_planif:
            command = potential_field_control(self.lidar(), np.array(self.corrected_pose), goal)
            return command
        else :
            if self.counter == counter_planif:
                start = self.tiny_slam.get_corrected_pose(self.odometer_values())
                start_map = (start[0], start[1])#self.tiny_slam.grid.conv_world_to_map(start[0], start[1])
                goal_map = (0,0)#self.tiny_slam.grid.conv_world_to_map(0, 0)
                self.planner.path = self.planner.plan(start_map, goal_map)
                self.planner.path = np.array(self.transform_list()[::-1])
                # print("départ, arrivée, trajectoire planifiée :", start_map, goal_map,self.planner.path)
                command = {"forward": 0,  "rotation": 0}

            else:
                if len(self.planner.path) == 0:
                    print("The robot has arrived.")
                    command = {"forward": 0, "rotation": 0}
                else:
                    tempgoal=self.planner.path[0]
                    # print("tempgoal",tempgoal)
                    command = potential_field_control(self.lidar(), np.array(self.corrected_pose), [tempgoal[0], tempgoal[1], 0])
                    # print(self.planner.path,pose)
                    self.tiny_slam.grid.display_cv(self.corrected_pose,[0,0,0], self.planner.path)
                if self.counter % 3 == 0:
                    if len(self.planner.path) != 0:
                        self.planner.path = self.planner.path[1:]
       
        
        return command

        """ SANS PLANIFICATION DE TRAJECTOIRE"""
        # self.counter+=1
        # goal = [-400, 120, 0]
        # if self.counter==1:
        #     print("Initialisation of the map")
        #     self.tiny_slam.update_map(self.lidar(), self.odometer_values())
            
        # if self.counter !=0 :
        #     score = self.tiny_slam.localise(self.lidar(), self.odometer_values())
        #     if score > 1000  and self.counter != 0 :#- 5*self.counter:
        #         print("score",score)
        #         self.corrected_pose = self.tiny_slam.get_corrected_pose(self.odometer_values())
        #         self.tiny_slam.update_map(self.lidar(), self.corrected_pose)
        # return potential_field_control(self.lidar(), np.array(self.corrected_pose), np.array(goal))

    def control_tp1(self):
        """
        Control function for TP1
        """
        self.tiny_slam.compute()

        # Compute new command speed to perform obstacle avoidance
        command = reactive_obst_avoid(self.lidar())
        return command

    def control_tp2(self):
        """
        Control function for TP2
        """
        pose = self.tiny_slam.get_corrected_pose(self.odometer_values())
        goal = [-100,00,0]
        # Compute new command speed to perform obstacle avoidance
        command = potential_field_control(self.lidar(), np.array(pose), np.array(goal))

        return command
