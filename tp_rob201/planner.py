import numpy as np
import heapq
from occupancy_grid import OccupancyGrid

class Planner:
    """Simple occupancy grid Planner"""

    def __init__(self, occupancy_grid: OccupancyGrid):
        self.grid = occupancy_grid
        # Origin of the odom frame in the map frame
        self.odom_pose_ref = np.array([0, 0, 0])
        self.path = False

    def get_neighbors(self, current_cell):
        # Get the neighbors of a cell
        neighbors = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                if current_cell[0] + i >= 0 and current_cell[0] + i < self.grid.x_max_world and current_cell[1] + j >= 0 and current_cell[1] + j < self.grid.y_max_world:
                    neighbors.append([current_cell[0] + i, current_cell[1] + j])
        return neighbors
    
    def heuristic(self,cell_1,cell_2):
        return np.linalg.norm(np.array(cell_1)-np.array(cell_2))
    
    def reconstruct_path(self,came_from, current):
        total_path = [self.grid.conv_map_to_world(current[0], current[1])]
        while current in came_from.keys():
            current = came_from[current]
            total_path.append(self.grid.conv_map_to_world(current[0], current[1]))
        return total_path
    
    def plan(self, start, goal):
        """
        Compute a path using A*, recompute plan if start or goal change
        start : [x, y, theta] nparray, start pose in world coordinates (theta unused)
        goal : [x, y, theta] nparray, goal pose in world coordinates (theta unused)
        """
        start = self.grid.conv_world_to_map(start[0], start[1])
        goal = self.grid.conv_world_to_map(goal[0], goal[1])

        openSet = []
        heapq.heappush(openSet, (0, start))
        cameFrom = {}
        gScore = {start: 0}
        fScore = {start: self.heuristic(start, goal)}

        while openSet != []:
            _, current = heapq.heappop(openSet)

            if current == goal:
                path = self.reconstruct_path(cameFrom, current)
                return path

            for neighbor in self.get_neighbors(current):
                neighbor = tuple(neighbor)  # Convert list to tuple
                tentative_gScore = gScore[current] + self.heuristic(current, neighbor)
                if neighbor not in gScore or tentative_gScore < gScore[neighbor]:
                    cameFrom[neighbor] = current
                    gScore[neighbor] = tentative_gScore
                    fScore[neighbor] = tentative_gScore + self.heuristic(neighbor, goal)
                    if neighbor not in [i[1] for i in openSet]:
                        heapq.heappush(openSet, (fScore[neighbor], neighbor))

        return False #fail

    def explore_frontiers(self):
        """ Frontier based exploration """
        goal = np.array([0, 0, 0])  # frontier to reach for exploration
        return goal
