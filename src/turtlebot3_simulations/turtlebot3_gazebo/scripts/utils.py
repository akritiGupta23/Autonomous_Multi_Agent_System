#!/usr/bin/env python3
import yaml
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict

def read_yaml_map_file(file_path: str) -> Dict:
    with open(file_path, 'r') as yaml_file:
        return yaml.safe_load(yaml_file)

def convert_map_image_to_grid(map_contents: Dict) -> Tuple[np.ndarray, float, Tuple[float, float, float]]:
    map_image_array = plt.imread('map.pgm')
    grid_resolution_value, grid_origin_coordinates = map_contents['resolution'], map_contents['origin']
    
    occupancy_grid_matrix = np.zeros_like(map_image_array)
    occupancy_grid_matrix[(map_image_array == 205) | (map_image_array == 254)] = 100
    occupancy_grid_matrix[map_image_array == 0] = 0
    
    return occupancy_grid_matrix, grid_resolution_value, grid_origin_coordinates

def transform_coordinates(x_position: float, y_position: float) -> Tuple[float, float]:
    return (y_position / 19.15 - 10, 10 - x_position / 19.15)

def inverse_transform_coordinates(x_position: float, y_position: float) -> Tuple[float, float]:
    return (19.15 * (10 - y_position), 19.15 * (10 + x_position))

