#!/usr/bin/env python3

import yaml
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


def load_map_yaml(file_path: str) -> dict:
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def map_image_to_grid(map_data: dict) -> Tuple[np.ndarray, float, Tuple[float, float, float]]:
    f = 'map.pgm'
    with open(f, 'rb') as pgmf:
        image = plt.imread(pgmf)
    resolution = map_data['resolution']
    origin = map_data['origin']
    occupancy_grid = np.zeros_like(image)
    occupancy_grid[image == 205] = 100
    occupancy_grid[image == 254] = 100
    occupancy_grid[image == 0] = 0
    return occupancy_grid, resolution, origin

def convert_coordinates(x, y):
    newx = (y/19.15) - 10
    newy = (10-x/19.15) 
    return (newx, newy)

def inverse_convert_coordinates(x, y):
    newx = 19.15 * (10 - y) 
    newy = 19.15 * (10 + x) 
    return (newx, newy)

