import json
import math
import numpy as np
from stl import mesh
from scipy.spatial import ConvexHull
from itertools import combinations
import argparse
import sys
import csv

class WeightedGraph:
    def __init__(self, triangles=None):
        # Use a dictionary to represent the adjacency list
        self.graph = dict()
        if triangles is not None:
          self.generate_graph(triangles)

    def generate_graph(self, triangles):
      def euclidean_distance(point1, point2):
        if len(point1) != 3 or len(point2) != 3:
            raise ValueError("Input points must be in 3D space with (x, y, z) coordinates.")
        distance = math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2 + (point2[2] - point1[2])**2)
        return distance

      for triangle in triangles:
        (vertex1, vertex2, vertex3) = triangle
        self.add_edge(vertex1, vertex2, euclidean_distance(vertex1, vertex2))
        self.add_edge(vertex2, vertex3, euclidean_distance(vertex2, vertex3))
        self.add_edge(vertex3, vertex1, euclidean_distance(vertex1, vertex3))


    def add_edge(self, start, end, weight):
        # If the start node is not in the graph, add it
        if start not in self.graph:
            self.graph[start] = set()

        # Add the edge with its weight
        self.graph[start].add((end, weight))

        # If the end node is not in the graph, add it
        if end not in self.graph:
            self.graph[end] = set()

        # For undirected graphs, you might want to add the reverse edge
        self.graph[end].add((start, weight))

    def remove_node(self, node):
      if node in self.graph:
        del self.graph[node]

    def get_graph(self):
        return self.graph

    def get_perimeter(self):
      # Compute the convex hull using scipy.spatial.ConvexHull
      coordinates = list(self.graph.keys())

      x_coordinates = []
      y_coordinates = []
      z_coordinates = []
      for (x, y, z) in coordinates:
        x_coordinates.append(x)
        y_coordinates.append(y)
        z_coordinates.append(z)

      if len(np.unique(x_coordinates)) == 1:
        constant_x = x_coordinates[0]
        coordinates = [(y, z) for (x, y, z) in coordinates]
        hull = ConvexHull(coordinates)
        convex_hull_coordinates = [(constant_x, coordinates[i][0], coordinates[i][1]) for i in hull.vertices]
      elif len(np.unique(y_coordinates)) == 1:
        constant_y = y_coordinates[0]
        coordinates = [(x, z) for (x, y, z) in coordinates]
        hull = ConvexHull(coordinates)
        convex_hull_coordinates = [(coordinates[i][0], constant_y, coordinates[i][1]) for i in hull.vertices]
      else:
        constant_z = z_coordinates[0]
        coordinates = [(x, y) for (x, y, z) in coordinates]
        hull = ConvexHull(coordinates)
        convex_hull_coordinates = [(coordinates[i][0], coordinates[i][1], constant_z) for i in hull.vertices]

      return convex_hull_coordinates

def parse_stl_binary(file_path):
  result_dict = {}

  mesh_data = mesh.Mesh.from_file(file_path)

  for i in range(len(mesh_data.normals)):
      # Facet_normal
      normal_components = tuple(int(component) for component in mesh_data.normals[i])

      #  väärtusete normaliseerimine
      z_value = normal_components[2]
      if z_value > 1:
          z_value = 1
      elif z_value < 0:
          z_value = -1

      x_value = normal_components[0]
      if x_value > 1:
          x_value = 1
      elif x_value < 0:
          x_value = -1

      y_value = normal_components[1]
      if y_value > 1:
          y_value = 1
      elif y_value < 0:
          y_value = -1

      adjusted_normal = (x_value, y_value, z_value)

      # Vertex -> tuple kujul
      vertices = [tuple(float(coord) for coord in vertex) for vertex in mesh_data.vectors[i]]

      current_vertices = result_dict.setdefault(adjusted_normal, [])
      current_vertices.append(vertices)

      result_dict[adjusted_normal] = current_vertices

  return result_dict

def are_equal(float1, float2, margin_of_error=0.1):
    return abs(float1 - float2) <= margin_of_error

def traverse_graph(graph, current_node, target_weight, visited=None, current_path=None):
    if visited is None:
        visited = set()
    if current_path is None:
        current_path = []

    visited.add(current_node)
    current_path.append(current_node)
    if current_node in graph:
        for neighbor, weight in graph[current_node]:
            #print(f"Current path: {current_path}")
            if are_equal(weight, target_weight):
                if (neighbor not in visited):
                    # Recursive call for the next node
                    #print(f"{current_node} -> {neighbor}; \t weight: {target_weight} \t current path: {current_path}")
                    result = traverse_graph(graph, neighbor, target_weight, visited, current_path.copy())

                    # If a valid path is found, return it
                    if result:
                        return result
                elif (neighbor==current_path[0]) and (len(current_path)>2):
                    current_path.append(current_path[0])
                    #print(f"{current_node} -> {neighbor}; \t weight: {target_weight} \t current path: {current_path}")
                    return current_path

    # If no valid path is found, backtrack
    current_path.pop()
    return None


def find_loops_with_equal_edge_weigths(graph):
  loops = []
  existing_loop_nodes = set()
  for node in list(graph.keys()):

    if node not in existing_loop_nodes:
      #print(f"Node {node} not in {existing_loop_nodes}")
      for edge in graph[node]:
        _, edge_weight = edge
        loop = traverse_graph(graph, node, edge_weight)
        if loop is not None:
          #print(f"New loop: {loop}")
          loops.append(loop)
          for loop_node in loop:
            existing_loop_nodes.add(loop_node)
          break

  return loops

def calculate_distance(point1, point2): #Euclidean distance formula
    return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2 + (point2[2] - point1[2])**2)

def find_center_point(coordinates):
    longest_chord = max(combinations(coordinates, 2), key=lambda pair: calculate_distance(pair[0], pair[1]))
    #print(longest_chord)

    point1, point2 = longest_chord
    diameter = calculate_distance(point1, point2)
    #print(point1,point2)

    center_point = (
        round((point1[0] + point2[0]) / 2, 5),
        round((point1[1] + point2[1]) / 2, 5),
        round((point1[2] + point2[2]) / 2, 5)
    )
    return center_point, diameter

def transpose_2d_array(matrix):
    # Use the zip function with the * operator to transpose the matrix
    transposed_matrix = list(zip(*matrix))
    # Convert the result back to a list of lists if needed
    transposed_matrix = [list(row) for row in transposed_matrix]
    return transposed_matrix

def project_points_onto_axis(points, projection_axis):
    projected_points = []
    for i, point in enumerate(points):
      projected_point = []

      (x, y, z) = point
      (ax, ay, az) = projection_axis
      if ax==0:
        projected_point.append(x)
      if ay==0:
        projected_point.append(y)
      if az==0:
        projected_point.append(z)

      projected_points.append(tuple(projected_point))
    return projected_points
        

def cmd_check(): # python | python3 hole_extraction.py /path/to/stl/file.stl /path/to/ouput/file.csv
                 # https://docs.python.org/3/library/argparse.htmlpyt

    parser = argparse.ArgumentParser(description='Finds and extracts hole center coordinates and radius from STL file and save the results in a CSV file')

    parser.add_argument('input_file', type=str, help='Path to the input STL file')
    parser.add_argument('output_file', type=str, help='Path to the output CSV file')

    args = parser.parse_args()

    if not args.input_file.endswith('.stl') or not args.output_file.endswith('.csv'):
        print('Error: Please use the format "python3 hole_extraction.py /path/to/stl/file.stl /path/to/output/file.csv"')
        sys.exit(1)

    return args.input_file, args.output_file

def main(input_file_path, output_file_path):
    facets = parse_stl_binary(input_file_path)

    holes = []
    for i, normal in enumerate(facets.keys()):
        facet_triangles = facets[normal]
        facet_graph = WeightedGraph(triangles=facet_triangles)
        for node in facet_graph.get_perimeter():
            facet_graph.remove_node(node)
        #print(facet_graph.graph.keys())
        loops = find_loops_with_equal_edge_weigths(facet_graph.graph)
        #print(f"Facet {i}")
        #print(f"Facet normal: {normal}")
        #print(f"Loops with equal edges: {loops}")

        for loop in loops:
            centroid, diameter = find_center_point(loop)
            hole = dict()
            hole["centroid"] = centroid
            hole["radius"] = round(diameter/2, 2)
            hole["normal"] = normal
            holes.append(hole)

    with open(output_file_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Hole normal", "Hole coordinates", "Hole radius"])
        for hole in holes:
           writer.writerow([hole["normal"], hole["centroid"], hole["radius"]])
    
    sys.stdout.write(f"Hole extraction completed. Output stored to file {output_file_path}.")
       

    return 0

if __name__ == "__main__":
    input_file_path, output_file_path = cmd_check()
    main(input_file_path, output_file_path)