# Hole extraction algorithm for 3D STL models

This code extracts essential information from a binary STL file related to a 3D model. It identifies existing holes and generates a CSV file containing the center coordinates and radius of the identified holes. Both through holes and blind holes are detected and outputted to file. The facet of each hole is defined through a normal vector. Currently, the algorithm is able to detect holes for all cuboid (not only cube) 3D models, which contain holes.  


## Librarys

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the required librarys.


```bash
pip install numpy
```
```bash
pip install numpy-stl
```
```bash
pip install scipy
```

## Usage

To run the code from CMD: ```python3 hole_extraction.py /path/to/stl/file.stl /path/to/output/file.csv```


## Getting Started

Follow these steps to get started with the hole extraction algorithm:

1. Install the required libraries as mentioned in the Libraries section.
2. Run the code using the provided command in the Usage section.
3. Review the generated CSV file for the extracted hole information.


## University Course Acknowledgment

This project was initiated and developed as part of University of Tartu's Algorithmics course during [2023/24 fall].
