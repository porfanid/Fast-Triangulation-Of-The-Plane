# **Fast Triangulation of the Plane**

## **📚 Project Overview**

This repository contains a robust and highly optimized Python implementation of the **Fast Triangulation of the Plane** algorithm, as described in the seminal 1985 paper by **Stefan Hertel and Kurt Mehlhorn**.  
The project aims to provide a faithful and efficient realization of their sweep-line based approach for triangulating simple polygons, including advanced optimizations and applications. This implementation was developed with a strong focus on aligning with the theoretical claims and computational complexity guarantees presented in the original research.

## **✨ Key Features & Technical Achievements**

This implementation goes beyond a basic triangulation, incorporating several advanced concepts and optimizations detailed in the Hertel & Mehlhorn paper:

* **Two Core Algorithms:**  
  * **Basic Triangulation:** An O(nlogn) sweep-line algorithm for triangulating the interior of a simple polygon.  
  * **Improved Triangulation:** An optimized O(n+slogs) sweep-line algorithm (where s is the number of "start" vertices), drastically reducing the impact of non-convexity on performance.  
* **O(1) Bend Handling:** Achieved through an innovative edge mapping system that avoids expensive BST INSERT/DELETE operations for bend points in the improved algorithm, maintaining constant-time processing for these common geometric events.  
* **Complete SPEC Integration:** Robust handling of vertical edges (SPEC cases) across all transition types (start, end, bend points), ensuring that co-linear vertical edge pairs are processed together as specified by the paper.  
* **Enhanced Applications:**  
  * **Convex Polygon Intersection:** Implements a sweep-line based approach for computing the intersection of a triangulated region with a convex polygon, moving towards the paper's claimed O(n+m) linear time complexity. Includes point location and boundary traversal techniques.  
  * **Convex Decomposition:** Decomposes a simple polygon into its constituent convex parts by intelligently merging triangles from the triangulation. This includes rigorous geometric checks for maintaining convexity at shared edges and identifying "essential" triangulation edges that resolve concave angles.  
* **Multi-Polygon Triangulation:** Supports the triangulation of multiple pairwise non-intersecting simple polygons using a unified sweep-line approach.  
* **Outer Triangulation:** Constructs the outer triangulation of a simple polygon by transforming the problem into a polygon-with-hole triangulation, utilizing a bounding box and bridge connection technique.  
* **Robust Geometric Primitives:** Includes precise implementations for point classification, orientation tests, line-segment intersections, and angle calculations, all with appropriate floating-point tolerance.  
* **Balanced Binary Search Tree (BST):** The core YStructure (sweep line status) is backed by a self-balancing BST (AVL-like), ensuring efficient O(logk) operations for insertions, deletions, and queries on active edges.

## **📄 Theoretical Basis**

This project is a direct implementation of the algorithms presented in:

* **Hertel, S., & Mehlhorn, K. (1985).** *Fast Triangulation of the Plane with Respect to Simple Polygons*. Information and Control, 64(1-3), 52-76.  
  * [Link to PDF (if publicly available, e.g., in docs/ folder)](https://www.google.com/search?q=./docs/fast_triangulation_of_the_plane.pdf)

The implementation rigorously follows the data structures (X-structure, Y-structure, C-structure, G-structure) and transition logic defined in the paper, with particular attention to the "Improved Algorithm" (Section 3\) and the "Applications" (Section 6).

## **🚀 Getting Started**

### **Prerequisites**

* Python 3.x (Tested with Python 3.8+)

### **Installation**

This project uses only standard Python libraries. No external packages are required.

1. **Clone the repository:**  
   git clone https://github.com/your-username/fast-triangulation-of-the-plane.git  
   cd fast-triangulation-of-the-plane

### **Usage**

The main implementation is in src/hertel\_mehlhorn\_triangulator.py. You can run the included test functions to see the algorithms in action.

1. **Run the main example:**  
   python src/hertel\_mehlhorn\_triangulator.py

   This will execute test\_sweep\_line\_triangulation() which demonstrates both the basic and improved triangulation algorithms, as well as the applications.  
2. **Run the comprehensive test suite:**  
   python tests/comprehensive\_tests.py

   This script runs a variety of tests, including complex polygons, edge cases, and performance analysis, providing detailed output on the algorithm's behavior and correctness.

## **🧪 Testing**

The repository includes a dedicated tests/ directory with multiple test scripts to ensure the correctness and alignment of the implementation with the Hertel & Mehlhorn paper.

* tests/test\_paper\_alignment.py: Focuses specifically on verifying the four key areas of paper alignment: O(1) bend handling, SPEC integration, enhanced applications, and multi-polygon/outer triangulation.  
* tests/comprehensive\_tests.py: Provides a broader suite of tests for complex polygon shapes, various edge cases, and includes basic complexity analysis.  
* tests/test\_improvements.py: Validates the efficiency improvements and refined logic in various components.

To run all tests, navigate to the project root and execute:  
python \-m unittest discover tests

*(Note: If you have pytest installed, you can also simply run pytest from the project root.)*

## **📂 Project Structure**

.  
├── .gitignore                      \# Specifies intentionally untracked files to ignore  
├── LICENSE.md                      \# MIT License for the project  
├── README.md                       \# This file\!  
├── PAPER\_ALIGNMENT\_SUMMARY.md      \# Detailed summary of paper alignment achievements  
├── docs/                           \# Documentation files  
│   └── fast\_triangulation\_of\_the\_plane.pdf \# The original research paper  
├── src/                            \# Main source code  
│   └── hertel\_mehlhorn\_triangulator.py \# Core implementation of the triangulation algorithms  
└── tests/                          \# Unit and integration tests  
    ├── \_\_init\_\_.py                 \# Makes 'tests' a Python package  
    ├── test\_paper\_alignment.py     \# Tests specific paper alignment criteria  
    ├── comprehensive\_tests.py      \# Comprehensive tests for various polygon types and edge cases  
    └── test\_improvements.py        \# Tests for specific algorithmic improvements

## **🔒 Archival Status**

This repository is now archived, marking the completion of the project. While no further active development is planned, the code remains available for reference, study, and potential future use. Issues and pull requests will no longer be actively monitored.

## **📜 License**

This project is licensed under the MIT License \- see the [LICENSE.md](http://docs.google.com/LICENSE.md) file for details.

## **🙏 Acknowledgements**

* **Stefan Hertel and Kurt Mehlhorn** for their foundational research on fast triangulation algorithms.  
* The computational geometry community for their invaluable resources and insights.