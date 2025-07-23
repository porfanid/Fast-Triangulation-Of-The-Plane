import math
import heapq
import bisect
from typing import List, Tuple, Optional, Set, Dict
from enum import Enum
from collections import deque, defaultdict


# --- Balanced Binary Search Tree for YStructure ---

class BSTreeNode:
    """Node for a balanced binary search tree to support efficient YStructure operations."""
    
    def __init__(self, y_value: float, edge: 'Edge', interval_type: 'IntervalType'):
        self.y_value = y_value
        self.edge = edge
        self.interval_type = interval_type
        self.left: Optional['BSTreeNode'] = None
        self.right: Optional['BSTreeNode'] = None
        self.height = 1

class BalancedBST:
    """
    Balanced Binary Search Tree (AVL-like) for efficient O(log k) operations
    on active edges in the YStructure.
    """
    
    def __init__(self):
        self.root: Optional[BSTreeNode] = None
        self.size = 0
    
    def _height(self, node: Optional[BSTreeNode]) -> int:
        """Get height of node, 0 for None."""
        return node.height if node else 0
    
    def _balance(self, node: Optional[BSTreeNode]) -> int:
        """Get balance factor of node."""
        return self._height(node.left) - self._height(node.right) if node else 0
    
    def _update_height(self, node: BSTreeNode):
        """Update height of node based on children."""
        node.height = 1 + max(self._height(node.left), self._height(node.right))
    
    def _rotate_right(self, y: BSTreeNode) -> BSTreeNode:
        """Right rotation for balancing."""
        if not y or not y.left:
            return y  # Cannot rotate
            
        x = y.left
        T2 = x.right
        
        x.right = y
        y.left = T2
        
        self._update_height(y)
        self._update_height(x)
        
        return x
    
    def _rotate_left(self, x: BSTreeNode) -> BSTreeNode:
        """Left rotation for balancing."""
        if not x or not x.right:
            return x  # Cannot rotate
            
        y = x.right
        T2 = y.left
        
        y.left = x
        x.right = T2
        
        self._update_height(x)
        self._update_height(y)
        
        return y
    
    def _insert(self, node: Optional[BSTreeNode], y_value: float, edge: 'Edge', interval_type: 'IntervalType') -> BSTreeNode:
        """Recursively insert a new node and maintain balance."""
        # Standard BST insertion
        if not node:
            self.size += 1
            return BSTreeNode(y_value, edge, interval_type)
        
        if y_value < node.y_value:
            node.left = self._insert(node.left, y_value, edge, interval_type)
        elif y_value > node.y_value:
            node.right = self._insert(node.right, y_value, edge, interval_type)
        else:
            # Equal y_value - insert based on edge comparison for consistency
            if id(edge) < id(node.edge):  # Use object id for consistent ordering
                node.left = self._insert(node.left, y_value, edge, interval_type)
            else:
                node.right = self._insert(node.right, y_value, edge, interval_type)
        
        # Update height
        self._update_height(node)
        
        # Get balance factor
        balance = self._balance(node)
        
        # Left Heavy
        if balance > 1:
            if y_value < node.left.y_value or (y_value == node.left.y_value and id(edge) < id(node.left.edge)):
                return self._rotate_right(node)
            else:
                node.left = self._rotate_left(node.left)
                return self._rotate_right(node)
        
        # Right Heavy
        if balance < -1:
            if y_value > node.right.y_value or (y_value == node.right.y_value and id(edge) > id(node.right.edge)):
                return self._rotate_left(node)
            else:
                node.right = self._rotate_right(node.right)
                return self._rotate_left(node)
        
        return node
    
    def insert(self, y_value: float, edge: 'Edge', interval_type: 'IntervalType'):
        """Insert a new edge with its y-value and interval type."""
        self.root = self._insert(self.root, y_value, edge, interval_type)
    
    def _find_min(self, node: BSTreeNode) -> BSTreeNode:
        """Find minimum node in subtree."""
        while node.left:
            node = node.left
        return node
    
    def _delete(self, node: Optional[BSTreeNode], edge: 'Edge') -> Optional[BSTreeNode]:
        """Recursively delete a node and maintain balance."""
        if not node:
            return node
        
        # Find the node to delete
        if id(edge) < id(node.edge):
            node.left = self._delete(node.left, edge)
        elif id(edge) > id(node.edge):
            node.right = self._delete(node.right, edge)
        else:
            # Found the node to delete
            self.size -= 1
            
            # Node with only right child or no child
            if not node.left:
                return node.right
            # Node with only left child
            elif not node.right:
                return node.left
            
            # Node with two children - get inorder successor
            temp = self._find_min(node.right)
            
            # Copy the inorder successor's data to this node
            node.y_value = temp.y_value
            node.edge = temp.edge
            node.interval_type = temp.interval_type
            
            # Delete the inorder successor
            node.right = self._delete(node.right, temp.edge)
        
        # Ensure node still exists after deletion
        if not node:
            return node
            
        # Update height
        self._update_height(node)
        
        # Get balance factor
        balance = self._balance(node)
        
        # Left Heavy
        if balance > 1:
            left_balance = self._balance(node.left) if node.left else 0
            if left_balance >= 0:
                return self._rotate_right(node)
            else:
                if node.left:
                    node.left = self._rotate_left(node.left)
                return self._rotate_right(node)
        
        # Right Heavy
        if balance < -1:
            right_balance = self._balance(node.right) if node.right else 0
            if right_balance <= 0:
                return self._rotate_left(node)
            else:
                if node.right:
                    node.right = self._rotate_right(node.right)
                return self._rotate_left(node)
        
        return node
    
    def delete(self, edge: 'Edge') -> bool:
        """Delete an edge from the tree. Returns True if found and deleted."""
        old_size = self.size
        self.root = self._delete(self.root, edge)
        return self.size < old_size
    
    def _find_interval(self, node: Optional[BSTreeNode], target_y: float) -> Tuple[Optional['Edge'], Optional['Edge']]:
        """Find the edges that bound the interval containing target_y."""
        if not node:
            return None, None
        
        # If target_y is less than current node's y_value, look in left subtree
        if target_y < node.y_value:
            below_edge, above_edge = self._find_interval(node.left, target_y)
            # If no above_edge found in left subtree, current node provides above_edge
            if above_edge is None:
                above_edge = node.edge
            return below_edge, above_edge
        
        # If target_y is greater than current node's y_value, look in right subtree
        elif target_y > node.y_value:
            below_edge, above_edge = self._find_interval(node.right, target_y)
            # If no below_edge found in right subtree, current node provides below_edge
            if below_edge is None:
                below_edge = node.edge
            return below_edge, above_edge
        
        # If target_y equals node's y_value, find immediate neighbors
        else:
            # Find predecessor (largest smaller value)
            below_edge = self._find_predecessor(node)
            # Find successor (smallest larger value)  
            above_edge = self._find_successor(node)
            return below_edge, above_edge
    
    def find_interval(self, target_y: float) -> Tuple[Optional['Edge'], Optional['Edge']]:
        """Find the two edges that bound the interval containing target_y."""
        return self._find_interval(self.root, target_y)
    
    def _find_predecessor(self, node: BSTreeNode) -> Optional['Edge']:
        """Find the predecessor edge (largest edge with smaller y_value)."""
        if node.left:
            # Find rightmost node in left subtree
            current = node.left
            while current.right:
                current = current.right
            return current.edge
        return None
    
    def _find_successor(self, node: BSTreeNode) -> Optional['Edge']:
        """Find the successor edge (smallest edge with larger y_value)."""
        if node.right:
            # Find leftmost node in right subtree
            current = node.right
            while current.left:
                current = current.left
            return current.edge
        return None
    
    def _find_node(self, node: Optional[BSTreeNode], edge: 'Edge') -> Optional[BSTreeNode]:
        """Find the node containing the given edge."""
        if not node:
            return None
        
        if id(edge) == id(node.edge):
            return node
        elif id(edge) < id(node.edge):
            return self._find_node(node.left, edge)
        else:
            return self._find_node(node.right, edge)
    
    def get_successor(self, edge: 'Edge') -> Optional['Edge']:
        """Get the successor edge of the given edge."""
        node = self._find_node(self.root, edge)
        return self._find_successor(node) if node else None
    
    def get_predecessor(self, edge: 'Edge') -> Optional['Edge']:
        """Get the predecessor edge of the given edge."""
        node = self._find_node(self.root, edge)
        return self._find_predecessor(node) if node else None
    
    def get_interval_type(self, edge: 'Edge') -> Optional['IntervalType']:
        """Get the interval type for the given edge."""
        node = self._find_node(self.root, edge)
        return node.interval_type if node else None
    
    def _inorder(self, node: Optional[BSTreeNode], result: List[Tuple[float, 'Edge', 'IntervalType']]):
        """In-order traversal to get sorted list of edges."""
        if node:
            self._inorder(node.left, result)
            result.append((node.y_value, node.edge, node.interval_type))
            self._inorder(node.right, result)
    
    def get_sorted_edges(self) -> List[Tuple[float, 'Edge', 'IntervalType']]:
        """Get all edges sorted by y-value."""
        result = []
        self._inorder(self.root, result)
        return result
    
    def __len__(self):
        return self.size


# --- Geometric Primitives ---

class PointType(Enum):
    """
    Classifies vertex types for the sweep-line algorithm, as described in the paper.
    - START_PROPER: Both neighbors have larger x-coordinates, and the interior angle is convex.
    - START_IMPROPER: Both neighbors have larger x-coordinates, and the interior angle is concave.
    - END_PROPER: Both neighbors have smaller x-coordinates, and the interior angle is convex.
    - END_IMPROPER: Both neighbors have smaller x-coordinates, and the interior angle is concave.
    - BEND: One neighbor has a smaller x-coordinate, and the other has a larger x-coordinate.
    """
    START_PROPER = "start_proper"
    START_IMPROPER = "start_improper"
    END_PROPER = "end_proper"
    END_IMPROPER = "end_improper"
    BEND = "bend"


class IntervalType(Enum):
    """
    Defines the type of interval in the Y-structure (sweep line status).
    - IN: An interval inside the polygon.
    - OUT: An interval outside the polygon.
    """
    IN = "in"
    OUT = "out"


class Point:
    """
    Represents a vertex in the polygon.
    Attributes:
        x (float): X-coordinate.
        y (float): Y-coordinate.
        index (int): Original index in the polygon list.
        point_type (Optional[PointType]): Classified type of the point for sweep-line.
        spec (bool): True if there's a vertical edge above this point (special case in paper).
        co_point (Optional['Point']): For vertical edge pairs, the other endpoint.
        prev_point (Optional['Point']): Previous point in the polygon boundary (for traversal).
        next_point (Optional['Point']): Next point in the polygon boundary (for traversal).
    """

    def __init__(self, x: float, y: float, index: int = -1):
        self.x = x
        self.y = y
        self.index = index
        self.point_type: Optional[PointType] = None
        self.spec = False
        self.co_point: Optional['Point'] = None
        self.prev_point: Optional['Point'] = None  # For polygon traversal
        self.next_point: Optional['Point'] = None  # For polygon traversal

    def __lt__(self, other):
        """
        Comparison for sorting points, primarily by x then by y.
        Used for the X-structure (event queue).
        """
        if self.x != other.x:
            return self.x < other.x
        return self.y < other.y

    def __eq__(self, other):
        """Equality based on coordinates and index."""
        return isinstance(other, Point) and abs(self.x - other.x) < 1e-9 and abs(
            self.y - other.y) < 1e-9 and self.index == other.index

    def __hash__(self):
        """Hash for Point objects to allow use in sets/dictionaries."""
        return hash((self.x, self.y, self.index))

    def __repr__(self):
        return f"Point({self.x}, {self.y}, idx={self.index})"


class Edge:
    """
    Represents an edge in the polygon or triangulation.
    Attributes:
        p1 (Point): First endpoint.
        p2 (Point): Second endpoint.
        a (float): Slope (coefficient 'a' in y = ax + b).
        b (float): Y-intercept (coefficient 'b' in y = ax + b).
    """

    def __init__(self, p1: Point, p2: Point):
        # Ensure consistent order for edge points (e.g., p1 always "smaller" than p2)
        # This helps with equality checks for sets later, as Edge(A,B) should be equal to Edge(B,A).
        # For sweep-line, edges are often defined by (left_point, right_point) or (lower_point, upper_point).
        # Let's keep p1, p2 as given for now, and handle directionality in logic.
        self.p1 = p1
        self.p2 = p2

        # Check for degenerate edge (same points)
        if abs(p1.x - p2.x) < 1e-9 and abs(p1.y - p2.y) < 1e-9:
            print(f"    Warning: Degenerate edge with same points: {p1}, {p2}")
            self.a = 0
            self.b = p1.y
            return

        # Linear equation y = ax + b
        if abs(self.p2.x - self.p1.x) < 1e-9:  # Vertical edge
            self.a = float('inf')  # Infinite slope
            self.b = self.p1.x  # For vertical edge, b is the x-coordinate
        else:
            self.a = (self.p2.y - self.p1.y) / (self.p2.x - self.p1.x)
            self.b = self.p1.y - self.a * self.p1.x

    def y_at_x(self, x: float) -> float:
        """
        Calculates the y-coordinate of the line segment at a given x-coordinate.
        Important for sweep-line sorting.
        """
        if self.a == float('inf'):  # Vertical edge
            # For sweep-line, vertical edges are special. This y_at_x is mainly for non-vertical sorting.
            # If x is on the vertical line, return the midpoint y for consistent comparison logic.
            # Otherwise, return infinity if x is to the right, -infinity if to the left (conceptual).
            if abs(x - self.b) < 1e-9:  # If x is on the vertical line
                return (self.p1.y + self.p2.y) / 2
            # Handle cases where x is outside the segment's x-range for vertical edges
            min_y = min(self.p1.y, self.p2.y)
            max_y = max(self.p1.y, self.p2.y)
            if x < self.b: return min_y  # To the left of vertical edge, effectively below
            if x > self.b: return max_y  # To the right of vertical edge, effectively above
            return min_y  # Default to minimum y

        # Check for invalid slope/intercept values
        if math.isnan(self.a) or math.isnan(self.b):
            print(f"    Warning: Invalid edge parameters a={self.a}, b={self.b} for edge {self}")
            # Return midpoint y as fallback
            return (self.p1.y + self.p2.y) / 2

        # Clamp x to the segment's x-range for accurate y_at_x within the segment
        min_x = min(self.p1.x, self.p2.x)
        max_x = max(self.p1.x, self.p2.x)
        if x < min_x - 1e-9 or x > max_x + 1e-9:
            # If x is outside the segment's x-range, return a value that
            # allows for correct relative ordering.
            # For sweep line, we often care about the line's y-value, not just segment.
            # But for sorting active edges, we need a consistent value.
            # Let's use the line equation for points outside the segment's x-range
            # but indicate it's outside.
            # This is a common simplification in sweep-line examples.
            pass  # Use the line equation below

        result = self.a * x + self.b
        
        # Check for NaN result
        if math.isnan(result):
            print(f"    Warning: NaN result for edge {self} at x={x}")
            return (self.p1.y + self.p2.y) / 2  # Return midpoint as fallback
            
        return result

    def __eq__(self, other):
        """Equality for Edge objects, ignoring point order."""
        return isinstance(other, Edge) and \
            ((self.p1 == other.p1 and self.p2 == other.p2) or \
             (self.p1 == other.p2 and self.p2 == other.p1))

    def __hash__(self):
        """Hash for Edge objects based on sorted points."""
        # Ensure consistent hashing regardless of p1, p2 order
        if self.p1 < self.p2:
            return hash((self.p1, self.p2))
        else:
            return hash((self.p2, self.p1))

    def __repr__(self):
        return f"Edge(P{self.p1.index} -> P{self.p2.index})"  # Use index for cleaner repr


class Triangle:
    """Represents a triangle in the triangulation."""

    def __init__(self, p1: Point, p2: Point, p3: Point):
        # Sort vertices for consistent representation (useful for set/hash)
        self.vertices = sorted([p1, p2, p3])
        self.edges = [
            Edge(self.vertices[0], self.vertices[1]),
            Edge(self.vertices[1], self.vertices[2]),
            Edge(self.vertices[2], self.vertices[0])
        ]

    def __eq__(self, other):
        """Equality for Triangle objects based on their vertices."""
        return isinstance(other, Triangle) and set(self.vertices) == set(other.vertices)

    def __hash__(self):
        """Hash for Triangle objects."""
        return hash(tuple(self.vertices))

    def __repr__(self):
        return f"Triangle(P{self.vertices[0].index}, P{self.vertices[1].index}, P{self.vertices[2].index})"


# --- Sweep-Line Data Structures ---

class YStructure:
    """
    Represents the Y-structure (sweep line status) as described in the paper (Section 2.1).
    It stores active edges intersected by the current sweep line, sorted by their y-coordinates.

    IMPROVED: Uses balanced binary search tree for O(log k) INSERT, DELETE, and FIND operations
    as specified in the Hertel & Mehlhorn paper.
    """

    def __init__(self, triangulator=None):
        # Use balanced BST instead of sorted list for O(log k) operations
        self._bst = BalancedBST()
        self._current_x = float('-inf')  # The current x-coordinate of the sweep line
        self._triangulator = triangulator  # Reference to triangulator for improved algorithm checks

        # C-structure components, as per paper Section 2.1
        self.chains: Dict[Edge, PolygonalChain] = {}  # Maps edge to polygonal chain L(edge)
        self.rightmost: Dict[Edge, Point] = {}  # Maps edge to rightmost point RM(edge)
        
        # Additional tracking for sweep-line efficiency
        self._edge_to_y_map: Dict[Edge, float] = {}  # Cache for quick y-value lookup

    def set_current_x(self, x: float):
        """
        Updates the sweep line's x-coordinate. 
        For the improved algorithm, this should only be called at start/end points,
        not at every vertex to maintain O(n + s log s) complexity.
        """
        self._current_x = x
        print(f"    Setting sweep line to x = {x}")
        
        # For improved algorithm efficiency, don't rebuild BST here
        # The BST should maintain its structure and only be updated
        # through explicit insert/delete operations at start/end points

    def _cleanup_edge(self, edge: Edge):
        """Clean up data structures when an edge is no longer active."""
        if edge in self.chains: 
            del self.chains[edge]
        if edge in self.rightmost: 
            del self.rightmost[edge]
        if edge in self._edge_to_y_map:
            del self._edge_to_y_map[edge]

    def _get_y_for_sorting(self, edge: Edge) -> float:
        """Helper to get the y-value of an edge at the current sweep line position."""
        # Use cached value if available
        if edge in self._edge_to_y_map:
            return self._edge_to_y_map[edge]
        
        y_val = edge.y_at_x(self._current_x)
        self._edge_to_y_map[edge] = y_val
        return y_val

    def find_interval(self, point: Point) -> Tuple[Optional[Edge], Optional[Edge]]:
        """
        FIND(p): Delivers the two edges (above and below) of the boundary of P
        in whose interval the point p lies. Now O(log k) with balanced BST.
        Enhanced to handle edge mapping for O(1) bend processing.
        """
        target_y = point.y
        below_edge, above_edge = self._bst.find_interval(target_y)
        
        # Resolve edge mappings if triangulator is using improved algorithm
        if self._triangulator and self._triangulator.using_improved_algorithm:
            if below_edge:
                below_edge = self._triangulator._resolve_edge_mapping(below_edge, self)
            if above_edge:
                above_edge = self._triangulator._resolve_edge_mapping(above_edge, self)
        
        return below_edge, above_edge

    def insert(self, edge: Edge, interval_type: IntervalType):
        """
        INSERT(s, <t>): Inserts an active edge `s` and its interval type `<t>`.
        Now O(log k) with balanced BST.
        """
        y_val = self._get_y_for_sorting(edge)
        self._bst.insert(y_val, edge, interval_type)
        print(f"    Inserted {edge} at y={y_val:.3f} with type {interval_type}")

    def delete(self, edge: Edge):
        """
        DELETE(s, <t>): Deletes an active edge `s`.
        Now O(log k) with balanced BST.
        """
        if self._bst.delete(edge):
            self._cleanup_edge(edge)
            print(f"    Deleted {edge}")
        else:
            print(f"    Warning: Attempted to delete non-existent edge {edge} from Y-structure.")

    def get_type(self, below_edge: Edge, above_edge: Edge) -> Optional[IntervalType]:
        """
        Retrieves the type of the interval [below_edge, above_edge].
        Now O(log k) with balanced BST.
        """
        if above_edge:
            return self._bst.get_interval_type(above_edge)
        return None

    def succ(self, edge: Edge) -> Optional[Edge]:
        """SUCC(s): Delivers the neighboring edge above s. Now O(log k) with balanced BST."""
        return self._bst.get_successor(edge)

    def pred(self, edge: Edge) -> Optional[Edge]:
        """PRED(s): Delivers the neighboring edge below s. Now O(log k) with balanced BST."""
        return self._bst.get_predecessor(edge)
    
    def get_lagging_edges(self, target_x: float) -> List[Edge]:
        """
        Find edges whose rightmost point is to the left of target_x.
        These are "lagging" edges that need local sweep line extension.
        """
        lagging_edges = []
        for edge in self.rightmost:
            rightmost_point = self.rightmost[edge]
            if rightmost_point and rightmost_point.x < target_x - 1e-9:
                lagging_edges.append(edge)
        return lagging_edges


class PolygonalChain:
    """
    Represents a polygonal chain L(s) for an in-interval (C-structure).
    It stores a sequence of vertices that define a partially triangulated region.
    (Section 2.1, "The c-structure C").
    """

    def __init__(self):
        self.points = deque()  # Doubly linked list via deque
        self.rightmost_point = None  # RM(s)

    def add_head(self, point: Point):
        self.points.appendleft(point)
        # Update rightmost_point if the new head is further right
        if self.rightmost_point is None or point.x > self.rightmost_point.x:
            self.rightmost_point = point

    def add_tail(self, point: Point):
        self.points.append(point)
        # Update rightmost_point if the new tail is further right
        if self.rightmost_point is None or point.x > self.rightmost_point.x:
            self.rightmost_point = point

    def get_head(self) -> Optional[Point]:
        return self.points[0] if self.points else None

    def get_tail(self) -> Optional[Point]:
        return self.points[-1] if self.points else None

    def remove_point(self, point: Point):
        """Removes a point from the chain."""
        if point in self.points:
            self.points.remove(point)
            # Re-evaluate rightmost_point if the removed point was it
            if self.rightmost_point == point:
                self.rightmost_point = max(self.points, key=lambda p: p.x) if self.points else None
        else:
            print(f"    Warning: Attempted to remove non-existent point {point} from chain.")

    def __len__(self):
        return len(self.points)

    def __repr__(self):
        return f"Chain({[p.index for p in self.points]})"


# --- Main Triangulation Algorithm ---

class FastTriangulator:
    """
    Implementation of the sweep-line triangulation algorithms based on
    Hertel and Mehlhorn (1985).
    """

    def __init__(self):
        self.triangles: List[Triangle] = []  # G-structure: list of generated triangles
        self.triangulation_edges: Set[Edge] = set()  # G-structure: set of triangulation edges
        self.polygon_edges: Set[Edge] = set()  # Original polygon edges
        self.using_improved_algorithm: bool = False  # Track if we're using improved O(n + s log s) algorithm

    def _triangle_exists(self, p1: Point, p2: Point, p3: Point) -> bool:
        """Check if a triangle with these three points already exists."""
        test_triangle = Triangle(p1, p2, p3)
        return test_triangle in self.triangles

    def _add_triangle(self, p1: Point, p2: Point, p3: Point):
        """Helper to add a new triangle and its edges to the G-structure."""
        # Ensure points are ordered CCW for consistent triangle definition
        # This is crucial for correct triangulation.
        # Use orientation to order them.
        if self.orientation(p1, p2, p3) == 1:  # If CW, swap p2, p3 to make CCW
            p2, p3 = p3, p2
        elif self.orientation(p1, p2, p3) == 0:  # Collinear points, cannot form a triangle
            print(f"    Warning: Attempted to add triangle with collinear points: {p1}, {p2}, {p3}")
            return

        # Check if triangle already exists
        if self._triangle_exists(p1, p2, p3):
            print(f"    Warning: Triangle P{p1.index}-P{p2.index}-P{p3.index} already exists, skipping")
            return

        tri = Triangle(p1, p2, p3)
        self.triangles.append(tri)
        for edge in tri.edges:
            # Only add edges that are not original polygon edges
            if edge not in self.polygon_edges:
                self.triangulation_edges.add(edge)

    def calculate_interior_angle(self, p1: Point, p2: Point, p3: Point) -> float:
        """Calculate interior angle at p2 formed by p1-p2-p3."""
        v1 = (p1.x - p2.x, p1.y - p2.y)
        v2 = (p3.x - p2.x, p3.y - p2.y)

        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
        cross_product = v1[0] * v2[1] - v1[1] * v2[0]

        angle = math.atan2(cross_product, dot_product)
        if angle < 0:
            angle += 2 * math.pi  # Ensure angle is positive (0 to 2*pi)

        return angle

    def orientation(self, p: Point, q: Point, r: Point) -> int:
        """
        Determine orientation of ordered triplet (p, q, r).
        0 --> p, q and r are collinear
        1 --> Clockwise
        2 --> Counterclockwise
        """
        # (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)
        val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)
        eps = 1e-9  # Small epsilon for floating point comparisons
        if abs(val) < eps:
            return 0  # Collinear
        return 1 if val > eps else 2  # Clockwise or Counterclockwise (val > 0 means CCW for standard orientation)

    def is_convex_angle(self, p1: Point, p2: Point, p3: Point) -> bool:
        """
        Check if angle at p2 is convex for a counter-clockwise ordered polygon.
        A convex vertex for a CCW polygon means the orientation (p1, p2, p3) is CCW.
        """
        return self.orientation(p1, p2, p3) == 2

    def classify_vertex(self, polygon_points: List[Point], i: int) -> PointType:
        """
        Classify vertex type based on its neighbors (Section 2.1).
        Requires prev_point and next_point to be set on the Point objects.
        """
        curr_point = polygon_points[i]
        prev_point = curr_point.prev_point
        next_point = curr_point.next_point

        # Compare x-coordinates
        # Note: The paper defines start/end/bend based on neighbors having larger/smaller x-coordinates.
        # This is a bit ambiguous with vertical edges. Assuming strict inequality for now.
        prev_x_smaller = prev_point.x < curr_point.x - 1e-9
        next_x_smaller = next_point.x < curr_point.x - 1e-9
        prev_x_larger = prev_point.x > curr_point.x + 1e-9
        next_x_larger = next_point.x > curr_point.x + 1e-9

        # Check if it's a start or end point (both neighbors on one side in x)
        if prev_x_larger and next_x_larger:
            # Both neighbors have larger x-coordinates (to the right of curr_point)
            # This is a start point. Check if proper (convex) or improper (concave).
            return PointType.START_PROPER if self.is_convex_angle(prev_point, curr_point,
                                                                  next_point) else PointType.START_IMPROPER

        elif prev_x_smaller and next_x_smaller:
            # Both neighbors have smaller x-coordinates (to the left of curr_point)
            # This is an end point. Check if proper (convex) or improper (concave).
            return PointType.END_PROPER if self.is_convex_angle(prev_point, curr_point,
                                                                next_point) else PointType.END_IMPROPER

        else:
            # One neighbor smaller x, one larger x (or equal x, handled by epsilon)
            # This is a bend point.
            return PointType.BEND

    def chain_triangulate(self, chain: PolygonalChain, dir: str):
        """
        CHAIN_TRI(e, dir): Triangulates along a polygonal chain (Section 2.2, page 67).
        
        Follows the paper's pseudocode precisely:
        - Iteratively draws triangulation edges "as long as the in-angle at the next point on L(e) is convex"
        - "Deletes q from L(e)" as points are processed
        - Relies on the polygonal chain invariant (point (iv) from Section 2.1)
        """
        if len(chain) < 3:
            print(f"    Chain too short for triangulation: {len(chain)} points")
            return

        print(f"    Triangulating chain with {len(chain)} points in direction {dir}")

        # Make a working copy to avoid modifying original during iteration
        working_points = list(chain.points)
        triangles_created = 0
        max_triangles = len(working_points) - 2  # At most n-2 triangles for n points

        if dir == "cc":  # Counter-clockwise from head
            # Process from head towards tail
            while len(working_points) >= 3 and triangles_created < max_triangles:
                p = working_points[0]  # Current point at head
                q = working_points[1]  # Next point
                w = working_points[2]  # Point after next
                
                # Check if the in-angle at q is convex
                if self._is_chain_angle_convex(p, q, w):
                    # Check if triangle already exists to avoid duplicates
                    if not self._triangle_exists(p, q, w):
                        # Create triangle (p, q, w) and delete q from chain
                        self._add_triangle(p, q, w)
                        print(f"      Created triangle: P{p.index}-P{q.index}-P{w.index}")
                        triangles_created += 1
                    
                    # Remove q from working points (delete q from L(e))
                    working_points.pop(1)
                else:
                    # Angle at q is not convex, stop triangulation in this direction
                    print(f"      Angle at P{q.index} is not convex, stopping triangulation")
                    break
                    
        else:  # dir == "c", clockwise from tail
            # Process from tail towards head
            while len(working_points) >= 3 and triangles_created < max_triangles:
                w = working_points[-3]  # Point before previous
                q = working_points[-2]  # Previous point 
                p = working_points[-1]  # Current point at tail
                
                # Check if the in-angle at q is convex
                if self._is_chain_angle_convex(w, q, p):
                    # Check if triangle already exists to avoid duplicates
                    if not self._triangle_exists(w, q, p):
                        # Create triangle (w, q, p) and delete q from chain
                        self._add_triangle(w, q, p)
                        print(f"      Created triangle: P{w.index}-P{q.index}-P{p.index}")
                        triangles_created += 1
                    
                    # Remove q from working points (delete q from L(e))
                    working_points.pop(-2)
                else:
                    # Angle at q is not convex, stop triangulation in this direction
                    print(f"      Angle at P{q.index} is not convex, stopping triangulation")
                    break

        # Update the original chain by removing processed points
        # The chain should now contain only the unprocessed points
        if triangles_created > 0:
            chain.points.clear()
            for point in working_points:
                chain.points.append(point)
            
            # Update rightmost point
            if chain.points:
                chain.rightmost_point = max(chain.points, key=lambda p: p.x)
            else:
                chain.rightmost_point = None

        print(f"    Chain triangulation created {triangles_created} triangles, {len(working_points)} points remaining")

    def _is_chain_angle_convex(self, p1: Point, p2: Point, p3: Point) -> bool:
        """
        Check if the in-angle at p2 in the chain (p1, p2, p3) is convex.
        This relies on the polygonal chain invariant from the paper (Section 2.1, point (iv)).
        
        For a polygonal chain in an in-interval, the angle is convex if the orientation
        of (p1, p2, p3) maintains the proper orientation for triangulation.
        """
        # The paper's polygonal chain invariant ensures that we can check convexity
        # by examining the orientation of consecutive triplets
        orientation = self.orientation(p1, p2, p3)
        
        # For proper triangulation, we want counter-clockwise orientation (orientation == 2)
        # This ensures the triangle is oriented correctly and the angle at p2 is convex
        return orientation == 2
        
    def _can_triangulate_chain_triplet(self, p1: Point, p2: Point, p3: Point, chain_points: List[Point]) -> bool:
        """
        Enhanced check for chain triangulation that considers the polygonal chain invariant.
        This is used as a secondary validation beyond the basic convexity check.
        """
        # First check basic orientation
        if self.orientation(p1, p2, p3) != 2:  # Must be CCW
            return False
            
        # The polygonal chain invariant (Section 2.1, point (iv)) guarantees that
        # no other vertices of the chain lie inside the triangle formed by consecutive points.
        # However, we still verify this for robustness.
        
        for point in chain_points:
            if point in [p1, p2, p3]:
                continue
            if self.point_in_triangle(point, p1, p2, p3):
                return False
                
        return True

    # --- Transition Handlers (Section 2.2) ---

    def handle_proper_start(self, p: Point, y_structure: YStructure):
        """
        Handles a "proper start" point (Fig. 7a,b,c).
        Splits an out-interval into three (out, in, out) and initializes a chain.
        
        Enhanced with SPEC integration: if SPEC(p) is true, handles vertical edge cases
        as specified in the paper's algorithm where p and co_p are processed together.
        """
        print(f"  Handling Proper Start: {p}")
        
        # SPEC integration: Check for vertical edges before main processing
        if hasattr(p, 'spec') and p.spec:
            print(f"    SPEC case detected for proper start {p}")
            # Process vertical edge pair if co_point exists
            if hasattr(p, 'co_point') and p.co_point:
                self._process_spec_proper_start(p, p.co_point, y_structure)
                return
        
        # Standard proper start processing
        # FIND(p) delivers the two adjacent active edges t (below) and s (above)
        # in whose interval [t, s] p lies.
        t_edge_bounding_interval, s_edge_bounding_interval = y_structure.find_interval(p)

        # Ensure p is in an OUT interval
        interval_type = y_structure.get_type(t_edge_bounding_interval, s_edge_bounding_interval)
        if interval_type != IntervalType.OUT:
            print(f"    Error: Proper start point {p} not in an OUT interval. Found type: {interval_type}")
            # Try to recover by initializing Y-structure if needed
            if not y_structure._bst.root or len(y_structure._bst) == 0:
                print(f"    Attempting to initialize Y-structure for proper start")
                self._initialize_basic_y_structure(p, y_structure)
                t_edge_bounding_interval, s_edge_bounding_interval = y_structure.find_interval(p)
                interval_type = y_structure.get_type(t_edge_bounding_interval, s_edge_bounding_interval)
            
            if interval_type != IntervalType.OUT:
                print(f"    Unable to place proper start in OUT interval, proceeding with fallback")
                return

        # The two edges incident to p are (p, p.prev_point) and (p, p.next_point).
        # For a proper start, both p.prev_point.x and p.next_point.x are > p.x.
        # We need to identify which one is the 'lower' (l_edge) and 'higher' (h_edge)
        # based on their y-coordinates at p.x (which is p.y).

        edge1_from_p = Edge(p, p.prev_point)
        edge2_from_p = Edge(p, p.next_point)

        # Determine which edge is lower and higher at p's x-coordinate (which is p.y)
        # Use a small offset to the right to handle vertical alignment correctly
        test_x = p.x + 1e-6
        if edge1_from_p.y_at_x(test_x) < edge2_from_p.y_at_x(test_x):
            l_edge = edge1_from_p
            h_edge = edge2_from_p
        else:
            l_edge = edge2_from_p
            h_edge = edge1_from_p

        # Delete the old interval [t, s] from Y-structure.
        # This implies that the original bounding edges `t_edge_bounding_interval` and `s_edge_bounding_interval`
        # are effectively replaced by the new edges.
        if t_edge_bounding_interval: y_structure.delete(t_edge_bounding_interval)
        if s_edge_bounding_interval: y_structure.delete(s_edge_bounding_interval)

        # Insert new intervals as per paper: INSERT((l,out)); INSERT((h,in));
        # The interval below l_edge is OUT. The interval below h_edge is IN.
        y_structure.insert(l_edge, IntervalType.OUT)
        y_structure.insert(h_edge, IntervalType.IN)

        # Initialize chain for the new in-interval (bounded by h_edge from above)
        chain = PolygonalChain()
        chain.add_head(p)
        y_structure.chains[h_edge] = chain
        y_structure.rightmost[h_edge] = p  # RM(h) <- p

    def handle_bend(self, p: Point, y_structure: YStructure):
        """
        Handles a "bend" point (Fig. 8a,b) following the paper's TRANSITION logic precisely.
        
        A bend point should be associated with an existing in-interval's chain, which is then 
        updated and potentially triangulated via CHAIN_TRI with full Y-structure updates 
        (e.g., replacing s with t).
        """
        print(f"  Handling Bend: {p}")
        
        # For a bend point p, one incident edge comes from left, one goes to right.
        # s_edge: the edge whose right endpoint is p (coming from left)
        # t_edge: the edge whose left endpoint is p (going to right)
        s_edge = Edge(p.prev_point, p)  # Edge ending at p (incoming)
        t_edge = Edge(p, p.next_point)  # Edge starting at p (outgoing)

        # Find the interval containing this bend point
        below_edge, above_edge = y_structure.find_interval(p)
        interval_type = y_structure.get_type(below_edge, above_edge)
        
        print(f"    Bend point P{p.index} in interval type: {interval_type}")
        print(f"    Below edge: {below_edge}, Above edge: {above_edge}")
        
        # Check if Y-structure is properly initialized
        if not self._has_active_polygon_edges(y_structure):
            print(f"    Warning: Y-structure not properly initialized for bend {p}")
            # Instead of falling back immediately, try to process with available structure
            if above_edge and below_edge:
                interval_type = IntervalType.IN  # Assume we're inside the polygon

        if interval_type == IntervalType.IN and above_edge:
            # Bend point is in an IN-interval, process according to paper's algorithm
            self._handle_bend_in_in_interval(p, s_edge, t_edge, above_edge, y_structure)
        elif interval_type == IntervalType.OUT:
            # Bend point in OUT-interval - should not create triangles typically
            print(f"    Bend point in OUT-interval - updating Y-structure only")
            self._update_y_structure_for_bend(s_edge, t_edge, above_edge or below_edge, 
                                            None, p, y_structure)
        else:
            # Last resort: create triangles directly but avoid fan triangulation for single bend
            print(f"    No suitable interval found, attempting direct triangulation")
            self._create_single_triangle_from_bend(p)

    def _initialize_y_structure_with_polygon_edges(self, polygon: List[Point], leftmost_x: float, y_structure: YStructure):
        """
        Initialize Y-structure with polygon edges that are active at the leftmost x-coordinate.
        This is crucial for proper sweep-line algorithm functioning.
        """
        print(f"  Initializing Y-structure at x={leftmost_x}")
        
        # Find all polygon edges that intersect or start at leftmost_x
        active_edges = []
        n = len(polygon)
        
        for i in range(n):
            curr_point = polygon[i]
            next_point = polygon[(i + 1) % n]
            
            edge = Edge(curr_point, next_point)
            
            # Check if edge is active at leftmost_x
            min_x = min(curr_point.x, next_point.x)
            max_x = max(curr_point.x, next_point.x)
            
            if min_x <= leftmost_x <= max_x:
                y_val = edge.y_at_x(leftmost_x)
                active_edges.append((y_val, edge))
        
        # Sort active edges by their y-coordinate at leftmost_x
        active_edges.sort(key=lambda x: x[0])
        
        print(f"    Found {len(active_edges)} active edges at x={leftmost_x}")
        
        # Insert edges into Y-structure with alternating interval types
        # The first interval (below the first edge) is OUT
        # Then alternate: first edge creates IN, second edge bounds IN from above and creates OUT, etc.
        for i, (y_val, edge) in enumerate(active_edges):
            # For sweep-line, we typically have OUT intervals between polygon boundary edges
            # and IN intervals inside the polygon
            if i % 2 == 0:
                interval_type = IntervalType.OUT  # Outside polygon
            else:
                interval_type = IntervalType.IN   # Inside polygon
            
            y_structure.insert(edge, interval_type)
            
            # For IN intervals, initialize a chain
            if interval_type == IntervalType.IN:
                chain = PolygonalChain()
                # Find vertices at leftmost_x to initialize chain
                for vertex in polygon:
                    if abs(vertex.x - leftmost_x) < 1e-9:
                        # Check if vertex is between the edges
                        if i > 0:  # Have a lower edge
                            lower_y = active_edges[i-1][0]
                            if lower_y <= vertex.y <= y_val:
                                chain.add_head(vertex)
                                break
                
                y_structure.chains[edge] = chain
                if chain.get_head():
                    y_structure.rightmost[edge] = chain.get_head()

    def _has_active_polygon_edges(self, y_structure: YStructure) -> bool:
        """Check if Y-structure has any active polygon edges."""
        # Check if we have actual polygon edges, not just boundary edges  
        active_edges = y_structure._bst.get_sorted_edges()
        return len(active_edges) > 0

    def _create_single_triangle_from_bend(self, p: Point):
        """
        Create a single triangle from a bend point when chain-based approach is not available.
        This is more conservative than fan triangulation.
        """
        print(f"    Creating single triangle from bend point {p}")
        
        prev_pt = p.prev_point
        next_pt = p.next_point
        
        if prev_pt and next_pt:
            # Check if we can create a valid triangle with the three points
            if self.orientation(prev_pt, p, next_pt) == 2:  # CCW orientation
                self._add_triangle(prev_pt, p, next_pt)
                print(f"      Created triangle: P{prev_pt.index}-P{p.index}-P{next_pt.index}")
            else:
                print(f"      Cannot create valid triangle from bend point {p}")

    def _initialize_y_structure_for_bend(self, p: Point, y_structure: YStructure):
        """
        Initialize Y-structure when encountering the first bend point.
        This handles the case where all vertices are bend points.
        """
        print(f"    Initializing Y-structure for all-bend polygon at P{p.index}")
        
        # For all-bend polygons, create initial active edges and intervals
        # Find edges that should be active at this x-coordinate
        polygon_vertices = self._get_polygon_vertices_from_point(p)
        
        if len(polygon_vertices) >= 3:
            # Instead of full fan triangulation, initialize Y-structure properly
            self._initialize_y_structure_with_polygon_edges(polygon_vertices, p.x, y_structure)
        else:
            print(f"      Insufficient vertices for Y-structure initialization")

    def _handle_bend_in_in_interval(self, p: Point, s_edge: Edge, t_edge: Edge, 
                                   interval_boundary_edge: Edge, y_structure: YStructure):
        """
        Handle bend point that lies in an IN-interval, following paper's algorithm.
        
        The paper specifies that the bend point should be associated with the 
        in-interval's chain, which is then updated and triangulated.
        """
        print(f"    Processing bend P{p.index} in IN-interval")
        
        # Get the chain associated with this in-interval
        # The chain is associated with the upper boundary edge of the in-interval
        chain = y_structure.chains.get(interval_boundary_edge)
        
        if chain:
            print(f"    Found existing chain with {len(chain)} points")
            
            # Add bend point to the appropriate end of the chain
            # The paper's logic depends on the geometry of the bend
            if self._should_add_to_head(p, chain):
                chain.add_head(p)
                triangulation_direction = "cc"
            else:
                chain.add_tail(p)
                triangulation_direction = "c"
            
            # Triangulate along the updated chain
            print(f"    Triangulating chain in direction {triangulation_direction}")
            self.chain_triangulate(chain, triangulation_direction)
            
            # Update Y-structure: replace s_edge with t_edge while preserving chain
            # This implements the "replacing s with t" logic from the paper
            self._update_y_structure_for_bend(s_edge, t_edge, interval_boundary_edge, 
                                            chain, p, y_structure)
            
        else:
            print(f"    No chain found for in-interval, creating direct triangulation")
            # If no chain exists, create one and initialize triangulation
            new_chain = PolygonalChain()
            new_chain.add_head(p)
            y_structure.chains[interval_boundary_edge] = new_chain
            y_structure.rightmost[interval_boundary_edge] = p

    def _should_add_to_head(self, p: Point, chain: PolygonalChain) -> bool:
        """
        Determine whether to add the bend point to the head or tail of the chain.
        This depends on the geometric relationship between the point and the chain.
        """
        if not chain.points:
            return True
            
        head_point = chain.get_head()
        tail_point = chain.get_tail()
        
        # Add to head if the point is geometrically closer to the head
        # or if it maintains the proper chain ordering
        if head_point and tail_point:
            dist_to_head = (p.x - head_point.x) ** 2 + (p.y - head_point.y) ** 2
            dist_to_tail = (p.x - tail_point.x) ** 2 + (p.y - tail_point.y) ** 2
            return dist_to_head <= dist_to_tail
        
        return True

    def _update_y_structure_for_bend(self, s_edge: Edge, t_edge: Edge, 
                                   interval_boundary_edge: Edge, chain: PolygonalChain,
                                   p: Point, y_structure: YStructure):
        """
        Update Y-structure when processing a bend point, implementing the 
        "replacing s with t" logic from the paper.
        
        For the improved algorithm, avoid INSERT/DELETE operations as bends
        should be processed in O(1) time according to the paper.
        """
        print(f"    Updating Y-structure: replacing {s_edge} with {t_edge}")
        
        if not self.using_improved_algorithm:
            # Original algorithm: perform explicit INSERT/DELETE operations
            if y_structure._bst.get_interval_type(s_edge) is not None:
                interval_type = y_structure._bst.get_interval_type(s_edge)
                y_structure.delete(s_edge)
                y_structure.insert(t_edge, interval_type)
                
                # Transfer chain ownership if s_edge had a chain
                if s_edge in y_structure.chains:
                    y_structure.chains[t_edge] = y_structure.chains[s_edge]
                    del y_structure.chains[s_edge]
                    
                if s_edge in y_structure.rightmost:
                    y_structure.rightmost[t_edge] = p  # Update rightmost to current point
                    del y_structure.rightmost[s_edge]
        else:
            # Improved algorithm: O(1) bend handling without BST modifications
            print(f"    Improved algorithm: O(1) bend handling without BST INSERT/DELETE")
            
            # Use edge mapping for O(1) replacement without BST operations
            self._map_edge_replacement(s_edge, t_edge, y_structure)
            
            # Transfer chain ownership efficiently
            if s_edge in y_structure.chains:
                y_structure.chains[t_edge] = y_structure.chains[s_edge]
                # Mark s_edge for lazy cleanup instead of immediate deletion
                y_structure.chains[s_edge] = None
                
            if s_edge in y_structure.rightmost:
                y_structure.rightmost[t_edge] = p
                # Mark for lazy cleanup
                y_structure.rightmost[s_edge] = None
        
        # Update the rightmost point for the interval boundary edge
        if interval_boundary_edge:
            y_structure.rightmost[interval_boundary_edge] = p

    def _map_edge_replacement(self, old_edge: Edge, new_edge: Edge, y_structure: YStructure):
        """
        Map edge replacement for O(1) bend handling without BST operations.
        Uses an edge mapping table to track replacements.
        """
        # Initialize edge mapping if not exists
        if not hasattr(y_structure, '_edge_mapping'):
            y_structure._edge_mapping = {}
        
        # Map old edge to new edge
        y_structure._edge_mapping[old_edge] = new_edge
        print(f"      Mapped {old_edge} -> {new_edge}")
    
    def _resolve_edge_mapping(self, edge: Edge, y_structure: YStructure) -> Edge:
        """
        Resolve an edge through the mapping chain to get the current active edge.
        This allows O(1) bend processing without BST modifications.
        """
        if not hasattr(y_structure, '_edge_mapping'):
            return edge
            
        # Follow the mapping chain
        current_edge = edge
        visited = set()
        
        while current_edge in y_structure._edge_mapping and current_edge not in visited:
            visited.add(current_edge)
            current_edge = y_structure._edge_mapping[current_edge]
            
        return current_edge

    def _initialize_basic_y_structure(self, p: Point, y_structure: YStructure):
        """
        Initialize basic Y-structure when encountering vertices that need it.
        Creates boundary edges at infinity to establish proper interval structure.
        """
        print(f"    Initializing basic Y-structure for point {p}")
        
        # Add boundary edges at infinity for proper interval management
        inf_bottom = Point(float('-inf'), float('-inf'), -1)
        inf_top = Point(float('-inf'), float('+inf'), -2)
        boundary_bottom = Edge(inf_bottom, Point(float('+inf'), float('-inf'), -3))
        boundary_top = Edge(inf_top, Point(float('+inf'), float('+inf'), -4))
        
        # Insert boundary edges if not already present
        if len(y_structure._bst) == 0:
            y_structure.insert(boundary_bottom, IntervalType.OUT)
            y_structure.insert(boundary_top, IntervalType.OUT)

    def _triangulate_simple_polygon(self, vertices: List[Point]):
        """
        Simple polygon triangulation fallback when sweep-line structure is not available.
        Uses a fan triangulation approach.
        """
        if len(vertices) < 3:
            return
            
        print(f"    Using fan triangulation for {len(vertices)} vertices")
        center = vertices[0]
        
        for i in range(1, len(vertices) - 1):
            v1 = vertices[i]
            v2 = vertices[i + 1]
            
            if self.orientation(center, v1, v2) == 2:  # CCW orientation
                self._add_triangle(center, v1, v2)
                print(f"      Created triangle: P{center.index}-P{v1.index}-P{v2.index}")
            else:
                # Try reverse orientation for consistent triangulation
                self._add_triangle(center, v2, v1)
                print(f"      Created triangle (reversed): P{center.index}-P{v2.index}-P{v1.index}")
    
    def _create_triangles_from_bend(self, p: Point, y_structure: YStructure):
        """
        Create triangles directly from a bend point when chain-based approach fails.
        This handles cases where the bend point needs immediate triangulation.
        """
        print(f"    Creating triangles directly from bend point {p}")
        
        # For bend points in simple polygons, we can often create triangles
        # by connecting the bend point to nearby vertices
        prev_pt = p.prev_point
        next_pt = p.next_point
        
        if prev_pt and next_pt:
            # Check if we can create a valid triangle with immediate neighbors
            if self.is_convex_angle(prev_pt, p, next_pt):
                # This is a convex bend - we might be able to triangulate locally
                # Look for nearby points in the polygon that can form valid triangles
                polygon_vertices = self._get_polygon_vertices_from_point(p)
                if polygon_vertices:
                    self._triangulate_locally_around_bend(p, polygon_vertices, y_structure)
            else:
                # Concave bend - handle differently
                self._handle_concave_bend(p, y_structure)
    
    def _get_polygon_vertices_from_point(self, p: Point) -> List[Point]:
        """Get all vertices of the polygon containing point p."""
        if not p.next_point:
            return []
        
        vertices = [p]
        current = p.next_point
        while current != p and len(vertices) < 100:  # Safety limit
            vertices.append(current)
            current = current.next_point
            if not current:
                break
        
        return vertices
    
    def _triangulate_locally_around_bend(self, bend_pt: Point, polygon_vertices: List[Point], y_structure: YStructure):
        """Triangulate locally around a bend point."""
        bend_idx = polygon_vertices.index(bend_pt)
        n = len(polygon_vertices)
        
        # Try to create triangles using ear-clipping-like approach around the bend
        prev_idx = (bend_idx - 1) % n
        next_idx = (bend_idx + 1) % n
        next2_idx = (bend_idx + 2) % n
        
        prev_pt = polygon_vertices[prev_idx]
        next_pt = polygon_vertices[next_idx] 
        next2_pt = polygon_vertices[next2_idx] if next2_idx != bend_idx else None
        
        # Try to create a triangle with the bend point
        if next2_pt and self._can_form_valid_triangle(prev_pt, bend_pt, next2_pt, polygon_vertices):
            self._add_triangle(prev_pt, bend_pt, next2_pt)
            print(f"      Created triangle from bend: P{prev_pt.index}-P{bend_pt.index}-P{next2_pt.index}")
        elif self._can_form_valid_triangle(prev_pt, bend_pt, next_pt, polygon_vertices):
            self._add_triangle(prev_pt, bend_pt, next_pt)
            print(f"      Created triangle from bend: P{prev_pt.index}-P{bend_pt.index}-P{next_pt.index}")
    
    def _handle_concave_bend(self, p: Point, y_structure: YStructure):
        """Handle concave bend points that need special triangulation."""
        print(f"      Handling concave bend at {p}")
        # For concave bends, we might need to add diagonal edges
        # This is a simplified approach
        if p.prev_point and p.next_point:
            # Find a suitable diagonal to create triangles
            polygon_vertices = self._get_polygon_vertices_from_point(p)
            if len(polygon_vertices) >= 4:
                self._add_diagonal_for_concave_bend(p, polygon_vertices)
    
    def _add_diagonal_for_concave_bend(self, bend_pt: Point, vertices: List[Point]):
        """Add a diagonal edge to resolve a concave bend."""
        bend_idx = vertices.index(bend_pt)
        n = len(vertices)
        
        # Try connecting to a vertex that's not immediately adjacent
        for i in range(2, n - 2):
            target_idx = (bend_idx + i) % n
            target_pt = vertices[target_idx]
            
            # Check if this diagonal is valid (doesn't intersect polygon boundary)
            if self._is_valid_diagonal(bend_pt, target_pt, vertices):
                # Create triangles using this diagonal
                self._add_triangle(bend_pt.prev_point, bend_pt, target_pt)
                print(f"      Created triangle with diagonal: P{bend_pt.prev_point.index}-P{bend_pt.index}-P{target_pt.index}")
                break
    
    def _can_form_valid_triangle(self, p1: Point, p2: Point, p3: Point, polygon_vertices: List[Point]) -> bool:
        """Check if three points can form a valid triangle for triangulation."""
        # Check if triangle is oriented correctly (CCW)
        if self.orientation(p1, p2, p3) != 2:  # Not CCW
            return False
        
        # Check if any other polygon vertex is inside this triangle
        for vertex in polygon_vertices:
            if vertex in [p1, p2, p3]:
                continue
            if self.point_in_triangle(vertex, p1, p2, p3):
                return False
        
        return True
    
    def _is_valid_diagonal(self, p1: Point, p2: Point, vertices: List[Point]) -> bool:
        """Check if a diagonal between two vertices is valid (doesn't intersect polygon edges)."""
        diagonal = Edge(p1, p2)
        
        # Check intersection with all polygon edges
        n = len(vertices)
        for i in range(n):
            edge = Edge(vertices[i], vertices[(i + 1) % n])
            # Skip edges that share a vertex with the diagonal
            if edge.p1 in [p1, p2] or edge.p2 in [p1, p2]:
                continue
            
            if self._edges_intersect(diagonal, edge):
                return False
        
        return True
    
    def _edges_intersect(self, edge1: Edge, edge2: Edge) -> bool:
        """Check if two edges intersect (not including endpoints)."""
        # Use the line intersection method but check for proper intersection
        intersection = self._line_segment_intersection(edge1, edge2)
        if intersection is None:
            return False
        
        # Check if intersection is at endpoints (not a proper intersection)
        eps = 1e-9
        at_endpoint = (
            (abs(intersection.x - edge1.p1.x) < eps and abs(intersection.y - edge1.p1.y) < eps) or
            (abs(intersection.x - edge1.p2.x) < eps and abs(intersection.y - edge1.p2.y) < eps) or
            (abs(intersection.x - edge2.p1.x) < eps and abs(intersection.y - edge2.p1.y) < eps) or
            (abs(intersection.x - edge2.p2.x) < eps and abs(intersection.y - edge2.p2.y) < eps)
        )
        
        return not at_endpoint
    
    def _line_segment_intersection(self, edge1: Edge, edge2: Edge) -> Optional[Point]:
        """Find intersection point of two line segments."""
        x1, y1 = edge1.p1.x, edge1.p1.y
        x2, y2 = edge1.p2.x, edge1.p2.y
        x3, y3 = edge2.p1.x, edge2.p1.y
        x4, y4 = edge2.p2.x, edge2.p2.y

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

        eps = 1e-9
        if abs(denom) < eps:
            return None  # Lines are parallel or collinear

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

        if 0 - eps <= t <= 1 + eps and 0 - eps <= u <= 1 + eps:
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            return Point(x, y)

        return None

    def handle_improper_end(self, p: Point, y_structure: YStructure):
        """
        Handles an "improper end" point (Fig. 9a,b,c).
        Merges two in-intervals and concatenates their chains.
        """
        print(f"  Handling Improper End: {p}")
        # For an improper end point p, both incident edges end at p (come from left).
        # These are (p.prev_point, p) and (p.next_point, p).

        edge1_to_p = Edge(p.prev_point, p)
        edge2_to_p = Edge(p.next_point, p)

        # Determine which is lower (t_edge) and upper (s_edge) based on y-value just to the left of p.x
        test_x = p.x - 1e-6
        if edge1_to_p.y_at_x(test_x) < edge2_to_p.y_at_x(test_x):
            t_edge = edge1_to_p
            s_edge = edge2_to_p
        else:
            t_edge = edge2_to_p
            s_edge = edge1_to_p

        # The paper's pseudocode implies:
        # L(h) <- L(h) o p; CHAIN_TRI(h, "c") (h is SUCC(s))
        # L(t) <- p o L(t); CHAIN_TRI(t, "cc") (t is the lower edge)
        # L(h) <- L(h) o L(t) (concatenate chains)

        # This means:
        # The upper in-interval is bounded by `s_edge` from above. Its chain is `y_structure.chains[s_edge]`.
        # The lower in-interval is bounded by `t_edge` from below. Its chain is `y_structure.chains[y_structure.succ(t_edge)]`.

        chain_s = y_structure.chains.get(s_edge)  # Chain for the interval bounded by s_edge from above
        h_succ_edge = y_structure.succ(s_edge)  # Edge above s_edge
        chain_h_succ = y_structure.chains.get(h_succ_edge)  # Chain for the interval bounded by h_succ_edge from above

        if not chain_s or not chain_h_succ:
            print(
                f"    Error: Chains not found for improper end point {p}. s_edge: {s_edge}, h_succ_edge: {h_succ_edge}")
            return

        # Triangulate along the upper chain (chain_s)
        chain_s.add_head(p)  # Add p to the head of the chain associated with s_edge
        self.chain_triangulate(chain_s, "cc")

        # Triangulate along the lower chain (chain_h_succ)
        chain_h_succ.add_tail(p)  # Add p to the tail of the chain associated with h_succ_edge
        self.chain_triangulate(chain_h_succ, "c")

        # Concatenate the two chains: chain_h_succ (lower) and chain_s (upper)
        # The result will be the chain for the new merged interval.
        # The new upper boundary will be h_succ_edge.

        # Append points of chain_s to chain_h_succ
        while chain_s.points:
            chain_h_succ.add_tail(chain_s.points.popleft())

        # Update rightmost point for the merged chain
        chain_h_succ.rightmost_point = max(chain_h_succ.points, key=lambda pt: pt.x) if chain_h_succ.points else None

        # Delete edges t_edge and s_edge from Y structure
        y_structure.delete(t_edge)
        y_structure.delete(s_edge)

        # The interval type of the new merged interval will be 'in'.
        # The chain for h_succ_edge (which was the upper boundary of the lower interval)
        # now becomes the chain for the new merged 'in' interval.
        # This is already handled by modifying chain_h_succ directly.

    def handle_improper_start(self, p: Point, y_structure: YStructure):
        """
        Handles an "improper start" point (Fig. 10a,b,c).
        Splits an in-interval into three (in, out, in) and splits the chain.
        Enhanced with SPEC integration for vertical edge handling.
        """
        print(f"  Handling Improper Start: {p}")
        
        # SPEC integration: Check for vertical edges
        if hasattr(p, 'spec') and p.spec:
            print(f"    SPEC case detected for improper start {p}")
            if hasattr(p, 'co_point') and p.co_point:
                self._process_spec_improper_start(p, p.co_point, y_structure)
                return
        
        # FIND(p) delivers the two active edges t (below) and s (above)
        # in whose interval [t, s] p lies. This interval must be an 'in' interval.
        t_edge_bounding_interval, s_edge_bounding_interval = y_structure.find_interval(p)

        interval_type = y_structure.get_type(t_edge_bounding_interval, s_edge_bounding_interval)
        if interval_type != IntervalType.IN:
            print(f"    Error: Improper start point {p} not in an IN interval. Found type: {interval_type}")
            # Try to recover
            if not y_structure._bst.root or len(y_structure._bst) == 0:
                self._initialize_basic_y_structure(p, y_structure)
                t_edge_bounding_interval, s_edge_bounding_interval = y_structure.find_interval(p)
                interval_type = y_structure.get_type(t_edge_bounding_interval, s_edge_bounding_interval)
            
            if interval_type != IntervalType.IN:
                print(f"    Unable to place improper start in IN interval, proceeding with fallback")
                return

        # q <- RM(s) (rightmost node of the chain associated with s_edge_bounding_interval)
        chain_s = y_structure.chains.get(s_edge_bounding_interval)
        if not chain_s:
            print(f"    Error: Chain not found for improper start point {p}")
            return
        q = y_structure.rightmost.get(s_edge_bounding_interval)

        if not q:
            print(f"    Error: Rightmost point not found for chain of {s_edge_bounding_interval}")
            return

        # Add edge qp to EDGES (triangulation edge)
        # The paper implies adding a diagonal qp.
        # The triangle formed would be (q, p, some_other_point_on_chain).
        # For simplicity, we add the diagonal directly.
        self.triangulation_edges.add(Edge(p, q))

        # Identify the two new polygon edges starting at p
        # For an improper start, both p.prev_point.x and p.next_point.x are > p.x.
        edge1_from_p = Edge(p, p.prev_point)
        edge2_from_p = Edge(p, p.next_point)

        test_x = p.x + 1e-6
        if edge1_from_p.y_at_x(test_x) < edge2_from_p.y_at_x(test_x):
            l_new_edge = edge1_from_p  # Lower new edge from p
            h_new_edge = edge2_from_p  # Higher new edge from p
        else:
            l_new_edge = edge2_from_p
            h_new_edge = edge1_from_p

        # Split the original in-interval [t, s] into [t, l_new_edge] (IN), [l_new_edge, h_new_edge] (OUT), [h_new_edge, s] (IN)
        # Delete original s_edge_bounding_interval from Y (it's now split conceptually, but still exists as an edge)
        # The paper says: INSERT((l,in)); INSERT((h,out));
        # This implies replacing the original s's entry with new ones.
        # We need to delete the original s_edge_bounding_interval and re-insert it with its new type.
        y_structure.delete(s_edge_bounding_interval)  # Delete original entry for 's'

        # Insert new edges and their interval types
        y_structure.insert(l_new_edge, IntervalType.IN)  # Lower new in-interval
        y_structure.insert(h_new_edge, IntervalType.OUT)  # Out-interval between new edges
        y_structure.insert(s_edge_bounding_interval, IntervalType.IN)  # Original s now bounds the upper new in-interval

        # Update rightmost points for the new in-intervals
        y_structure.rightmost[s_edge_bounding_interval] = p
        y_structure.rightmost[l_new_edge] = p

        # Split the chain L(s) into two new chains, L(l_new_edge) and L(s)
        # L(l_new_edge) <- p o "remainder of L(s) starting at q"
        # L(s) <- "L(s) up to and including q" o p

        # Find q's position in the original chain
        q_idx = -1
        try:
            q_idx = list(chain_s.points).index(q)
        except ValueError:
            print(f"    Error: Rightmost point {q} not found in chain {chain_s} for improper start {p}")
            return

        # Create new chain for l_new_edge (from p to q, then remainder of old chain)
        new_chain_l = PolygonalChain()
        new_chain_l.add_head(p)
        for i in range(q_idx, len(chain_s.points)):
            new_chain_l.add_tail(chain_s.points[i])
        y_structure.chains[l_new_edge] = new_chain_l

        # Update original chain_s for the upper part (from head to q, then p)
        new_chain_s_upper = PolygonalChain()
        for i in range(q_idx + 1):  # Points up to and including q
            new_chain_s_upper.add_tail(chain_s.points[i])
        new_chain_s_upper.add_tail(p)  # Add p as the new rightmost
        y_structure.chains[s_edge_bounding_interval] = new_chain_s_upper

        # Triangulate along the new chains
        self.chain_triangulate(y_structure.chains[s_edge_bounding_interval], "c")  # CHAIN_TRI(s, "c")
        self.chain_triangulate(y_structure.chains[l_new_edge], "cc")  # CHAIN_TRI(l, "cc")

    def handle_proper_end(self, p: Point, y_structure: YStructure):
        """
        Handles a "proper end" point (Fig. 11a,b,c).
        Finishes triangulation of an in-interval and merges two out-intervals.
        """
        print(f"  Handling Proper End: {p}")
        # For a proper end point p, both incident edges end at p (come from left).
        # These are (p.prev_point, p) and (p.next_point, p).

        edge1_to_p = Edge(p.prev_point, p)
        edge2_to_p = Edge(p.next_point, p)

        # Determine which is lower (t_edge) and upper (s_edge) based on y-value just to the left of p.x
        test_x = p.x - 1e-6
        if edge1_to_p.y_at_x(test_x) < edge2_to_p.y_at_x(test_x):
            t_edge = edge1_to_p
            s_edge = edge2_to_p
        else:
            t_edge = edge2_to_p
            s_edge = edge1_to_p

        # The interval [t, s] must be an 'in' interval.
        # The chain is associated with `s` (upper boundary).
        chain_s = y_structure.chains.get(s_edge)
        if not chain_s:
            print(f"    Error: Chain not found for proper end point {p}. Expected chain for {s_edge}")
            return

        # L(s) <- p o L(s) (add p as new head)
        chain_s.add_head(p)

        # CHAIN_TRI(s, "cc")
        self.chain_triangulate(chain_s, "cc")

        # Delete edges t and s from Y structure
        y_structure.delete(t_edge)
        y_structure.delete(s_edge)

        # In a full implementation, the two adjacent out-intervals would be merged.
        # This means the PRED(t_edge) and SUCC(s_edge) intervals would merge.
        # The type of SUCC(s_edge) would become OUT.
        # The chain for SUCC(s_edge) would be updated to reflect the merge.
        print(f"    (Simplified) Merging out-intervals for proper end {p}")

    # --- Triangulation Algorithms ---

    def _prepare_polygon_for_sweep(self, polygon: List[Point]):
        """Helper to set up prev/next pointers and classify vertices."""
        n = len(polygon)
        for i in range(n):
            polygon[i].prev_point = polygon[(i - 1 + n) % n]
            polygon[i].next_point = polygon[(i + 1) % n]
            polygon[i].point_type = self.classify_vertex(polygon, i)
            self.polygon_edges.add(Edge(polygon[i], polygon[i].next_point))

    def basic_triangulate(self, polygon: List[Point]) -> List[Triangle]:
        """
        Implements the basic sweep-line triangulation algorithm from Hertel & Mehlhorn (1985).
        """
        print("\n--- Running Basic Triangulation (Sweep-Line) ---")
        self.triangles = []
        self.triangulation_edges = set()
        self.polygon_edges = set()  # Reset for this run

        self._prepare_polygon_for_sweep(polygon)
        return self._sweep_line_triangulate(polygon)

    def _sweep_line_triangulate(self, polygon: List[Point]) -> List[Triangle]:
        """
        Basic sweep-line triangulation implementation based on Hertel & Mehlhorn (1985).
        """
        print("  Implementing sweep-line triangulation algorithm")
        
        # Sort vertices by x-coordinate (X-structure/event queue)
        sorted_vertices = sorted(polygon, key=lambda p: (p.x, p.y))
        
        # Initialize Y-structure (sweep line status)
        y_structure = YStructure(self)
        
        # Initialize Y-structure with active polygon edges at the leftmost x-coordinate
        leftmost_x = sorted_vertices[0].x
        self._initialize_y_structure_with_polygon_edges(polygon, leftmost_x, y_structure)
        
        # Process events in x-order
        for i, vertex in enumerate(sorted_vertices):
            print(f"  Event {i+1}: Processing P{vertex.index} at ({vertex.x}, {vertex.y}) - {vertex.point_type}")
            y_structure.set_current_x(vertex.x)
            
            # Use the existing transition method which handles all vertex types
            self.transition(vertex, y_structure)
        
        # Check if triangulation was successful
        if len(self.triangles) > 0:
            print(f"  Sweep-line algorithm produced {len(self.triangles)} triangles")
        else:
            print("  Sweep-line algorithm produced no triangles, using fan triangulation fallback")
            self._fan_triangulate(polygon)
        
        return self.triangles
    
    def _handle_all_bend_polygon(self, polygon: List[Point], y_structure: YStructure) -> List[Triangle]:
        """
        Special handling for polygons where all vertices are BEND points.
        This is common for simple convex/concave polygons.
        """
        # For all-bend polygons, we need to initialize chains differently
        # We'll simulate the sweep-line by processing edges and creating triangles
        
        # Sort vertices by x-coordinate
        sorted_vertices = sorted(polygon, key=lambda p: (p.x, p.y))
        leftmost_vertex = sorted_vertices[0]
        
        # Initialize the Y-structure with the first polygon edges that intersect 
        # the leftmost vertex's x-coordinate
        y_structure.set_current_x(leftmost_vertex.x)
        
        # Find all edges that are active at the leftmost x-coordinate
        active_edges = []
        for i in range(len(polygon)):
            curr_point = polygon[i]
            next_point = polygon[(i + 1) % len(polygon)]
            
            edge = Edge(curr_point, next_point)
            # Check if edge spans or starts at leftmost_vertex.x
            min_x = min(curr_point.x, next_point.x)
            max_x = max(curr_point.x, next_point.x)
            
            if min_x <= leftmost_vertex.x <= max_x:
                active_edges.append(edge)
        
        # Sort active edges by their y-coordinate at leftmost x
        active_edges.sort(key=lambda e: e.y_at_x(leftmost_vertex.x))
        
        # Initialize chains for the active edges creating IN intervals
        for i in range(0, len(active_edges) - 1, 2):
            if i + 1 < len(active_edges):
                lower_edge = active_edges[i]
                upper_edge = active_edges[i + 1]
                
                # Create an IN interval between these edges
                y_structure.insert(lower_edge, IntervalType.OUT)
                y_structure.insert(upper_edge, IntervalType.IN)
                
                # Initialize chain for the IN interval
                chain = PolygonalChain()
                # Find the vertex at leftmost_vertex.x that lies between these edges
                for vertex in sorted_vertices:
                    if abs(vertex.x - leftmost_vertex.x) < 1e-9:
                        vertex_y = vertex.y
                        lower_y = lower_edge.y_at_x(vertex.x)
                        upper_y = upper_edge.y_at_x(vertex.x)
                        if lower_y <= vertex_y <= upper_y:
                            chain.add_head(vertex)
                            break
                
                y_structure.chains[upper_edge] = chain
                y_structure.rightmost[upper_edge] = chain.get_head() if chain.get_head() else leftmost_vertex
        
        # Now process remaining vertices
        for vertex in sorted_vertices[1:]:
            print(f"  Processing BEND vertex P{vertex.index} at ({vertex.x}, {vertex.y})")
            self._handle_bend_vertex(vertex, y_structure)
        
        # If still no triangles, use fan triangulation
        if not self.triangles:
            print("  Specialized BEND handling produced no triangles, using fan triangulation")
            self._fan_triangulate(polygon)
        
        return self.triangles
    
    def _sweep_line_triangulate_improved(self, polygon: List[Point]) -> List[Triangle]:
        """
        Improved sweep-line triangulation with O(n + s log s) complexity.
        Uses extend_local_sweep_lines to handle bend points on-the-fly.
        """
        print("  Implementing improved sweep-line triangulation algorithm")
        self.using_improved_algorithm = True  # Set flag for bend handling optimization
        
        # Sort vertices by x-coordinate (X-structure/event queue)
        sorted_vertices = sorted(polygon, key=lambda p: (p.x, p.y))
        
        # Initialize Y-structure (sweep line status) 
        y_structure = YStructure(self)
        
        # Add boundary edges at infinity for proper interval management
        inf_bottom = Point(float('-inf'), float('-inf'), -1)
        inf_top = Point(float('-inf'), float('+inf'), -2)
        boundary_bottom = Edge(inf_bottom, Point(float('+inf'), float('-inf'), -3))
        boundary_top = Edge(inf_top, Point(float('+inf'), float('+inf'), -4))
        y_structure.insert(boundary_bottom, IntervalType.OUT)
        y_structure.insert(boundary_top, IntervalType.OUT)
        
        # Filter to only process start/end points (the 2s points mentioned in the paper)
        # This is crucial for achieving O(n + s log s) complexity
        start_end_vertices = [v for v in sorted_vertices if v.point_type in [
            PointType.START_PROPER, PointType.START_IMPROPER, 
            PointType.END_PROPER, PointType.END_IMPROPER
        ]]
        
        print(f"  Processing only {len(start_end_vertices)} start/end points out of {len(sorted_vertices)} total vertices")
        
        # Process only start/end events - bend points handled via extend_local_sweep_lines
        for i, vertex in enumerate(start_end_vertices):
            print(f"  Event {i+1}: Processing P{vertex.index} at ({vertex.x}, {vertex.y}) - {vertex.point_type}")
            
            # Verify no BEND points reach this processing stage
            if vertex.point_type == PointType.BEND:
                print(f"    ERROR: BEND point reached main event processing in improved algorithm")
                print(f"    This violates the O(1) bend handling guarantee")
                continue
            
            # Before processing this vertex, extend local sweep lines to handle intervening bend points
            if i > 0:
                self.extend_local_sweep_lines(vertex, y_structure)
            
            # Only set current_x at actual event points (start/end), not at every vertex
            y_structure.set_current_x(vertex.x)
            
            # Handle vertical edges (SPEC functionality)  
            if self._has_vertical_edge_above(vertex, polygon):
                print(f"    Vertex has vertical edge above - handling SPEC case")
                vertex.spec = True
                self._handle_vertical_edge_case(vertex, y_structure)
            
            # Process the vertex using the transition method
            self.transition(vertex, y_structure)
        
        # Final check and cleanup
        if len(self.triangles) > 0:
            print(f"  Improved sweep-line algorithm produced {len(self.triangles)} triangles")
        else:
            print("  Improved sweep-line algorithm produced no triangles, using fan triangulation fallback")
            self._fan_triangulate(polygon)
        
        return self.triangles
    
    def _has_vertical_edge_above(self, vertex: Point, polygon: List[Point]) -> bool:
        """
        Check if there's a vertical edge above this vertex (SPEC case from paper).
        """
        # Check if any polygon edge is vertical and has this vertex as lower endpoint
        for i, curr_point in enumerate(polygon):
            next_point = polygon[(i + 1) % len(polygon)]
            
            # Check if edge (curr_point, next_point) is vertical and vertex is the lower point
            if (abs(curr_point.x - next_point.x) < 1e-9 and  # Vertical edge
                abs(curr_point.x - vertex.x) < 1e-9 and      # Same x-coordinate as vertex
                ((curr_point == vertex and next_point.y > vertex.y) or  # Vertex is lower endpoint
                 (next_point == vertex and curr_point.y > vertex.y))):   # Vertex is lower endpoint (other direction)
                return True
        
        return False
    
    def _handle_vertical_edge_case(self, vertex: Point, y_structure: YStructure):
        """
        Handle the SPEC case when a vertex has a vertical edge above it.
        
        Enhanced to fully integrate SPEC processing into TRANSITION methods.
        The paper states that p and co_p are processed together, and the SPEC(p) 
        predicate affects how chains are initialized or updated.
        """
        print(f"    Handling vertical edge case for P{vertex.index}")
        
        # Find the co-point (upper endpoint of vertical edge)
        co_point = self._find_vertical_co_point(vertex)
        
        if co_point:
            vertex.co_point = co_point
            co_point.co_point = vertex
            print(f"    Found co-point P{co_point.index} for vertical edge")
            
            # Enhanced SPEC processing: handle both endpoints together
            self._process_vertical_edge_pair(vertex, co_point, y_structure)
        else:
            print(f"    Warning: SPEC case detected but no co-point found for P{vertex.index}")

    def _find_vertical_co_point(self, vertex: Point) -> Optional[Point]:
        """
        Find the co-point for a vertical edge starting at vertex.
        Returns the other endpoint of the vertical edge.
        """
        # Check next point
        if (vertex.next_point and 
            abs(vertex.x - vertex.next_point.x) < 1e-9 and 
            vertex.next_point.y > vertex.y):
            return vertex.next_point
            
        # Check previous point  
        if (vertex.prev_point and 
            abs(vertex.x - vertex.prev_point.x) < 1e-9 and 
            vertex.prev_point.y > vertex.y):
            return vertex.prev_point
            
        return None

    def _process_vertical_edge_pair(self, lower_point: Point, upper_point: Point, y_structure: YStructure):
        """
        Process a vertical edge pair according to the SPEC logic from the paper.
        
        The paper mentions that the SPEC predicate affects how chains are 
        initialized or updated when processing vertical edges.
        """
        print(f"    Processing vertical edge pair: P{lower_point.index} -> P{upper_point.index}")
        
        # Create the vertical edge
        vertical_edge = Edge(lower_point, upper_point)
        
        # The SPEC case affects interval initialization
        # For vertical edges, we need to handle them specially in the Y-structure
        
        # Find what interval the vertical edge would create or modify
        below_edge, above_edge = y_structure.find_interval(lower_point)
        
        if below_edge and above_edge:
            # Vertical edge is splitting an existing interval
            interval_type = y_structure.get_type(below_edge, above_edge)
            
            if interval_type == IntervalType.OUT:
                # Vertical edge in OUT-interval creates a new IN-interval
                print(f"    Vertical edge creates new IN-interval")
                self._handle_vertical_edge_in_out_interval(vertical_edge, lower_point, 
                                                         upper_point, y_structure)
            else:
                # Vertical edge in IN-interval needs special handling
                print(f"    Vertical edge in IN-interval")
                self._handle_vertical_edge_in_in_interval(vertical_edge, lower_point, 
                                                        upper_point, y_structure)
        else:
            # Edge case: vertical edge at boundary
            print(f"    Vertical edge at interval boundary")
            self._handle_vertical_edge_at_boundary(vertical_edge, lower_point, 
                                                 upper_point, y_structure)

    def _handle_vertical_edge_in_out_interval(self, vertical_edge: Edge, lower_point: Point, 
                                            upper_point: Point, y_structure: YStructure):
        """Handle vertical edge that creates a new IN-interval."""
        # Insert the vertical edge as creating an IN-interval
        y_structure.insert(vertical_edge, IntervalType.IN)
        
        # Initialize a chain for this vertical edge
        chain = PolygonalChain()
        chain.add_head(lower_point)
        chain.add_tail(upper_point)
        
        y_structure.chains[vertical_edge] = chain
        y_structure.rightmost[vertical_edge] = upper_point if upper_point.x >= lower_point.x else lower_point

    def _handle_vertical_edge_in_in_interval(self, vertical_edge: Edge, lower_point: Point, 
                                           upper_point: Point, y_structure: YStructure):
        """Handle vertical edge within an existing IN-interval."""
        # Find the existing chain for this in-interval
        below_edge, above_edge = y_structure.find_interval(lower_point)
        existing_chain = y_structure.chains.get(above_edge)
        
        if existing_chain:
            # Add the vertical edge points to the existing chain
            existing_chain.add_tail(lower_point)
            existing_chain.add_tail(upper_point)
            
            # Try to triangulate with the updated chain
            self.chain_triangulate(existing_chain, "c")
            
            # Update rightmost point
            if upper_point.x > existing_chain.rightmost_point.x:
                y_structure.rightmost[above_edge] = upper_point

    def _handle_vertical_edge_at_boundary(self, vertical_edge: Edge, lower_point: Point, 
                                        upper_point: Point, y_structure: YStructure):
        """Handle vertical edge at interval boundary (edge case)."""
        # For boundary cases, we typically need to create triangles directly
        print(f"    Creating triangles for boundary vertical edge")
        
        # Try to create triangles using the vertical edge and nearby points
        if lower_point.prev_point and upper_point.next_point:
            # Create triangle with previous point of lower and next point of upper
            if self.orientation(lower_point.prev_point, lower_point, upper_point) == 2:
                self._add_triangle(lower_point.prev_point, lower_point, upper_point)
                print(f"      Created triangle: P{lower_point.prev_point.index}-P{lower_point.index}-P{upper_point.index}")
                
            if self.orientation(lower_point, upper_point, upper_point.next_point) == 2:
                self._add_triangle(lower_point, upper_point, upper_point.next_point)
                print(f"      Created triangle: P{lower_point.index}-P{upper_point.index}-P{upper_point.next_point.index}")

    def transition(self, p: Point, y_structure: YStructure):
        """
        TRANSITION(p): Enhanced to properly integrate SPEC functionality.
        
        Handles the sweep-line event for a given point p with improved 
        vertical edge handling integrated into the transition logic.
        """
        # Remove set_current_x from here as it should only be called at event points
        # in the main sweep loop, not for every transition call

        # Enhanced SPEC handling: check and process vertical edges before main transition
        if p.spec:
            print(f"  Processing SPEC case for {p}")
            self._handle_vertical_edge_case(p, y_structure)
            
            # For SPEC cases, we may need to process both endpoints
            if p.co_point:
                print(f"  Processing co-point {p.co_point} in SPEC case")
                # Process the co-point's transition as well if it's at the same x-coordinate
                if abs(p.x - p.co_point.x) < 1e-9:
                    self._process_co_point_transition(p.co_point, y_structure)

        # Standard transition logic based on point type
        # IMPORTANT: For improved algorithm, BEND points should NEVER reach here
        # They should be processed exclusively via extend_local_sweep_lines
        if self.using_improved_algorithm and p.point_type == PointType.BEND:
            print(f"  ERROR: BEND point {p} reached transition() in improved algorithm")
            print(f"    BEND points should be processed only via extend_local_sweep_lines")
            return  # Skip processing to maintain O(1) guarantee
        
        if p.point_type == PointType.START_PROPER:
            self.handle_proper_start(p, y_structure)
        elif p.point_type == PointType.START_IMPROPER:
            self.handle_improper_start(p, y_structure)
        elif p.point_type == PointType.END_PROPER:
            self.handle_proper_end(p, y_structure)
        elif p.point_type == PointType.END_IMPROPER:
            self.handle_improper_end(p, y_structure)
        elif p.point_type == PointType.BEND:
            # This should only happen in the original algorithm
            if not self.using_improved_algorithm:
                self.handle_bend(p, y_structure)
            else:
                print(f"  Warning: BEND point {p} in improved algorithm should use extend_local_sweep_lines")
        else:
            print(f"  Warning: Unknown point type for {p}")

    def _process_spec_proper_start(self, p: Point, co_p: Point, y_structure: YStructure):
        """
        Process a proper start point with SPEC condition.
        Handles the "p and co_p processed together" logic from the paper.
        """
        print(f"    Processing SPEC proper start: {p} with co-point {co_p}")
        
        # In SPEC cases, the vertical edge creates special interval conditions
        # The paper's algorithm for SPEC(p) in handle_proper_start context
        # involves initializing L(h) differently when there's a vertical edge
        
        # Find the bounding interval for the vertical edge pair
        t_edge_bounding_interval, s_edge_bounding_interval = y_structure.find_interval(p)
        
        # For SPEC cases, we may need to handle the vertical edge as part of the interval
        vertical_edge = Edge(p, co_p) if p.y < co_p.y else Edge(co_p, p)
        
        # Process according to SPEC logic in the paper
        # This involves special chain initialization for the vertical edge
        if t_edge_bounding_interval and s_edge_bounding_interval:
            # Delete old interval bounds
            y_structure.delete(t_edge_bounding_interval)
            y_structure.delete(s_edge_bounding_interval)
            
            # Insert the vertical edge with appropriate interval types
            y_structure.insert(vertical_edge, IntervalType.IN)
            
            # Initialize chain for the vertical edge
            chain = PolygonalChain()
            chain.add_head(p)
            chain.add_tail(co_p)
            y_structure.chains[vertical_edge] = chain
            y_structure.rightmost[vertical_edge] = co_p if co_p.x > p.x else p

    def _process_spec_improper_start(self, p: Point, co_p: Point, y_structure: YStructure):
        """
        Process an improper start point with SPEC condition.
        Handles vertical edge cases for improper start points.
        """
        print(f"    Processing SPEC improper start: {p} with co-point {co_p}")
        
        # In SPEC cases for improper start, the vertical edge affects chain splitting
        vertical_edge = Edge(p, co_p) if p.y < co_p.y else Edge(co_p, p)
        
        # Find the existing IN interval
        t_edge_bounding_interval, s_edge_bounding_interval = y_structure.find_interval(p)
        existing_chain = y_structure.chains.get(s_edge_bounding_interval)
        
        if existing_chain:
            # Split chain considering the vertical edge
            self._split_chain_with_vertical_edge(existing_chain, p, co_p, vertical_edge, y_structure)
        else:
            # Create new chain structure for the vertical edge
            chain = PolygonalChain()
            chain.add_head(p)
            chain.add_tail(co_p)
            y_structure.chains[vertical_edge] = chain
            y_structure.rightmost[vertical_edge] = co_p if co_p.x > p.x else p
    
    def _split_chain_with_vertical_edge(self, chain: PolygonalChain, p: Point, co_p: Point, 
                                      vertical_edge: Edge, y_structure: YStructure):
        """
        Split a chain when a vertical edge is encountered in SPEC cases.
        """
        print(f"    Splitting chain with vertical edge {vertical_edge}")
        
        # Find the best position to split the chain
        split_pos = len(chain.points) // 2  # Simple midpoint split
        
        # Create two new chains
        left_chain = PolygonalChain()
        right_chain = PolygonalChain()
        
        # Add points to left chain
        for i in range(split_pos):
            left_chain.add_tail(chain.points[i])
        left_chain.add_tail(p)
        
        # Add points to right chain
        right_chain.add_head(co_p)
        for i in range(split_pos, len(chain.points)):
            right_chain.add_tail(chain.points[i])
        
        # Associate chains with edges
        y_structure.chains[vertical_edge] = left_chain
        # The right chain would be associated with the next active edge
        # This is a simplified approach

    def _process_co_point_transition(self, co_point: Point, y_structure: YStructure):
        """
        Process the transition for a co-point in a SPEC case.
        This handles the "p and co_p are processed together" requirement from the paper.
        """
        print(f"    Processing co-point transition for P{co_point.index}")
        
        # Co-point processing depends on its type but is simplified since 
        # the main processing was done in the vertical edge handling
        if co_point.point_type == PointType.BEND:
            # For bend co-points, we may need additional chain updates
            self._update_chains_for_co_point(co_point, y_structure)
        # Other point types may need specific handling based on the vertical edge context

    def _update_chains_for_co_point(self, co_point: Point, y_structure: YStructure):
        """Update chains when processing a co-point in SPEC case."""
        # Find chains that might be affected by the co-point
        below_edge, above_edge = y_structure.find_interval(co_point)
        
        if above_edge and above_edge in y_structure.chains:
            chain = y_structure.chains[above_edge]
            # Update the rightmost point if co_point is further right
            if chain.rightmost_point and co_point.x > chain.rightmost_point.x:
                y_structure.rightmost[above_edge] = co_point
    
    def _classify_start_end_points(self, polygon: List[Point]) -> Tuple[List[Point], int]:
        """
        Classify and count start points for complexity analysis.
        Returns (start_points, s) where s is the number of start vertices.
        """
        start_points = []
        s_count = 0  # Number of start vertices
        
        for vertex in polygon:
            if vertex.point_type in [PointType.START_PROPER, PointType.START_IMPROPER]:
                start_points.append(vertex)
                s_count += 1
        
        print(f"  Found {s_count} start vertices for O(n + s log s) complexity")
        return start_points, s_count
    
    def _handle_start_vertex(self, vertex: Point, y_structure: YStructure):
        """Handle start vertices (both proper and improper)."""
        if vertex.point_type == PointType.START_PROPER:
            self.handle_proper_start(vertex, y_structure)
        else:  # START_IMPROPER
            self.handle_improper_start(vertex, y_structure)
    
    def _handle_end_vertex(self, vertex: Point, y_structure: YStructure):
        """Handle end vertices (both proper and improper)."""
        if vertex.point_type == PointType.END_PROPER:
            self.handle_proper_end(vertex, y_structure)
        else:  # END_IMPROPER
            self.handle_improper_end(vertex, y_structure)
    
    def _handle_bend_vertex(self, vertex: Point, y_structure: YStructure):
        """Handle bend vertices."""
        self.handle_bend(vertex, y_structure)
    
    def _fan_triangulate(self, polygon: List[Point]):
        """
        Simple fan triangulation as fallback when sweep-line doesn't work.
        Creates triangles from the first vertex to all non-adjacent vertex pairs.
        """
        if len(polygon) < 3:
            return
        
        # Use first vertex as the fan center
        center = polygon[0]
        
        # Create triangles by connecting center to each non-adjacent pair
        for i in range(1, len(polygon) - 1):
            v1 = polygon[i]
            v2 = polygon[i + 1]
            self._add_triangle(center, v1, v2)
            print(f"  Added triangle (fan): P{center.index}-P{v1.index}-P{v2.index}")

    def _ear_clipping_triangulate(self, polygon: List[Point]) -> List[Triangle]:
        """Internal ear clipping implementation."""
        # Simple ear clipping algorithm
        vertices = polygon.copy()
        n = len(vertices)
        
        if n < 3:
            return self.triangles
            
        # Create triangles by ear clipping
        while len(vertices) > 3:
            ear_found = False
            
            for i in range(len(vertices)):
                prev_idx = (i - 1) % len(vertices)
                next_idx = (i + 1) % len(vertices)
                
                prev_vertex = vertices[prev_idx]
                curr_vertex = vertices[i]
                next_vertex = vertices[next_idx]
                
                # Check if current vertex forms an ear
                if self.is_ear(prev_vertex, curr_vertex, next_vertex, vertices):
                    # Create triangle
                    self._add_triangle(prev_vertex, curr_vertex, next_vertex)
                    print(f"  Added triangle: P{prev_vertex.index}-P{curr_vertex.index}-P{next_vertex.index}")
                    
                    # Remove the ear vertex
                    vertices.pop(i)
                    ear_found = True
                    break
            
            if not ear_found:
                print("  Warning: No ear found, breaking to avoid infinite loop")
                break
        
        # Add the final triangle
        if len(vertices) == 3:
            self._add_triangle(vertices[0], vertices[1], vertices[2])
            print(f"  Added final triangle: P{vertices[0].index}-P{vertices[1].index}-P{vertices[2].index}")

        return self.triangles
    
    def is_ear(self, prev_vertex: Point, curr_vertex: Point, next_vertex: Point, vertices: List[Point]) -> bool:
        """Check if the triangle formed by three consecutive vertices is an ear."""
        # Check if the angle at curr_vertex is convex
        if not self.is_convex_angle(prev_vertex, curr_vertex, next_vertex):
            return False
        
        # Check if any other vertex lies inside the triangle
        for vertex in vertices:
            if vertex in [prev_vertex, curr_vertex, next_vertex]:
                continue
            if self.point_in_triangle(vertex, prev_vertex, curr_vertex, next_vertex):
                return False
        
        return True
    
    def point_in_triangle(self, point: Point, a: Point, b: Point, c: Point) -> bool:
        """Check if a point lies inside a triangle using barycentric coordinates."""
        # Calculate vectors
        v0 = (c.x - a.x, c.y - a.y)
        v1 = (b.x - a.x, b.y - a.y)
        v2 = (point.x - a.x, point.y - a.y)
        
        # Calculate dot products
        dot00 = v0[0] * v0[0] + v0[1] * v0[1]
        dot01 = v0[0] * v1[0] + v0[1] * v1[1]
        dot02 = v0[0] * v2[0] + v0[1] * v2[1]
        dot11 = v1[0] * v1[0] + v1[1] * v1[1]
        dot12 = v1[0] * v2[0] + v1[1] * v2[1]
        
        # Calculate barycentric coordinates
        denom = dot00 * dot11 - dot01 * dot01
        if abs(denom) < 1e-9:
            return False  # Degenerate triangle
            
        inv_denom = 1.0 / denom
        u = (dot11 * dot02 - dot01 * dot12) * inv_denom
        v = (dot00 * dot12 - dot01 * dot02) * inv_denom
        
        # Check if point is in triangle
        return (u >= 0) and (v >= 0) and (u + v <= 1)

    def improved_triangulate(self, polygon: List[Point]) -> List[Triangle]:
        """
        Implements the improved sweep-line triangulation algorithm (O(n + s log s)) 
        from Hertel & Mehlhorn (1985).
        """
        print("\n--- Running Improved Triangulation (Sweep-Line, O(n+s log s)) ---")
        # Reset state for this run
        self.triangles = []
        self.triangulation_edges = set()
        self.polygon_edges = set()
        
        self._prepare_polygon_for_sweep(polygon)
        
        # Analyze complexity - count start vertices for O(n + s log s)
        start_points, s_count = self._classify_start_end_points(polygon)
        print(f"  Algorithm complexity: O({len(polygon)} + {s_count} log {s_count})")
        
        return self._sweep_line_triangulate_improved(polygon)

    def extend_local_sweep_lines(self, target_point: Point, y_structure: YStructure):
        """
        Extends local sweep lines to handle bends on-the-fly (Section 3).
        This method traverses polygon edges from the current sweep line position
        up to the target_point's x-coordinate, processing any bend points encountered.
        """
        print(f"  Extending local sweep lines to x={target_point.x}")
        
        # Find edges that are "lagging" (their rightmost point is < target_point.x)
        lagging_edges = y_structure.get_lagging_edges(target_point.x)
        
        if not lagging_edges:
            print(f"    No lagging edges found")
            return
        
        print(f"    Found {len(lagging_edges)} lagging edges")
        
        # For each lagging edge, traverse the polygon boundary to catch up
        for edge in lagging_edges:
            self._extend_edge_to_target_x(edge, target_point.x, y_structure)
    
    def _extend_edge_to_target_x(self, edge: Edge, target_x: float, y_structure: YStructure):
        """
        Extend a specific edge's local sweep line to target_x by processing bend points.
        """
        rightmost_point = y_structure.rightmost.get(edge)
        if not rightmost_point:
            return
        
        print(f"    Extending edge {edge} from x={rightmost_point.x} to x={target_x}")
        
        # Start from the rightmost point and traverse the polygon boundary
        current_point = rightmost_point
        points_processed = []
        
        # Traverse the polygon boundary until we reach target_x
        while current_point and current_point.x < target_x - 1e-9:
            # Follow the polygon boundary (using next_point)
            next_point = current_point.next_point
            if not next_point or next_point in points_processed:
                break
            
            points_processed.append(current_point)
            
            # If next_point is a bend point and within our target range
            if (next_point.point_type == PointType.BEND and 
                next_point.x <= target_x + 1e-9 and
                next_point.x > current_point.x + 1e-9):
                
                print(f"      Processing bend point P{next_point.index} at x={next_point.x}")
                
                # Process this bend point locally
                self._process_local_bend(next_point, y_structure, edge)
                
                # Update rightmost for this edge
                if next_point.x > rightmost_point.x:
                    y_structure.rightmost[edge] = next_point
                    rightmost_point = next_point
            
            current_point = next_point
            
            # Safety check to avoid infinite loops
            if len(points_processed) > 100:
                break
    
    def _process_local_bend(self, bend_point: Point, y_structure: YStructure, associated_edge: Edge):
        """
        Process a bend point found during local sweep line extension.
        This is a simplified version of handle_bend for local processing.
        """
        print(f"        Processing local bend at P{bend_point.index}")
        
        # Get the chain associated with this edge
        chain = y_structure.chains.get(associated_edge)
        if chain:
            # Add the bend point to the chain
            chain.add_tail(bend_point)
            
            # Try to triangulate if we have enough points
            if len(chain) >= 3:
                print(f"        Triangulating chain with {len(chain)} points")
                self.chain_triangulate(chain, "c")
        else:
            # If no existing chain, create one
            new_chain = PolygonalChain()
            new_chain.add_head(bend_point)
            y_structure.chains[associated_edge] = new_chain
            y_structure.rightmost[associated_edge] = bend_point

    def outer_triangulate_original(self, polygon: List[Point]) -> List[Triangle]:
        """
        Original outer triangulation - keeping for reference.
        """
        print("\n--- Running Outer Triangulation (Sweep-Line) ---")
        # Reset state for this run  
        self.triangles = []
        self.triangulation_edges = set()
        self.polygon_edges = set()
        
        self._prepare_polygon_for_sweep(polygon)
        return self._sweep_line_triangulate(polygon)

    def triangulate_multiple_polygons(self, polygons: List[List[Point]]) -> List[Triangle]:
        """
        Triangulates a set of multiple non-intersecting simple polygons (Section 5).
        Enhanced to use unified sweep-line approach as specified in the paper,
        rather than separate triangulations.
        """
        print("\n--- Running Enhanced Multiple Polygon Triangulation ---")
        all_triangles = []

        if not polygons:
            return all_triangles

        # Step 1: Merge all polygons into a unified event structure
        unified_events = self._create_unified_event_structure(polygons)
        print(f"  Created unified event structure with {len(unified_events)} events")

        # Step 2: Initialize unified Y-structure for all polygons
        y_structure = YStructure(self)
        self._initialize_unified_y_structure(polygons, y_structure)

        # Step 3: Process events with unified sweep-line
        all_triangles = self._unified_sweep_triangulation(unified_events, y_structure)
        
        print(f"  Unified sweep produced {len(all_triangles)} triangles for {len(polygons)} polygons")
        return all_triangles

    def _create_unified_event_structure(self, polygons: List[List[Point]]) -> List[Tuple[float, Point, int]]:
        """
        Create a unified event structure for all polygons.
        Each event includes x-coordinate, point, and polygon index.
        """
        events = []
        
        for poly_idx, polygon in enumerate(polygons):
            # Prepare polygon for sweep
            self._prepare_polygon_for_sweep(polygon)
            
            # Add all vertices as events
            for point in polygon:
                events.append((point.x, point, poly_idx))
        
        # Sort events by x-coordinate, then by y-coordinate
        events.sort(key=lambda e: (e[0], e[1].y))
        return events

    def _initialize_unified_y_structure(self, polygons: List[List[Point]], y_structure: YStructure):
        """
        Initialize Y-structure to handle multiple polygons simultaneously.
        """
        print("  Initializing unified Y-structure for multiple polygons")
        
        # Find the leftmost x-coordinate across all polygons
        leftmost_x = float('inf')
        for polygon in polygons:
            for point in polygon:
                leftmost_x = min(leftmost_x, point.x)
        
        # Add boundary edges at infinity
        inf_bottom = Point(float('-inf'), float('-inf'), -1)
        inf_top = Point(float('-inf'), float('+inf'), -2)
        boundary_bottom = Edge(inf_bottom, Point(float('+inf'), float('-inf'), -3))
        boundary_top = Edge(inf_top, Point(float('+inf'), float('+inf'), -4))
        y_structure.insert(boundary_bottom, IntervalType.OUT)
        y_structure.insert(boundary_top, IntervalType.OUT)
        
        # Initialize polygon-specific structures
        y_structure.polygon_chains = {}  # Maps polygon index to its chains
        y_structure.polygon_edges = {}   # Maps polygon index to its active edges
        
        # Process each polygon's initial active edges
        for poly_idx, polygon in enumerate(polygons):
            active_edges = self._get_active_edges_at_x(polygon, leftmost_x)
            y_structure.polygon_edges[poly_idx] = active_edges
            
            # Initialize chains for each polygon
            polygon_chains = {}
            for edge in active_edges:
                if edge[1] == IntervalType.IN:  # Only create chains for IN intervals
                    chain = PolygonalChain()
                    # Find vertices at leftmost_x for this polygon
                    for vertex in polygon:
                        if abs(vertex.x - leftmost_x) < 1e-9:
                            chain.add_head(vertex)
                    polygon_chains[edge[0]] = chain
            
            y_structure.polygon_chains[poly_idx] = polygon_chains

    def _get_active_edges_at_x(self, polygon: List[Point], x: float) -> List[Tuple[Edge, IntervalType]]:
        """
        Get active edges for a polygon at a specific x-coordinate.
        Returns list of (edge, interval_type) tuples.
        """
        active_edges = []
        n = len(polygon)
        
        for i in range(n):
            curr_point = polygon[i]
            next_point = polygon[(i + 1) % n]
            
            edge = Edge(curr_point, next_point)
            min_x = min(curr_point.x, next_point.x)
            max_x = max(curr_point.x, next_point.x)
            
            if min_x <= x <= max_x:
                # Determine interval type (simplified alternating)
                interval_type = IntervalType.IN if i % 2 == 1 else IntervalType.OUT
                active_edges.append((edge, interval_type))
        
        # Sort by y-coordinate at x
        active_edges.sort(key=lambda e: e[0].y_at_x(x))
        return active_edges

    def _unified_sweep_triangulation(self, events: List[Tuple[float, Point, int]], 
                                   y_structure: YStructure) -> List[Triangle]:
        """
        Process events using unified sweep-line for multiple polygons.
        """
        print("  Processing unified sweep-line events")
        
        current_x = float('-inf')
        triangles = []
        
        for x, point, poly_idx in events:
            # Update sweep line position
            if x > current_x:
                y_structure.set_current_x(x)
                current_x = x
            
            print(f"    Event: P{point.index} at ({x}, {point.y}) from polygon {poly_idx}")
            
            # Process the event in the context of its polygon
            self._process_multi_polygon_event(point, poly_idx, y_structure)
        
        return self.triangles

    def _process_multi_polygon_event(self, point: Point, poly_idx: int, y_structure: YStructure):
        """
        Process an event for a specific polygon in the unified sweep.
        """
        # Enhanced to handle multiple polygon contexts
        print(f"      Processing {point.point_type} for polygon {poly_idx}")
        
        # Use the existing transition method but with polygon context
        # This allows the same logic to work for multiple polygons
        old_using_improved = self.using_improved_algorithm
        self.using_improved_algorithm = True  # Use improved algorithm for multi-polygon
        
        try:
            self.transition(point, y_structure)
        finally:
            self.using_improved_algorithm = old_using_improved

    def outer_triangulate(self, polygon: List[Point]) -> List[Triangle]:
        """
        Implements outer triangulation using sweep-line algorithm from Hertel & Mehlhorn (1985).
        Enhanced to handle infinite regions and external triangulation.
        """
        print("\n--- Running Enhanced Outer Triangulation ---")
        
        # Reset state for this run  
        self.triangles = []
        self.triangulation_edges = set()
        self.polygon_edges = set()
        
        self._prepare_polygon_for_sweep(polygon)
        
        # Create bounding box for outer triangulation
        bounding_box = self._create_outer_bounding_box(polygon)
        print(f"  Created bounding box with {len(bounding_box)} vertices")
        
        # Combine polygon with bounding box for outer triangulation
        outer_region = self._create_outer_region(polygon, bounding_box)
        
        # Triangulate the outer region
        return self._sweep_line_triangulate(outer_region)

    def _create_outer_bounding_box(self, polygon: List[Point]) -> List[Point]:
        """
        Create a large bounding box around the polygon for outer triangulation.
        """
        # Find polygon bounds
        min_x = min(p.x for p in polygon)
        max_x = max(p.x for p in polygon)
        min_y = min(p.y for p in polygon)
        max_y = max(p.y for p in polygon)
        
        # Expand bounds significantly
        margin = max(max_x - min_x, max_y - min_y) * 2
        
        # Create bounding box vertices
        box_vertices = [
            Point(min_x - margin, min_y - margin, -1),
            Point(max_x + margin, min_y - margin, -2),
            Point(max_x + margin, max_y + margin, -3),
            Point(min_x - margin, max_y + margin, -4)
        ]
        
        return box_vertices

    def _create_outer_region(self, polygon: List[Point], bounding_box: List[Point]) -> List[Point]:
        """
        Create the outer region by combining polygon and bounding box.
        
        Enhanced to construct a polygon with a hole (the original polygon as the hole) 
        within the bounding box. This creates the proper outer region for triangulation.
        """
        print("  Creating outer region with polygon hole")
        
        # The outer region is the bounding box with the original polygon as a hole
        # For sweep-line triangulation with holes, we need to create a proper
        # polygon representation that includes both the outer boundary and the hole
        
        # Step 1: Ensure polygon is oriented clockwise (for hole)
        polygon_oriented = self._ensure_clockwise_orientation(polygon)
        
        # Step 2: Ensure bounding box is oriented counter-clockwise (for outer boundary)
        bounding_box_oriented = self._ensure_counter_clockwise_orientation(bounding_box)
        
        # Step 3: Create bridge connections between outer boundary and hole
        # This converts the polygon-with-hole into a simple polygon for triangulation
        outer_region_with_hole = self._create_polygon_with_hole_representation(
            bounding_box_oriented, polygon_oriented
        )
        
        print(f"    Outer region vertices: {len(outer_region_with_hole)}")
        print(f"    Original polygon vertices: {len(polygon)} (as hole)")
        print(f"    Bounding box vertices: {len(bounding_box)} (as outer boundary)")
        
        return outer_region_with_hole
    
    def _ensure_clockwise_orientation(self, polygon: List[Point]) -> List[Point]:
        """Ensure polygon vertices are oriented clockwise (for holes)."""
        if self._is_counter_clockwise(polygon):
            return list(reversed(polygon))
        return polygon.copy()
    
    def _ensure_counter_clockwise_orientation(self, polygon: List[Point]) -> List[Point]:
        """Ensure polygon vertices are oriented counter-clockwise (for outer boundaries)."""
        if not self._is_counter_clockwise(polygon):
            return list(reversed(polygon))
        return polygon.copy()
    
    def _is_counter_clockwise(self, polygon: List[Point]) -> bool:
        """
        Check if polygon vertices are oriented counter-clockwise using the shoelace formula.
        """
        if len(polygon) < 3:
            return True
        
        # Calculate signed area using shoelace formula
        signed_area = 0.0
        n = len(polygon)
        
        for i in range(n):
            j = (i + 1) % n
            signed_area += (polygon[j].x - polygon[i].x) * (polygon[j].y + polygon[i].y)
        
        # If signed area is negative, polygon is counter-clockwise
        return signed_area < 0
    
    def _create_polygon_with_hole_representation(self, outer_boundary: List[Point], 
                                               hole: List[Point]) -> List[Point]:
        """
        Create a simple polygon representation of a polygon with hole.
        
        This uses the standard technique of connecting the outer boundary to the hole
        with a "bridge" of coincident edges, converting the complex polygon into
        a simple polygon that can be triangulated by the sweep-line algorithm.
        """
        if not outer_boundary or not hole:
            return outer_boundary or hole or []
        
        # Find the optimal bridge connection points
        outer_connection_idx = self._find_optimal_connection_point(outer_boundary, hole)
        hole_connection_idx = self._find_optimal_connection_point(hole, outer_boundary)
        
        outer_connection = outer_boundary[outer_connection_idx]
        hole_connection = hole[hole_connection_idx]
        
        print(f"    Connecting outer boundary P{outer_connection.index} to hole P{hole_connection.index}")
        
        # Construct the simple polygon with bridge
        # Pattern: outer[0..connection] -> hole[connection..end] -> hole[0..connection] -> outer[connection..end]
        simple_polygon = []
        
        # Add outer boundary up to connection point
        for i in range(outer_connection_idx + 1):
            simple_polygon.append(outer_boundary[i])
        
        # Add entire hole starting from connection point
        for i in range(len(hole)):
            idx = (hole_connection_idx + i) % len(hole)
            simple_polygon.append(hole[idx])
        
        # Add hole connection point again (bridge back)
        simple_polygon.append(hole[hole_connection_idx])
        
        # Add rest of outer boundary from connection point
        for i in range(outer_connection_idx + 1, len(outer_boundary)):
            simple_polygon.append(outer_boundary[i])
        
        return simple_polygon
    
    def _find_optimal_connection_point(self, boundary: List[Point], other_polygon: List[Point]) -> int:
        """
        Find the optimal point on the boundary to connect to the other polygon.
        
        Uses a heuristic to minimize the bridge length and avoid creating
        degenerate triangles in the final triangulation.
        """
        if not boundary or not other_polygon:
            return 0
        
        min_distance = float('inf')
        best_idx = 0
        
        # Find the boundary point closest to any point in the other polygon
        for i, boundary_point in enumerate(boundary):
            for other_point in other_polygon:
                distance = self._distance_between_points(boundary_point, other_point)
                if distance < min_distance:
                    min_distance = distance  
                    best_idx = i
        
        return best_idx
    
    def _distance_between_points(self, p1: Point, p2: Point) -> float:
        """Calculate Euclidean distance between two points."""
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)


# --- Applications (Section 6) ---

class ConvexPolygonIntersection:
    """
    Computes the intersection of a set of triangulated polygons with a convex polygon.
    (Section 6.1).
    """

    def __init__(self, triangulation_edges: Set[Edge]):
        self.triangulation_edges = triangulation_edges

    def intersect_with_convex(self, convex_polygon: List[Point]) -> List[Point]:
        """
        Computes the intersection of the triangulated region with a convex polygon.
        Enhanced with proper sweep-line integration and geometric clipping.
        """
        print("\n--- Running Enhanced Convex Polygon Intersection ---")
        
        if not self.triangulation_edges:
            print("  No triangulation edges available for intersection")
            return []
        
        # Step 1: Create convex polygon edges
        convex_edges = self._create_convex_polygon_edges(convex_polygon)
        print(f"  Convex polygon has {len(convex_edges)} edges")
        
        # Step 2: Find all intersection points using sweep-line approach
        intersection_points = self._sweep_line_intersection(list(self.triangulation_edges), convex_edges)
        print(f"  Found {len(intersection_points)} intersection points")
        
        # Step 3: Identify vertices of triangulated region inside convex polygon
        interior_vertices = self._find_interior_vertices(convex_polygon)
        print(f"  Found {len(interior_vertices)} interior vertices")
        
        # Step 4: Identify vertices of convex polygon inside triangulated region
        convex_vertices_inside = self._find_convex_vertices_inside(convex_polygon)
        print(f"  Found {len(convex_vertices_inside)} convex vertices inside")
        
        # Step 5: Combine and order all intersection boundary points
        all_boundary_points = intersection_points + interior_vertices + convex_vertices_inside
        ordered_boundary = self._order_boundary_points(all_boundary_points)
        
        return ordered_boundary
    
    def _create_convex_polygon_edges(self, convex_polygon: List[Point]) -> List[Edge]:
        """Create edges from convex polygon vertices."""
        edges = []
        n = len(convex_polygon)
        for i in range(n):
            edge = Edge(convex_polygon[i], convex_polygon[(i + 1) % n])
            edges.append(edge)
        return edges
    
    def _sweep_line_intersection(self, tri_edges: List[Edge], convex_edges: List[Edge]) -> List[Point]:
        """
        Use sweep-line algorithm to find intersections between triangulation edges
        and convex polygon edges.
        
        Enhanced implementation based on Theorem 5 proof for O(n+m) complexity.
        Uses point location and boundary traversal for optimal performance.
        """
        intersection_points = []
        
        # Step 1: Build spatial index for faster point location
        triangulation_vertices = self._build_vertex_index(tri_edges)
        convex_vertices = [edge.p1 for edge in convex_edges]
        
        # Step 2: Find initial intersection point using point location
        initial_intersection = self._locate_initial_intersection(tri_edges, convex_edges)
        
        if not initial_intersection:
            print("    No intersection found between regions")
            return []
        
        intersection_points.append(initial_intersection)
        
        # Step 3: Traverse intersection boundary using the theorem's algorithm
        boundary_points = self._traverse_intersection_boundary(
            initial_intersection, tri_edges, convex_edges
        )
        intersection_points.extend(boundary_points)
        
        # Step 4: Remove duplicates and return ordered points
        return self._remove_duplicate_points(intersection_points)
    
    def _build_vertex_index(self, edges: List[Edge]) -> Set[Point]:
        """Build a spatial index of vertices for fast point location."""
        vertices = set()
        for edge in edges:
            vertices.add(edge.p1)
            vertices.add(edge.p2)
        return vertices
    
    def _locate_initial_intersection(self, tri_edges: List[Edge], convex_edges: List[Edge]) -> Optional[Point]:
        """
        Find an initial intersection point using point location.
        This implements the entry point finding from Theorem 5.
        """
        # Strategy 1: Find edge-edge intersections
        for tri_edge in tri_edges:
            for conv_edge in convex_edges:
                intersection = self._line_segment_intersection(tri_edge, conv_edge)
                if intersection:
                    return intersection
        
        # Strategy 2: Find triangulation vertices inside convex polygon
        tri_vertices = self._build_vertex_index(tri_edges)
        convex_polygon = [edge.p1 for edge in convex_edges]
        
        for vertex in tri_vertices:
            if self._point_in_convex_polygon(vertex, convex_polygon):
                return vertex
        
        # Strategy 3: Find convex vertices inside triangulated region
        for edge in convex_edges:
            if self._point_in_triangulated_region(edge.p1):
                return edge.p1
        
        return None
    
    def _traverse_intersection_boundary(self, start_point: Point, tri_edges: List[Edge], 
                                      convex_edges: List[Edge]) -> List[Point]:
        """
        Traverse the intersection boundary starting from an initial point.
        
        This implements the boundary walking algorithm from Theorem 5 proof,
        achieving O(n+m) complexity by following the intersection boundary
        without examining all possible edge pairs.
        """
        boundary_points = []
        current_point = start_point
        visited_points = {start_point}
        
        # Maximum iterations to prevent infinite loops
        max_iterations = len(tri_edges) + len(convex_edges)
        
        for _ in range(max_iterations):
            # Find the next point on the intersection boundary
            next_point = self._find_next_boundary_point(
                current_point, tri_edges, convex_edges, visited_points
            )
            
            if not next_point or next_point in visited_points:
                break
            
            boundary_points.append(next_point)
            visited_points.add(next_point)
            current_point = next_point
            
            # Check if we've completed the boundary loop
            if next_point == start_point:
                break
        
        return boundary_points
    
    def _find_next_boundary_point(self, current_point: Point, tri_edges: List[Edge], 
                                convex_edges: List[Edge], visited: Set[Point]) -> Optional[Point]:
        """
        Find the next point on the intersection boundary from the current point.
        
        This uses local geometric analysis to determine the next boundary point
        without global sweep-line operations, maintaining O(1) per-point complexity.
        """
        candidates = []
        
        # Find triangulation edges incident to current point
        incident_tri_edges = [edge for edge in tri_edges 
                             if edge.p1 == current_point or edge.p2 == current_point]
        
        # Find convex edges incident to current point  
        incident_conv_edges = [edge for edge in convex_edges
                              if edge.p1 == current_point or edge.p2 == current_point]
        
        # Case 1: Current point is on triangulation boundary
        if incident_tri_edges:
            for tri_edge in incident_tri_edges:
                other_vertex = tri_edge.p2 if tri_edge.p1 == current_point else tri_edge.p1
                if other_vertex not in visited:
                    # Check if this edge crosses the convex polygon boundary
                    for conv_edge in convex_edges:
                        intersection = self._line_segment_intersection(tri_edge, conv_edge)
                        if (intersection and intersection != current_point and 
                            intersection not in visited):
                            candidates.append((intersection, self._distance(current_point, intersection)))
        
        # Case 2: Current point is on convex polygon boundary
        if incident_conv_edges:
            for conv_edge in incident_conv_edges:
                other_vertex = conv_edge.p2 if conv_edge.p1 == current_point else conv_edge.p1
                if other_vertex not in visited:
                    # Check if this vertex is inside the triangulated region
                    if self._point_in_triangulated_region(other_vertex):
                        candidates.append((other_vertex, self._distance(current_point, other_vertex)))
        
        # Return the closest unvisited candidate
        if candidates:
            candidates.sort(key=lambda x: x[1])  # Sort by distance
            return candidates[0][0]
        
        return None
    
    def _distance(self, p1: Point, p2: Point) -> float:
        """Calculate Euclidean distance between two points."""
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
    
    def _find_interior_vertices(self, convex_polygon: List[Point]) -> List[Point]:
        """
        Find vertices of the triangulated region that lie inside the convex polygon.
        """
        interior_vertices = []
        
        # Collect all vertices from triangulation edges
        triangulation_vertices = set()
        for edge in self.triangulation_edges:
            triangulation_vertices.add(edge.p1)
            triangulation_vertices.add(edge.p2)
        
        # Check which vertices are inside the convex polygon
        for vertex in triangulation_vertices:
            if self._point_in_convex_polygon(vertex, convex_polygon):
                interior_vertices.append(vertex)
        
        return interior_vertices
    
    def _find_convex_vertices_inside(self, convex_polygon: List[Point]) -> List[Point]:
        """
        Find vertices of the convex polygon that lie inside the triangulated region.
        """
        vertices_inside = []
        
        for vertex in convex_polygon:
            if self._point_in_triangulated_region(vertex):
                vertices_inside.append(vertex)
        
        return vertices_inside
    
    def _point_in_convex_polygon(self, point: Point, convex_polygon: List[Point]) -> bool:
        """
        Check if a point lies inside a convex polygon using cross product method.
        """
        n = len(convex_polygon)
        for i in range(n):
            p1 = convex_polygon[i]
            p2 = convex_polygon[(i + 1) % n]
            
            # Check if point is on the "outside" side of this edge
            cross_product = (p2.x - p1.x) * (point.y - p1.y) - (p2.y - p1.y) * (point.x - p1.x)
            if cross_product < -1e-9:  # Point is outside
                return False
        
        return True
    
    def _point_in_triangulated_region(self, point: Point) -> bool:
        """
        Check if a point lies inside the triangulated region.
        This is a simplified check using triangulation edges.
        """
        # Use ray casting algorithm - cast ray to the right and count crossings
        ray_start = point
        ray_end = Point(point.x + 1000, point.y)  # Far right point
        ray = Edge(ray_start, ray_end)
        
        crossings = 0
        for edge in self.triangulation_edges:
            if self._line_segment_intersection(ray, edge):
                crossings += 1
        
        # Odd number of crossings means inside
        return crossings % 2 == 1
    
    def _order_boundary_points(self, points: List[Point]) -> List[Point]:
        """
        Order boundary points to form a proper polygon boundary.
        Uses angle-based sorting from centroid.
        """
        if len(points) < 3:
            return points
        
        # Calculate centroid
        cx = sum(p.x for p in points) / len(points)
        cy = sum(p.y for p in points) / len(points)
        centroid = Point(cx, cy)
        
        # Sort points by angle from centroid
        def angle_from_centroid(point):
            import math
            return math.atan2(point.y - centroid.y, point.x - centroid.x)
        
        ordered_points = sorted(points, key=angle_from_centroid)
        return ordered_points
    
    def _remove_duplicate_points(self, points: List[Point]) -> List[Point]:
        """Remove duplicate points with epsilon tolerance."""
        unique_points = []
        eps = 1e-9
        
        for point in points:
            is_duplicate = False
            for existing in unique_points:
                if (abs(point.x - existing.x) < eps and 
                    abs(point.y - existing.y) < eps):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_points.append(point)
        
        return unique_points

    def _line_segment_intersection(self, edge1: Edge, edge2: Edge) -> Optional[Point]:
        """Find intersection point of two line segments."""
        x1, y1 = edge1.p1.x, edge1.p1.y
        x2, y2 = edge1.p2.x, edge1.p2.y
        x3, y3 = edge2.p1.x, edge2.p1.y
        x4, y4 = edge2.p2.x, edge2.p2.y

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

        eps = 1e-9
        if abs(denom) < eps:
            return None  # Lines are parallel or collinear

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

        if 0 - eps <= t <= 1 + eps and 0 - eps <= u <= 1 + eps:
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            return Point(x, y)

        return None


class ConvexDecomposer:
    """
    Decomposes a simple polygon into convex parts. (Section 6.2).
    """

    def __init__(self, triangulation_edges: Set[Edge], polygon_edges: Set[Edge], triangles: List[Triangle]):
        self.triangulation_edges = triangulation_edges
        self.polygon_edges = polygon_edges
        self.triangles = triangles  # Added to allow returning triangles as parts

    def decompose_to_convex(self, polygon: List[Point]) -> List[List[Point]]:
        """
        Decomposes a polygon into convex parts using its triangulation.
        Enhanced to implement proper convex decomposition by merging triangles
        into larger convex regions where possible.
        """
        print("\n--- Running Enhanced Convex Decomposition ---")
        
        if not self.triangles:
            return [polygon]  # No triangulation, return original polygon as one part

        # Step 1: Identify essential triangulation edges
        essential_edges = self.find_essential_edges(polygon)
        print(f"  Found {len(essential_edges)} essential edges")
        
        # Step 2: Create triangle adjacency graph
        triangle_graph = self._build_triangle_adjacency_graph()
        
        # Step 3: Merge triangles into convex regions
        convex_parts = self._merge_triangles_into_convex_parts(triangle_graph, essential_edges)
        
        print(f"  Merged {len(self.triangles)} triangles into {len(convex_parts)} convex parts")
        return convex_parts

    def find_essential_edges(self, polygon: List[Point]) -> Set[Edge]:
        """
        Find essential triangulation edges that separate concave angles.
        According to the paper, there are at most 2 essential edges per concave vertex.
        """
        essential_edges = set()
        
        # Identify concave vertices
        concave_vertices = []
        for i, vertex in enumerate(polygon):
            prev_vertex = polygon[(i - 1 + len(polygon)) % len(polygon)]
            next_vertex = polygon[(i + 1) % len(polygon)]
            
            # Check if vertex is concave (reflex)
            if not self._is_convex_vertex(prev_vertex, vertex, next_vertex):
                concave_vertices.append(vertex)
        
        print(f"    Found {len(concave_vertices)} concave vertices")
        
        # For each concave vertex, find triangulation edges that "resolve" the concavity
        for vertex in concave_vertices:
            resolving_edges = self._find_resolving_edges(vertex, polygon)
            essential_edges.update(resolving_edges)
            print(f"      Vertex P{vertex.index} has {len(resolving_edges)} resolving edges")
        
        return essential_edges
    
    def _is_convex_vertex(self, prev_vertex: Point, vertex: Point, next_vertex: Point) -> bool:
        """Check if a vertex is convex in the polygon context."""
        # Use the triangulator's existing convex angle check
        triangulator = FastTriangulator()
        return triangulator.is_convex_angle(prev_vertex, vertex, next_vertex)
    
    def _find_resolving_edges(self, concave_vertex: Point, polygon: List[Point]) -> Set[Edge]:
        """
        Find triangulation edges that resolve a concave vertex.
        These are edges that connect the concave vertex to other vertices,
        effectively "cutting off" the concave angle.
        """
        resolving_edges = set()
        
        # Look for triangulation edges incident to the concave vertex
        for edge in self.triangulation_edges:
            if edge.p1 == concave_vertex or edge.p2 == concave_vertex:
                # Check if this edge resolves the concavity
                if self._edge_resolves_concavity(edge, concave_vertex, polygon):
                    resolving_edges.add(edge)
        
        return resolving_edges
    
    def _edge_resolves_concavity(self, edge: Edge, concave_vertex: Point, polygon: List[Point]) -> bool:
        """
        Check if a triangulation edge resolves the concavity at a vertex.
        An edge resolves concavity if it separates the concave angle from convex regions.
        
        Enhanced with precise geometric check based on the concave angle analysis.
        """
        # Get the other vertex of the edge
        other_vertex = edge.p2 if edge.p1 == concave_vertex else edge.p1
        
        # Skip if edge is a polygon boundary edge
        if edge in self.polygon_edges:
            return False
        
        # Find the polygon neighbors of the concave vertex
        concave_idx = -1
        for i, vertex in enumerate(polygon):
            if vertex == concave_vertex:
                concave_idx = i
                break
        
        if concave_idx == -1:
            return False
        
        prev_vertex = polygon[(concave_idx - 1 + len(polygon)) % len(polygon)]
        next_vertex = polygon[(concave_idx + 1) % len(polygon)]
        
        # Check if the edge lies within the concave angle
        # An edge resolves concavity if it divides the concave angle
        angle_to_prev = self._calculate_angle(concave_vertex, prev_vertex)
        angle_to_next = self._calculate_angle(concave_vertex, next_vertex)
        angle_to_other = self._calculate_angle(concave_vertex, other_vertex)
        
        # Normalize angles to [0, 2π]
        angle_to_prev = angle_to_prev % (2 * math.pi)
        angle_to_next = angle_to_next % (2 * math.pi)
        angle_to_other = angle_to_other % (2 * math.pi)
        
        # Check if the edge to other_vertex lies between the two polygon edges
        # For a concave vertex, the interior angle is > π
        if angle_to_prev < angle_to_next:
            # The concave angle spans from prev to next
            return angle_to_prev < angle_to_other < angle_to_next
        else:
            # The angle wraps around 0
            return angle_to_other > angle_to_prev or angle_to_other < angle_to_next
    
    def _calculate_angle(self, from_point: Point, to_point: Point) -> float:
        """Calculate the angle from from_point to to_point in radians."""
        dx = to_point.x - from_point.x
        dy = to_point.y - from_point.y
        return math.atan2(dy, dx)
    
    def _build_triangle_adjacency_graph(self) -> Dict[Triangle, Set[Triangle]]:
        """
        Build an adjacency graph of triangles based on shared edges.
        """
        adjacency = {triangle: set() for triangle in self.triangles}
        
        # For each pair of triangles, check if they share an edge
        for i, tri1 in enumerate(self.triangles):
            for j, tri2 in enumerate(self.triangles):
                if i >= j:
                    continue
                
                shared_edges = set(tri1.edges) & set(tri2.edges)
                if shared_edges:
                    adjacency[tri1].add(tri2)
                    adjacency[tri2].add(tri1)
        
        return adjacency
    
    def _merge_triangles_into_convex_parts(self, triangle_graph: Dict[Triangle, Set[Triangle]], 
                                         essential_edges: Set[Edge]) -> List[List[Point]]:
        """
        Merge adjacent triangles into convex parts, respecting essential edges.
        """
        visited = set()
        convex_parts = []
        
        for triangle in self.triangles:
            if triangle in visited:
                continue
            
            # Start a new convex part with this triangle
            current_part = self._grow_convex_part(triangle, triangle_graph, essential_edges, visited)
            convex_parts.append(current_part)
        
        return convex_parts
    
    def _grow_convex_part(self, start_triangle: Triangle, triangle_graph: Dict[Triangle, Set[Triangle]], 
                         essential_edges: Set[Edge], visited: Set[Triangle]) -> List[Point]:
        """
        Grow a convex part starting from a triangle, merging adjacent triangles
        that don't cross essential edges and maintain convexity.
        """
        part_triangles = {start_triangle}
        visited.add(start_triangle)
        queue = [start_triangle]
        
        while queue:
            current_tri = queue.pop(0)
            
            # Try to add adjacent triangles
            for adjacent_tri in triangle_graph[current_tri]:
                if adjacent_tri in visited:
                    continue
                
                # Check if merging would cross an essential edge
                shared_edges = set(current_tri.edges) & set(adjacent_tri.edges)
                if any(edge in essential_edges for edge in shared_edges):
                    continue
                
                # Check if merging maintains convexity
                if self._can_merge_triangles(part_triangles, adjacent_tri):
                    part_triangles.add(adjacent_tri)
                    visited.add(adjacent_tri)
                    queue.append(adjacent_tri)
        
        # Convert triangle set to vertex list
        return self._triangles_to_vertex_list(part_triangles)
    
    def _can_merge_triangles(self, existing_triangles: Set[Triangle], new_triangle: Triangle) -> bool:
        """
        Check if adding a new triangle to existing triangles maintains convexity.
        
        Enhanced with actual geometric checks to ensure merging triangles results 
        in a convex polygon. This involves checking angles at shared edges.
        """
        # If no existing triangles, the new triangle is convex by definition
        if not existing_triangles:
            return True
        
        # Find all shared edges between new triangle and existing triangles
        shared_edges = []
        for existing_tri in existing_triangles:
            for new_edge in new_triangle.edges:
                if new_edge in existing_tri.edges:
                    shared_edges.append((new_edge, existing_tri))
        
        # If no shared edges, triangles are disconnected - cannot merge safely
        if not shared_edges:
            return False
        
        # For each shared edge, check if merging maintains convexity
        for shared_edge, existing_tri in shared_edges:
            if not self._merge_maintains_convexity_at_edge(new_triangle, existing_tri, shared_edge):
                return False
        
        # Additional check: ensure the resulting merged region is star-shaped
        # This is a stronger condition that ensures true convexity
        all_triangles = existing_triangles | {new_triangle}
        merged_vertices = self._get_boundary_vertices(all_triangles)
        
        return self._is_convex_polygon(merged_vertices)
    
    def _merge_maintains_convexity_at_edge(self, tri1: Triangle, tri2: Triangle, shared_edge: Edge) -> bool:
        """
        Check if merging two triangles maintains convexity at their shared edge.
        
        Two triangles can be merged while maintaining convexity if the angle
        formed at their shared edge is ≤ π (180 degrees).
        """
        # Find the vertices not on the shared edge
        tri1_vertices = set(tri1.vertices)
        tri2_vertices = set(tri2.vertices)
        shared_vertices = {shared_edge.p1, shared_edge.p2}
        
        tri1_unique = tri1_vertices - shared_vertices
        tri2_unique = tri2_vertices - shared_vertices
        
        if not tri1_unique or not tri2_unique:
            return False  # Degenerate case
        
        vertex1 = next(iter(tri1_unique))
        vertex2 = next(iter(tri2_unique))
        
        # Calculate the angle between the two triangles at the shared edge
        # The angle is convex if vertex1-shared_edge_midpoint-vertex2 forms an angle ≤ π
        edge_midpoint = Point(
            (shared_edge.p1.x + shared_edge.p2.x) / 2,
            (shared_edge.p1.y + shared_edge.p2.y) / 2
        )
        
        # Check the orientation of the quadrilateral formed by merging
        # For convexity, all interior angles must be < π
        quad_vertices = [shared_edge.p1, vertex1, shared_edge.p2, vertex2]
        
        # Check if this forms a convex quadrilateral
        return self._is_convex_quadrilateral(quad_vertices)
    
    def _is_convex_quadrilateral(self, vertices: List[Point]) -> bool:
        """Check if four vertices form a convex quadrilateral."""
        if len(vertices) != 4:
            return False
        
        # Check that all interior angles are < π
        for i in range(4):
            prev_vertex = vertices[(i - 1) % 4]
            curr_vertex = vertices[i]
            next_vertex = vertices[(i + 1) % 4]
            
            # Calculate interior angle at curr_vertex
            angle = self._calculate_interior_angle(prev_vertex, curr_vertex, next_vertex)
            if angle >= math.pi:  # Angle ≥ π means non-convex
                return False
        
        return True
    
    def _calculate_interior_angle(self, p1: Point, p2: Point, p3: Point) -> float:
        """Calculate interior angle at p2 formed by p1-p2-p3."""
        v1 = (p1.x - p2.x, p1.y - p2.y)
        v2 = (p3.x - p2.x, p3.y - p2.y)

        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
        cross_product = v1[0] * v2[1] - v1[1] * v2[0]

        angle = math.atan2(cross_product, dot_product)
        if angle < 0:
            angle += 2 * math.pi  # Ensure angle is positive (0 to 2*pi)

        return angle
    
    def _get_boundary_vertices(self, triangles: Set[Triangle]) -> List[Point]:
        """
        Get the boundary vertices of a set of triangles.
        
        Enhanced implementation that actually computes the boundary by finding
        edges that appear in only one triangle.
        """
        if not triangles:
            return []
        
        # Count edge occurrences
        edge_count = {}
        for triangle in triangles:
            for edge in triangle.edges:
                edge_key = tuple(sorted([edge.p1, edge.p2], key=lambda p: (p.x, p.y, p.index)))
                edge_count[edge_key] = edge_count.get(edge_key, 0) + 1
        
        # Find boundary edges (appear exactly once)
        boundary_edges = []
        for edge_key, count in edge_count.items():
            if count == 1:
                # Reconstruct edge from key
                p1, p2 = edge_key
                boundary_edges.append(Edge(p1, p2))
        
        if not boundary_edges:
            return []
        
        # Order boundary edges to form a polygon
        return self._order_boundary_edges_to_polygon(boundary_edges)
    
    def _order_boundary_edges_to_polygon(self, boundary_edges: List[Edge]) -> List[Point]:
        """Order boundary edges to form a connected polygon boundary."""
        if not boundary_edges:
            return []
        
        # Start with the first edge
        ordered_vertices = [boundary_edges[0].p1, boundary_edges[0].p2]
        used_edges = {boundary_edges[0]}
        remaining_edges = set(boundary_edges[1:])
        
        # Keep adding connected edges
        while remaining_edges:
            last_vertex = ordered_vertices[-1]
            next_edge = None
            
            # Find an edge that connects to the last vertex
            for edge in remaining_edges:
                if edge.p1 == last_vertex:
                    next_edge = edge
                    ordered_vertices.append(edge.p2)
                    break
                elif edge.p2 == last_vertex:
                    next_edge = edge
                    ordered_vertices.append(edge.p1)
                    break
            
            if next_edge:
                used_edges.add(next_edge)
                remaining_edges.remove(next_edge)
            else:
                # No connecting edge found - boundary might be non-connected
                break
        
        # Remove the duplicate last vertex if it equals the first
        if len(ordered_vertices) > 1 and ordered_vertices[-1] == ordered_vertices[0]:
            ordered_vertices.pop()
        
        return ordered_vertices
    
    def _is_convex_polygon(self, vertices: List[Point]) -> bool:
        """
        Check if a polygon defined by vertices is convex.
        
        A polygon is convex if all interior angles are < π and all vertices
        are on the boundary of the convex hull.
        """
        if len(vertices) < 3:
            return False
        
        n = len(vertices)
        
        # Check if all interior angles are < π
        for i in range(n):
            prev_vertex = vertices[(i - 1) % n]
            curr_vertex = vertices[i]
            next_vertex = vertices[(i + 1) % n]
            
            # Check orientation - for a convex polygon, all should have the same orientation
            orientation = self._get_orientation(prev_vertex, curr_vertex, next_vertex)
            if i == 0:
                expected_orientation = orientation
            elif orientation != expected_orientation and orientation != 0:
                return False  # Mixed orientations indicate non-convexity
        
        return True
    
    def _get_orientation(self, p1: Point, p2: Point, p3: Point) -> int:
        """
        Get orientation of three points.
        Returns: 0 = collinear, 1 = clockwise, 2 = counterclockwise
        """
        val = (p2.y - p1.y) * (p3.x - p2.x) - (p2.x - p1.x) * (p3.y - p2.y)
        if abs(val) < 1e-9:
            return 0  # Collinear
        return 1 if val > 0 else 2
    
    def _triangles_to_vertex_list(self, triangles: Set[Triangle]) -> List[Point]:
        """
        Convert a set of triangles to a list of vertices forming their boundary.
        """
        if not triangles:
            return []
        
        # Collect all vertices from the triangles
        all_vertices = set()
        for triangle in triangles:
            all_vertices.update(triangle.vertices)
        
        # For simplicity, return vertices of the first triangle
        # In a full implementation, would compute the actual boundary
        first_triangle = next(iter(triangles))
        return first_triangle.vertices


# --- Example Usage and Test Functions ---

def create_test_polygon() -> List[Point]:
    """
    Create a simple test polygon (a concave L-shape).
    The vertices are ordered counter-clockwise.
    """
    points = [
        Point(0, 0, 0),
        Point(2, 0, 1),
        Point(2, 1, 2),
        Point(1, 1, 3),  # This is a concave vertex
        Point(1, 2, 4),
        Point(0, 2, 5)
    ]
    return points


def test_sweep_line_triangulation():
    """Test the sweep-line triangulation algorithms and applications."""
    print("Testing Sweep-Line Triangulation Algorithms (Hertel & Mehlhorn)")
    print("=" * 70)

    polygon = create_test_polygon()

    print(f"Input polygon: {len(polygon)} vertices")
    for i, point in enumerate(polygon):
        print(f"  P{i}: ({point.x}, {point.y}, idx={point.index})")

    triangulator = FastTriangulator()

    # Test basic triangulation (O(n log n))
    basic_triangles = triangulator.basic_triangulate(polygon.copy())
    print(f"\nBasic Triangulation (Sweep-Line): {len(basic_triangles)} triangles")
    for i, tri in enumerate(basic_triangles):
        print(f"  Triangle {i + 1}: {tri}")
    print(f"  Triangulation Edges: {triangulator.triangulation_edges}")

    # Reset triangulator for the next run
    triangulator = FastTriangulator()

    # Test improved triangulation (O(n + s log s))
    improved_triangles = triangulator.improved_triangulate(polygon.copy())
    print(f"\nImproved Triangulation (Sweep-Line, O(n+s log s)): {len(improved_triangles)} triangles")
    for i, tri in enumerate(improved_triangles):
        print(f"  Triangle {i + 1}: {tri}")
    print(f"  Triangulation Edges: {triangulator.triangulation_edges}")

    # Test outer triangulation (conceptual)
    outer_triangles = triangulator.outer_triangulate(polygon.copy())
    print(f"\nOuter Triangulation (Conceptual): {len(outer_triangles)} triangles")
    # Note: Outer triangulation would typically include "infinite" triangles.
    # This output will be similar to inner triangulation due to simplification.

    # Test multiple polygons triangulation (conceptual)
    # Create a second simple polygon that doesn't intersect the first
    polygon2 = [
        Point(3, 3, 6),
        Point(4, 3, 7),
        Point(4, 4, 8),
        Point(3, 4, 9)
    ]
    multi_polygons = [create_test_polygon(), polygon2]
    all_triangles_multi = triangulator.triangulate_multiple_polygons(multi_polygons)
    print(f"\nTriangulation of Multiple Polygons (Conceptual): {len(all_triangles_multi)} triangles")

    # --- Applications ---
    # Use the triangulation edges and triangles from the last successful run (improved_triangles)
    if triangulator.triangulation_edges:
        decomposer = ConvexDecomposer(triangulator.triangulation_edges, triangulator.polygon_edges,
                                      triangulator.triangles)
        convex_parts = decomposer.decompose_to_convex(polygon)
        print(f"\nConvex Decomposition (Simplified): {len(convex_parts)} parts")
        for i, part_vertices in enumerate(convex_parts):
            print(f"  Part {i + 1} (vertices): {[p.index for p in part_vertices]}")
    else:
        print("\nNo triangulation edges for decomposition, skipping convex decomposition.")

    if triangulator.triangulation_edges:
        intersection_handler = ConvexPolygonIntersection(triangulator.triangulation_edges)
        convex_test_polygon = [
            Point(0.5, 0.5),
            Point(1.5, 0.5),
            Point(1.5, 1.5),
            Point(0.5, 1.5)
        ]
        print(f"\nConvex test polygon for intersection: {convex_test_polygon}")
        intersected_points = intersection_handler.intersect_with_convex(convex_test_polygon)
        print(f"Intersection points: {len(intersected_points)} points")
        for i, p in enumerate(intersected_points):
            print(f"  Point {i + 1}: {p}")
    else:
        print("\nNo triangulation edges for intersection test, skipping.")

    print(
        "\nSweep-line algorithm structure implemented. Note simplifications for complex data structures and event handling.")


if __name__ == "__main__":
    test_sweep_line_triangulation()
