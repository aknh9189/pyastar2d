#include <queue>
#include <limits>
#include <cmath>
#include <Python.h>
#include <numpy/arrayobject.h>
#include <iostream>
#include <experimental_heuristics.h>


const float INF = std::numeric_limits<float>::infinity();

// represents a single pixel
class Node {
  public:
    int idx; // index in the flattened grid
    float cost; // cost of traversing this pixel
    int path_length; // the length of the path to reach this node

    Node(int i, float c, int path_length) : idx(i), cost(c), path_length(path_length) {}
};

// the top of the priority queue is the greatest element by default,
// but we want the smallest, so flip the sign
bool operator<(const Node &n1, const Node &n2) {
  return n1.cost > n2.cost;
}

// See for various grid heuristics:
// http://theory.stanford.edu/~amitp/GameProgramming/Heuristics.html#S7
// L_\inf norm (diagonal distance)
inline float linf_norm(int i0, int j0, int i1, int j1) {
  return std::max(std::abs(i0 - i1), std::abs(j0 - j1));
}

// L_1 norm (manhattan distance)
inline float l1_norm(int i0, int j0, int i1, int j1) {
  return std::abs(i0 - i1) + std::abs(j0 - j1);
}


// weights:        flattened h x w grid of costs
// h, w:           height and width of grid
// start, goal:    index of start/goal in flattened grid
// diag_ok:        if true, allows diagonal moves (8-conn.)
// paths (output): for each node, stores previous node in path
static PyObject * astar(PyObject *self, PyObject *args) {
  const PyArrayObject* weights_object;
  int h;
  int w;
  int start;
  int goal;
  int diag_ok;
  int heuristic_override;

  if (!PyArg_ParseTuple(
        args, "Oiiiiii", // i = int, O = object
        &weights_object,
        &h, &w,
        &start, &goal,
        &diag_ok, &heuristic_override
        ))
    return NULL;

  float* weights = (float*) weights_object->data;
  int* paths = new int[h * w];
  int path_length = -1;

  Node start_node(start, 0., 1);

  float* costs = new float[h * w];
  for (int i = 0; i < h * w; ++i)
    costs[i] = INF;
  costs[start] = 0.;

  std::priority_queue<Node> nodes_to_visit;
  nodes_to_visit.push(start_node);

  int* nbrs = new int[8];
  
  int goal_i = goal / w;
  int goal_j = goal % w;
  int start_i = start / w;
  int start_j = start % w;

  heuristic_ptr heuristic_func = select_heuristic(heuristic_override);

  Node closest_node(start, 0.0, 1);
  float closest_node_h = INF;

  // std::cout << "in astar" << std::endl;

  while (!nodes_to_visit.empty()) {
    // .top() doesn't actually remove the node
    Node cur = nodes_to_visit.top();

    if (cur.idx == goal) {
      path_length = cur.path_length;
      break;
    }

    nodes_to_visit.pop();

    int row = cur.idx / w;
    int col = cur.idx % w;
    // check bounds and find up to eight neighbors: top to bottom, left to right
    nbrs[0] = (diag_ok && row > 0 && col > 0)          ? cur.idx - w - 1   : -1;
    nbrs[1] = (row > 0)                                ? cur.idx - w       : -1;
    nbrs[2] = (diag_ok && row > 0 && col + 1 < w)      ? cur.idx - w + 1   : -1;
    nbrs[3] = (col > 0)                                ? cur.idx - 1       : -1;
    nbrs[4] = (col + 1 < w)                            ? cur.idx + 1       : -1;
    nbrs[5] = (diag_ok && row + 1 < h && col > 0)      ? cur.idx + w - 1   : -1;
    nbrs[6] = (row + 1 < h)                            ? cur.idx + w       : -1;
    nbrs[7] = (diag_ok && row + 1 < h && col + 1 < w ) ? cur.idx + w + 1   : -1;

    float heuristic_cost;
    for (int i = 0; i < 8; ++i) {
      if (nbrs[i] >= 0) {
        // the sum of the cost so far and the cost of this move
        float new_cost = costs[cur.idx] + weights[nbrs[i]];
        if (new_cost < costs[nbrs[i]]) {
          // estimate the cost to the goal based on legal moves
          // Get the heuristic method to use
          if (heuristic_override == DEFAULT) {
            if (diag_ok) {
              heuristic_cost = linf_norm(nbrs[i] / w, nbrs[i] % w, goal_i, goal_j);
            } else {
              heuristic_cost = l1_norm(nbrs[i] / w, nbrs[i] % w, goal_i, goal_j);
            }
          } else {
            heuristic_cost = heuristic_func(
              nbrs[i] / w, nbrs[i] % w, goal_i, goal_j, start_i, start_j);
          }

          // update the closest node to the goal based on the heuristic
          if (heuristic_cost < closest_node_h) {
            closest_node_h = heuristic_cost;
            closest_node = Node(nbrs[i], heuristic_cost, cur.path_length + 1);
          }

          // paths with lower expected cost are explored first
          float priority = new_cost + heuristic_cost;
          nodes_to_visit.push(Node(nbrs[i], priority, cur.path_length + 1));

          costs[nbrs[i]] = new_cost;
          paths[nbrs[i]] = cur.idx;
        }
      }
    }
  }
  
  PyObject *return_val;
  return_val = PyTuple_New(2);
  // std::cout << "made it to return" << std::endl;
  if (path_length >= 0) {
    npy_intp dims[2] = {path_length, 2};
    PyArrayObject* path = (PyArrayObject*) PyArray_SimpleNew(2, dims, NPY_INT32);
    npy_int32 *iptr, *jptr;
    int idx = goal;
    for (npy_intp i = dims[0] - 1; i >= 0; --i) {
        iptr = (npy_int32*) (path->data + i * path->strides[0]);
        jptr = (npy_int32*) (path->data + i * path->strides[0] + path->strides[1]);

        *iptr = idx / w;
        *jptr = idx % w;

        idx = paths[idx];
    }
    // std::cout << "made it to return 2" << std::endl;
    PyTuple_SET_ITEM(return_val, 0, Py_True);
    PyTuple_SET_ITEM(return_val, 1, PyArray_Return(path));
  }
  else if (closest_node.path_length > 0) { // if a goal is not found, return a path to the node we reached closest to the goal
    npy_intp dims[2] = {closest_node.path_length, 2};
    PyArrayObject* path = (PyArrayObject*) PyArray_SimpleNew(2, dims, NPY_INT32);
    npy_int32 *iptr, *jptr;
    int idx = closest_node.idx;
    for (npy_intp i = dims[0] - 1; i >= 0; --i) {
        iptr = (npy_int32*) (path->data + i * path->strides[0]);
        jptr = (npy_int32*) (path->data + i * path->strides[0] + path->strides[1]);

        *iptr = idx / w;
        *jptr = idx % w;

        idx = paths[idx];
    }
    PyTuple_SET_ITEM(return_val, 0, Py_False);
    PyTuple_SET_ITEM(return_val, 1, PyArray_Return(path));
  } else { // if a goal is not found, return a path to the node we reached closest to the goal
    PyTuple_SET_ITEM(return_val, 0, Py_False);
    PyTuple_SET_ITEM(return_val, 1, Py_BuildValue("")); // no soln --> return None
  }
  // std::cout << "made it past return" << std::endl;

  delete[] costs;
  delete[] nbrs;
  delete[] paths;

  // std::cout << "made it past return 2" << std::endl;
  return return_val;
}

static PyMethodDef astar_methods[] = {
    {"astar", (PyCFunction)astar, METH_VARARGS, "astar"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef astar_module = {
    PyModuleDef_HEAD_INIT,"astar", NULL, -1, astar_methods
};

PyMODINIT_FUNC PyInit_astar(void) {
  import_array();
  return PyModule_Create(&astar_module);
}
