#include <queue>
#include <unordered_set>
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

    Node(int i, float c) : idx(i), cost(c) {}
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

int find_path_length(int* paths, int start, int last_idx) {
  int path_length = 0;
  int cur = last_idx;
  while (cur != start) {
    cur = paths[cur];
    path_length++;
  }
  return path_length;
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
  bool* in_open = new bool[h * w];
  bool reached_goal = false;

  Node start_node(start, 0.);

  float* costs = new float[h * w];
  for (int i = 0; i < h * w; ++i)
    costs[i] = INF;
  costs[start] = 0.;

  std::priority_queue<Node> nodes_to_visit;
  nodes_to_visit.push(start_node);
  in_open[start] = true;

  int* nbrs = new int[8];
  
  int goal_i = goal / w;
  int goal_j = goal % w;
  int start_i = start / w;
  int start_j = start % w;

  heuristic_ptr heuristic_func = select_heuristic(heuristic_override);

  Node closest_node(start, 0.0);
  float closest_node_h = INF;

  // std::cout << "in astar" << std::endl;

  while (!nodes_to_visit.empty()) {
    // .top() doesn't actually remove the node
    Node cur = nodes_to_visit.top();

    if (cur.idx == goal) {
      reached_goal = true;
      break;
    }

    nodes_to_visit.pop();
    in_open[cur.idx] = false;

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
        // check if this node is in the closed list
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
            closest_node = Node(nbrs[i], heuristic_cost);
          }

          // paths with lower expected cost are explored first
          float priority = new_cost + heuristic_cost;

          costs[nbrs[i]] = new_cost;
          paths[nbrs[i]] = cur.idx;
          if (!in_open[nbrs[i]]) {
            nodes_to_visit.push(Node(nbrs[i], priority));
            in_open[nbrs[i]] = true;
          }
          // if (path_lengths[cur.idx] - path_lengths[nbrs[i]] != -1) {
          //   std::cout << "we've found the broken part" << std::endl;
          //   std::cout << "cur.idx is " << cur.idx << std::endl;
          //   std::cout << "cur.path_length is " << cur.path_length << std::endl;
          //   std::cout << "nbrs[i] is " << nbrs[i] << std::endl;
          //   std::cout << "path_lengths[cur.idx] is " << path_lengths[cur.idx] << std::endl;
          //   std::cout << "path_lengths[nbrs[i]] is " << path_lengths[nbrs[i]] << std::endl;
          // }
        }
      }
    }
  }
  
  PyObject *return_val;
  return_val = PyTuple_New(2);
  if (reached_goal) {
    int path_length = find_path_length(paths, start, goal);
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
    PyTuple_SET_ITEM(return_val, 0, Py_True);
    PyTuple_SET_ITEM(return_val, 1, PyArray_Return(path));
  }
  else { // if a goal is not found, return a path to the node we reached closest to the goal
    int closest_node_path_length = find_path_length(paths, start, closest_node.idx);
    npy_intp dims[2] = {closest_node_path_length, 2};
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
  } 

  delete[] costs;
  delete[] nbrs;
  delete[] paths;
  delete[] in_open;

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
