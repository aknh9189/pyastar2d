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
// start:    index of start in flattened grid
// diag_ok:        if true, allows diagonal moves (8-conn.)
// fill_radius:    how far, in cells, to plan out to 
// costs (output): 2d grid of min cost to each cell within radius of start
//                 np.inf for infinte cost to point
//                 np.nan for points outside of radius
static PyObject *djikstra(PyObject *self, PyObject *args) {
  const PyArrayObject* weights_object;
  int h;
  int w;
  int start;
  int fill_radius;
  int diag_ok;

  if (!PyArg_ParseTuple(
        args, "Oiiiii", // i = int, O = object
        &weights_object,
        &h, &w,
        &start, &fill_radius,
        &diag_ok
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
  
  int start_i = start / w;
  int start_j = start % w;

  int max_r2 = fill_radius * fill_radius;

  while (!nodes_to_visit.empty()) {
    // .top() doesn't actually remove the node
    Node cur = nodes_to_visit.top();

    nodes_to_visit.pop();

    int row = cur.idx / w;
    int col = cur.idx % w;

    // if we're too far from the start, don't search the neighbors
    int r2 = (row - start_i) * (row - start_i) + (col - start_j) * (col - start_j);
    if (r2 > max_r2)
      continue;

    // check bounds and find up to eight neighbors: top to bottom, left to right
    nbrs[0] = (diag_ok && row > 0 && col > 0)          ? cur.idx - w - 1   : -1;
    nbrs[1] = (row > 0)                                ? cur.idx - w       : -1;
    nbrs[2] = (diag_ok && row > 0 && col + 1 < w)      ? cur.idx - w + 1   : -1;
    nbrs[3] = (col > 0)                                ? cur.idx - 1       : -1;
    nbrs[4] = (col + 1 < w)                            ? cur.idx + 1       : -1;
    nbrs[5] = (diag_ok && row + 1 < h && col > 0)      ? cur.idx + w - 1   : -1;
    nbrs[6] = (row + 1 < h)                            ? cur.idx + w       : -1;
    nbrs[7] = (diag_ok && row + 1 < h && col + 1 < w ) ? cur.idx + w + 1   : -1;

    for (int i = 0; i < 8; ++i) {
      if (nbrs[i] >= 0) {

        // the sum of the cost so far and the cost of this move
        float new_cost = costs[cur.idx] + weights[nbrs[i]];
        if (new_cost < costs[nbrs[i]]) {
          // estimate the cost to the goal based on legal moves

          // paths with lower expected cost are explored first
          float priority = new_cost;
          nodes_to_visit.push(Node(nbrs[i], priority, cur.path_length + 1));

          costs[nbrs[i]] = new_cost;
          paths[nbrs[i]] = cur.idx;
        }
      }
    }
  }

  PyObject *return_val;

  npy_intp dims[2] = {h, w};
  PyArrayObject* costs_python = (PyArrayObject*) PyArray_SimpleNew(2, dims, NPY_FLOAT32);
  npy_float32 *loc_ptr;
  for (npy_intp i = 0; i <= dims[0] - 1; i++) {
    for (npy_intp j = 0; j <= dims[1] - 1; j++) {
      // std::cout << " setting " << i << ", " << j << std::endl;
      // std::cout << " strides " << costs_python->strides[0] << ", " << costs_python->strides[1] << std::endl;
      // std::cout << " data " << costs_python->data << std::endl;
      // std::cout << " offset " << i * costs_python->strides[0] + j * costs_python->strides[1] << std::endl;
      // std::cout << " costs: " << costs[i * w + j] << std::endl;
      loc_ptr = (npy_float32*) (costs_python->data + i * costs_python->strides[0] + j * costs_python->strides[1]);
      int r2 = (i - start_i) * (i - start_i) + (j - start_j) * (j - start_j);
      if (r2 > max_r2) {
        *loc_ptr = NAN;
      } else {
        *loc_ptr = costs[i * w + j];
      }

    }
  }

  return_val = PyArray_Return(costs_python);
  delete[] costs;
  delete[] nbrs;
  delete[] paths;
  return return_val;
}

static PyMethodDef djikstra_methods[] = {
    {"djikstra", (PyCFunction)djikstra, METH_VARARGS, "djikstra"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef djikstra_module = {
    PyModuleDef_HEAD_INIT,"djikstra", NULL, -1, djikstra_methods
};

PyMODINIT_FUNC PyInit_djikstra(void) {
  import_array();
  return PyModule_Create(&djikstra_module);
}
