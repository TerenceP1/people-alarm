# Documentation

## main.cpp

### class Matrix

This class is where all the matrix operations are implemented. There are several matrix operations.

One thing at first that may seem confusing is how I'm passing the Matrixes in. A cl_mem is an opaque handle rather than a pointer and `clSetKernelArg`
