# Documentation

## main.cpp

### class Matrix

This class is where all the matrix operations are implemented. There are several matrix operations.

One thing at first that may seem confusing is how I'm passing the Matrixes in. A cl_mem is an opaque handle rather than a pointer and `clSetKernelArg` can detect it and turn it into a pointer. That is the only way so I can't put pointers in structs so `matrix` is a sort of memory header with the acual data in memory directly after this header.
