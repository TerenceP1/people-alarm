#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"
#include <Windows.h>
#include <mmsystem.h>
#include <thread>
#include <iostream>
#include <CL/cl.h>
#include <CL/cl_gl.h>
#include <CL/cl_ext.h>
#include <CL/cl_platform.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#pragma comment(lib, "Winmm.lib")
using namespace std;

cl_program program;
cl_command_queue queue2;
cl_context context;
size_t local_work_size;

typedef struct
{
    int rows, cols; // # of rows and columns
    float data;     // row by row flattened
} matrix;

class Matrix
{
public:
    float *data;
    bool isLoaded = false;
    bool gpuInit = false;
    // cl_mem buffer; // invalid if isLoaded is false
    cl_mem tBuf;
    // void *mapping;
    int rows, cols;
    int noDelete = 1;

    void load()
    {
        cl_int err;
        /*
        buffer = clCreateBuffer(
            context,
            CL_MEM_READ_WRITE,
            sizeof(float) * rows * cols,
            NULL,
            NULL);
        cl_int err;
        err = clEnqueueWriteBuffer(
            queue2,
            buffer,
            CL_TRUE,
            0,
            sizeof(float) * rows * cols,
            data,
            NULL,
            NULL,
            NULL);
        cout << "w1ld-err" << err << endl;
        cout << "Copied ";
        for (int i = 0; i < rows * cols; i++)
        {
            cout << data[i] << ' ';
        }
        cout << endl;
        */
        matrix *tmp = (matrix *)(new char[sizeof(matrix) + sizeof(float) * (rows * cols - 1)]);
        tmp->rows = rows;
        tmp->cols = cols;
        /*tmp.data = clEnqueueMapBuffer(
            queue2,
            buffer,
            CL_TRUE,
            CL_MAP_READ | CL_MAP_WRITE,
            0,
            sizeof(float) * rows * cols,
            NULL,
            NULL,
            &evt,
            &err);
            */
        clFlush(queue2);
        clFinish(queue2);
        // cout << "Mapping error: " << err << endl;
        copy(data, data + rows * cols, &(tmp->data));
        cout << "Copy round 2\n";
        // mapping = tmp.data;
        // cout << "Mapping[0]=" << ((float *)mapping)[0] << endl;
        tBuf = clCreateBuffer(
            context,
            CL_MEM_READ_WRITE,
            sizeof(matrix) + sizeof(float) * (rows * cols - 1),
            NULL,
            NULL);
        err = clEnqueueWriteBuffer(
            queue2,
            tBuf,
            CL_TRUE,
            0,
            sizeof(matrix) + sizeof(float) * (rows * cols - 1),
            tmp,
            NULL,
            NULL,
            NULL);
        cout << "w2ld-err" << err << endl;
        // cout << "copiedbuf: " << (int)(tmp.data) << endl;
        // cout << "buf: " << (int)(buffer) << endl;
        cout << "sizeof(cl_mem)" << sizeof(cl_mem) << endl;
        clFlush(queue2);
        clFinish(queue2);
        isLoaded = true;
    }

    Matrix(int r, int c, bool gpuLoad)
    {
        rows = r;
        cols = c;
        if (gpuLoad)
        {

            /*
        buffer = clCreateBuffer(
            context,
            CL_MEM_READ_WRITE,
            sizeof(float) * rows * cols,
            NULL,
            NULL);
        cl_int err;
        err = clEnqueueWriteBuffer(
            queue2,
            buffer,
            CL_TRUE,
            0,
            sizeof(float) * rows * cols,
            data,
            NULL,
            NULL,
            NULL);
        cout << "w1ld-err" << err << endl;
        cout << "Copied ";
        for (int i = 0; i < rows * cols; i++)
        {
            cout << data[i] << ' ';
        }
        cout << endl;
        */
            matrix *tmp = (matrix *)(new char[sizeof(matrix) + sizeof(float) * (rows * cols - 1)]);
            tmp->rows = rows;
            tmp->cols = cols;
            /*tmp.data = clEnqueueMapBuffer(
                queue2,
                buffer,
                CL_TRUE,
                CL_MAP_READ | CL_MAP_WRITE,
                0,
                sizeof(float) * rows * cols,
                NULL,
                NULL,
                &evt,
                &err);
                */
            clFlush(queue2);
            clFinish(queue2);
            // cout << "Mapping error: " << err << endl;
            cout << "Copy round 2\n";
            // mapping = tmp.data;
            // cout << "Mapping[0]=" << ((float *)mapping)[0] << endl;
            tBuf = clCreateBuffer(
                context,
                CL_MEM_READ_WRITE,
                sizeof(matrix) + sizeof(float) * (rows * cols - 1),
                NULL,
                NULL);
            cl_int err;
            err = clEnqueueWriteBuffer(
                queue2,
                tBuf,
                CL_TRUE,
                0,
                sizeof(matrix) + sizeof(float) * (rows * cols - 1),
                tmp,
                NULL,
                NULL,
                NULL);
            clFlush(queue2);
            clFinish(queue2);
            isLoaded = true;
            gpuInit = true;
        }
        data = new float[r * c];
    }

    void gpuPull()
    {
        data = new float[rows * cols];
        cout << "GPUPULL CODE: "
             << clEnqueueReadBuffer(
                    queue2,
                    tBuf,
                    CL_TRUE,
                    sizeof(int) * 2,
                    sizeof(float) * rows * cols,
                    data,
                    NULL,
                    NULL,
                    NULL)
             << endl;
        clFlush(queue2);
        clFinish(queue2);
        gpuInit = false;
    }

    void unload()
    {
        clReleaseMemObject(tBuf);
        isLoaded = false;
    }

    ~Matrix()
    {
        noDelete--;
        cout << "Destructor call!\n";
        if (noDelete > 0)
        {
            cout << "Destruction blocked!\n";
            return;
        }
        if (!gpuInit)
        {
            cout << "Run delete data\n";
            delete[] data;
        }
        if (isLoaded)
        {
            cout << "I shall unload!!\n";
            unload();
        }
        cout << "Destructor call done!\n";
    }

    Matrix operator*(Matrix &a)
    {
        if (rows != a.cols)
        {
            cerr << "MULTIPLY FAILED BECAUSE OF DIMENSION MISMATCH!!!\n";
            exit(1);
        }
        if (!isLoaded)
        {
            load();
        }
        if (!(a.isLoaded))
        {
            a.load();
        }
        Matrix res(a.rows, cols, true);
        cl_kernel kernel = clCreateKernel(
            program,
            "mlt",
            NULL);
        cl_int err;
        err = clSetKernelArg(
            kernel,
            0,
            sizeof(cl_mem),
            &tBuf);
        cout << "arg0err" << err << endl;
        err = clSetKernelArg(
            kernel,
            1,
            sizeof(cl_mem),
            &(a.tBuf));
        cout << "arg1err" << err << endl;
        err = clSetKernelArg(
            kernel,
            2,
            sizeof(cl_mem),
            &(res.tBuf));
        cout << "arg2err" << err << endl;
        /*cout << "Mapping check: " << (int)mapping << endl;
        err = clSetKernelArg(
            kernel,
            3,
            sizeof(void *),
            &(mapping));
        cout << "arg3err" << err << endl;*/
        size_t wSz = a.rows * cols;
        cout << "kernel!: "
             << clEnqueueNDRangeKernel(queue2,
                                       kernel,
                                       1, NULL,
                                       &wSz, NULL,
                                       0, NULL, NULL)
             << endl;
        clFlush(queue2);
        clFinish(queue2);
        clReleaseKernel(kernel);
        cout << "PLS no destructor\n";
        /*Matrix fr = *res;
        void *resDumper = res;
        delete[] resDumper;
        fr.noDelete++;
        return fr;*/
        res.noDelete++;
        return res;
    }

    Matrix operator+(Matrix &a)
    {
        if (!((rows == a.rows) && (cols == a.cols)))
        {
            cerr << "ADD FAILED BECAUSE OF DIMENSION MISMATCH!!!\n";
            exit(1);
        }
        if (!isLoaded)
        {
            load();
        }
        if (!(a.isLoaded))
        {
            a.load();
        }
        Matrix res(rows, cols, true);
        cl_kernel kernel = clCreateKernel(
            program,
            "add",
            NULL);
        clSetKernelArg(
            kernel,
            0,
            sizeof(cl_mem),
            &tBuf);
        clSetKernelArg(
            kernel,
            1,
            sizeof(cl_mem),
            &(a.tBuf));
        clSetKernelArg(
            kernel,
            2,
            sizeof(cl_mem),
            &(res.tBuf));
        size_t wSz = a.rows * cols;
        clEnqueueNDRangeKernel(
            queue2,
            kernel,
            1,
            NULL,
            &wSz,
            NULL,
            0,
            NULL,
            NULL);
        clFlush(queue2);
        clFinish(queue2);
        clReleaseKernel(kernel);
        cout << "Pls no destructor here." << endl;
        res.noDelete++;
        return res;
    }

    Matrix operator-(Matrix &a)
    {
        if (!((rows == a.rows) && (cols == a.cols)))
        {
            cerr << "SUB FAILED BECAUSE OF DIMENSION MISMATCH!!!\n";
            exit(1);
        }
        if (!isLoaded)
        {
            load();
        }
        if (!(a.isLoaded))
        {
            a.load();
        }
        Matrix res(rows, cols, true);
        cl_kernel kernel = clCreateKernel(
            program,
            "sub",
            NULL);
        clSetKernelArg(
            kernel,
            0,
            sizeof(cl_mem),
            &tBuf);
        clSetKernelArg(
            kernel,
            1,
            sizeof(cl_mem),
            &(a.tBuf));
        clSetKernelArg(
            kernel,
            2,
            sizeof(cl_mem),
            &(res.tBuf));
        size_t wSz = a.rows * cols;
        clEnqueueNDRangeKernel(
            queue2,
            kernel,
            1,
            NULL,
            &wSz,
            NULL,
            0,
            NULL,
            NULL);
        clFlush(queue2);
        clFinish(queue2);
        clReleaseKernel(kernel);
        cout << "Pls no destructor here." << endl;
        res.noDelete++;
        return res;
    }

    Matrix trans()
    {
        if (!isLoaded)
        {
            load();
        }
        Matrix res(cols, rows, true);
        cl_kernel kernel = clCreateKernel(
            program,
            "trans",
            NULL);
        clSetKernelArg(
            kernel,
            0,
            sizeof(cl_mem),
            &tBuf);
        clSetKernelArg(
            kernel,
            1,
            sizeof(cl_mem),
            &(res.tBuf));
        size_t wSz = rows * cols;
        clEnqueueNDRangeKernel(
            queue2,
            kernel,
            1,
            NULL,
            &wSz,
            NULL,
            0,
            NULL,
            NULL);
        clFlush(queue2);
        clFinish(queue2);
        clReleaseKernel(kernel);
        cout << "Pls no destructor here." << endl;
        res.noDelete++;
        return res;
    }

    Matrix relu()
    {
        if (!isLoaded)
        {
            load();
        }
        Matrix res(rows, cols, true);
        cl_kernel kernel = clCreateKernel(
            program,
            "relu",
            NULL);
        clSetKernelArg(
            kernel,
            0,
            sizeof(cl_mem),
            &tBuf);
        clSetKernelArg(
            kernel,
            1,
            sizeof(cl_mem),
            &(res.tBuf));
        size_t wSz = rows * cols;
        clEnqueueNDRangeKernel(
            queue2,
            kernel,
            1,
            NULL,
            &wSz,
            NULL,
            0,
            NULL,
            NULL);
        clFlush(queue2);
        clFinish(queue2);
        clReleaseKernel(kernel);
        cout << "Pls no destructor here." << endl;
        res.noDelete++;
        return res;
    }
};

string slurp(string nm)
{
    ifstream in(nm);
    stringstream a;
    a << in.rdbuf();
    return a.str();
}

int main(int argc, char **argv)
{
    cout << "test\n";
    // Get a context

    cl_platform_id platform;
    cl_device_id device;
    clGetPlatformIDs(1, &platform, nullptr);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    char *name; // CL_DEVICE_NAME
    size_t len;
    char *vendor; // CL_DEVICE_VENDOR

    clGetDeviceInfo(
        device,
        CL_DEVICE_NAME,
        NULL,
        NULL,
        &len);
    cout << "name length: " << len << endl;
    name = new char[len];
    clGetDeviceInfo(
        device,
        CL_DEVICE_NAME,
        len,
        name,
        NULL);
    string nstr(name, len);
    cout << "name: " << nstr << endl;
    clGetDeviceInfo(
        device,
        CL_DEVICE_NAME,
        NULL,
        NULL,
        &len);
    cout << "vendor length: " << len << endl;
    vendor = new char[len];
    clGetDeviceInfo(
        device,
        CL_DEVICE_NAME,
        len,
        vendor,
        NULL);
    string nstr2(vendor, len);
    cout << "vendor: " << nstr2 << endl;
    char *version;
    clGetDeviceInfo(
        device,
        CL_DEVICE_VERSION,
        NULL,
        NULL,
        &len);
    cout << "version length: " << len << endl;
    version = new char[len];
    clGetDeviceInfo(
        device,
        CL_DEVICE_VERSION,
        len,
        version,
        NULL);
    string nstr3(version, len);
    cout << "version: " << nstr3 << endl;
    cl_uint max_dims;
    clGetDeviceInfo(
        device,
        CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
        sizeof(cl_uint),
        &max_dims,
        NULL);
    cout << "Maximum work item dimensions: " << max_dims << endl;
    size_t *max_work_items = new size_t[max_dims];
    clGetDeviceInfo(
        device,
        CL_DEVICE_MAX_WORK_ITEM_SIZES,
        sizeof(size_t) * max_dims,
        max_work_items,
        NULL);
    cout << "Max work items: ";
    for (int i = 0; i < max_dims; i++)
    {
        cout << max_work_items[i] << " ";
    }
    cout << endl;
    local_work_size = max_work_items[0];
    string code = slurp("kernel.cl");
    context = clCreateContext(
        NULL,
        1,
        &device,
        NULL,
        NULL,
        NULL);
    // Make program

    queue2 = clCreateCommandQueueWithProperties(
        context,
        device,
        NULL,
        NULL);
    const char *prg = code.c_str();
    const size_t len2 = code.length();
    program = clCreateProgramWithSource(
        context,
        1,
        &prg,
        &len2,
        NULL);
    clBuildProgram(
        program,
        1,
        &device,
        NULL,
        NULL,
        NULL);
    cout << "Program built.\n";
    size_t logL;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logL);
    char *blog = new char[logL];
    cout << "Collecting log..." << endl;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logL, blog, NULL);
    cout << "Log:\n"
         << blog << endl;
    cout << "Program created!!!!!!!\n";

    cout << "Let's test!\n";
    bool blah = true;
    if (blah)
    {
        Matrix a(2, 2, false);
        Matrix b(2, 2, false);
        a.data[0] = -1;
        a.data[1] = 2;
        a.data[2] = 3;
        a.data[3] = 4;
        b.data[0] = 5;
        b.data[1] = 6;
        b.data[2] = 7;
        b.data[3] = 8;
        a.load();
        b.load();
        cout << "Loaded!" << endl;
        Matrix c = a.relu();
        c.gpuPull();
        cout << "Result: ";
        for (int i = 0; i < 4; i++)
        {
            cout << c.data[i] << " ";
        }
        cout << endl
             << "WOOOOOOOO PASSSSSSSS!!!\n";
        cout << "Only destroy here!!!\n";
    }
    clReleaseProgram(program);
    clReleaseCommandQueue(queue2);
    clReleaseContext(context);
    cout << "All done -- end of code!\n";
    return 0;
}
