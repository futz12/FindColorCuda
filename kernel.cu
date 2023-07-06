
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cmath>
#include <ctime>

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

struct Color_BGR
{
    int B, G, R;
};

struct Color_Lab
{
    float L, a, b;
};

Color_Lab BGR2Lab(Color_BGR x)
{
#define gamma(x) (((x) > 0.04045) ? std::pow(((x)+0.055f) / 1.055f, 2.4f) : ((x) / 12.92));

    const float param_13 = 1.0f / 3.0f;
    const float param_16116 = 16.0f / 116.0f;
    const float Xn = 0.950456f;
    const float Yn = 1.0f;
    const float Zn = 1.088754f;


    float RR = gamma(x.R / 255.0);
    float GG = gamma(x.G / 255.0);
    float BB = gamma(x.B / 255.0);

    float X, Y, Z, fX, fY, fZ;

    X = 0.4124564f * RR + 0.3575761f * GG + 0.1804375f * BB;
    Y = 0.2126729f * RR + 0.7151522f * GG + 0.0721750f * BB;
    Z = 0.0193339f * RR + 0.1191920f * GG + 0.9503041f * BB;

    X /= (Xn);
    Y /= (Yn);
    Z /= (Zn);

    if (Y > 0.008856f)
        fY = std::pow(Y, param_13);
    else
        fY = 7.787f * Y + param_16116;

    if (X > 0.008856f)
        fX = std::pow(X, param_13);
    else
        fX = 7.787f * X + param_16116;

    if (Z > 0.008856)
        fZ = std::pow(Z, param_13);
    else
        fZ = 7.787f * Z + param_16116;

    float L, a, b;

    L = 116.0f * fY - 16.0f;
    L = L > 0.0f ? L : 0.0f;
    a = 500.0f * (fX - fY);
    b = 200.0f * (fY - fZ);

    return { L,a,b };
}

cudaError_t FindColorCuda(Color_BGR *src, float *ret,Color_Lab target,unsigned int size);

__global__ void FindColorCudaKernel(Color_BGR *src, float* ret, Color_Lab target)
{
    int i = blockIdx.x * 256 + threadIdx.x;

#define gamma(x) (((x) > 0.04045) ? pow(((x)+0.055f) / 1.055f, 2.4f) : ((x) / 12.92));

    const float param_13 = 1.0f / 3.0f;
    const float param_16116 = 16.0f / 116.0f;
    const float Xn = 0.950456f;
    const float Yn = 1.0f;
    const float Zn = 1.088754f;

    float RR = gamma(src[i].R / 255.0);
    float GG = gamma(src[i].G / 255.0);
    float BB = gamma(src[i].B / 255.0);

    float X, Y, Z, fX, fY, fZ;

    X = 0.4124564f * RR + 0.3575761f * GG + 0.1804375f * BB;
    Y = 0.2126729f * RR + 0.7151522f * GG + 0.0721750f * BB;
    Z = 0.0193339f * RR + 0.1191920f * GG + 0.9503041f * BB;

    X /= (Xn);
    Y /= (Yn);
    Z /= (Zn);

    if (Y > 0.008856f)
        fY = pow(Y, param_13);
    else
        fY = 7.787f * Y + param_16116;

    if (X > 0.008856f)
        fX = pow(X, param_13);
    else
        fX = 7.787f * X + param_16116;

    if (Z > 0.008856)
        fZ = pow(Z, param_13);
    else
        fZ = 7.787f * Z + param_16116;

    float L, a, b;

    L = 116.0f * fY - 16.0f;
    L = L > 0.0f ? L : 0.0f;
    a = 500.0f * (fX - fY);
    b = 200.0f * (fY - fZ);

    ret[i] = sqrt((L - target.L) * (L - target.L) + (a - target.a) * (a - target.a) + (b - target.b) * (b - target.b));
}

Color_BGR src_mat[1024 * 1024];
float ret_mat[1024 * 1024];


int main()
{
    for (int i = 0; i < 1024 * 1024; i++)
    {
        src_mat[i] = { std::rand() % 256,std::rand() % 256, std::rand() % 256 };
    }
    //Pre Run for Best Speed
    cudaError_t cudaStatus = FindColorCuda(src_mat, ret_mat, BGR2Lab({ 190,35,41 }), 1024 * 1024);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "FindColorCuda failed!");
        return 1;
    }

    int st = clock();
    // Add vectors in parallel.
    cudaStatus = FindColorCuda(src_mat, ret_mat, BGR2Lab({190,35,41}), 1024 * 1024);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "FindColorCuda failed!");
        return 1;
    }
    printf("Cost: %d\n", clock() - st);

    int count = 0;
    for (int i = 0; i < 1024*1024 ; i++)
    {
        if (ret_mat[i] < 2)
            count++;
    }
    printf("%d", count);
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

//Helper
cudaError_t FindColorCuda(Color_BGR* src, float* ret, Color_Lab target, unsigned int size)
{
    Color_BGR* dev_src = nullptr;
    float* dev_ret = nullptr;
    cudaError cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_src, size * sizeof(Color_BGR));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_ret, size * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_src, src, size * sizeof(Color_BGR), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    FindColorCudaKernel <<<size/256, 256 >>> (dev_src,dev_ret,target);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "FindColorCuda launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    cudaStatus = cudaMemcpy( ret, dev_ret, size * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_ret);
    cudaFree(dev_src);

    return cudaStatus;
}