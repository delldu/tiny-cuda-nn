/************************************************************************************
***
***     Copyright 2023 Dell Du(18588220928@163.com), All Rights Reserved.
***
***     File Author: Dell, 2023年 03月 07日 星期二 18:29:34 CST
***
************************************************************************************/
#include "../include/mesh_common.h"

#define MEGA_SIZE 1000000.0f

float get_gpu_memory()
{
	size_t avail, total, used;
	cudaMemGetInfo(&avail, &total);

	used = total - avail;
    tlog::info() << "GPU memory used " << (float)used / MEGA_SIZE << " M"
                 << ", free " << (float)avail / MEGA_SIZE << " M";

    return (float)avail / MEGA_SIZE;
}

bool has_cuda_device()
{
    int i, count = 0;

    cudaGetDeviceCount(&count);
    if (count == 0)
        throw std::runtime_error{fmt::format("NO GPU Device")};

	for (i = 0; i < count; i++) {
		cudaDeviceProp prop;
		if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
			if (prop.major >= 1)
				break;
		}
	}
	if (i == count)
        throw std::runtime_error{fmt::format("NO GPU supporting CUDA")};

	cudaSetDevice(i);
	cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 1024 * 128);

	return true;
}
