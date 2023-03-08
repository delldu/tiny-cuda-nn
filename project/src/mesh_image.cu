/************************************************************************************
***
***     Copyright 2023 Dell Du(18588220928@163.com), All Rights Reserved.
***
***     File Author: Dell, 2023年 03月 07日 星期二 18:29:34 CST
***
************************************************************************************/
// #include "../include/mesh_image.h"
#include "../include/meshbox.h"
#include "../include/mesh_common.h"

#include <dirent.h>
#include <iostream>
#include <sys/stat.h> // dir
#include <sys/types.h>


GPUMemory<float> load_image(const std::string& filename, int& width, int& height)
{
    // width * height * RGBA
    float* out = load_stbi(&width, &height, filename.c_str());

    GPUMemory<float> result(width * height * 4);
    result.copy_from_host(out);
    free(out); // release memory of image data

    return result;
}

template <typename T>
__global__ void to_ldr(const uint64_t num_elements, const uint32_t n_channels,
    const uint32_t stride, const T* __restrict__ in, uint8_t* __restrict__ out)
{
    const uint64_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= num_elements)
        return;

    const uint64_t pixel = i / n_channels;
    const uint32_t channel = i - pixel * n_channels;

    out[i] = (uint8_t)(
        powf(fmaxf(fminf(in[pixel * stride + channel], 1.0f), 0.0f), 1.0f / 2.2f) * 255.0f + 0.5f);
}

template <typename T>
void save_image(const T* image, int width, int height, int n_channels, int channel_stride,
    const std::string& filename)
{
    GPUMemory<uint8_t> image_ldr(width * height * n_channels);
    linear_kernel(to_ldr<T>, 0, nullptr, width * height * n_channels, n_channels, channel_stride,
        image, image_ldr.data());

    std::vector<uint8_t> image_ldr_host(width * height * n_channels);
    CUDA_CHECK_THROW(cudaMemcpy(
        image_ldr_host.data(), image_ldr.data(), image_ldr.size(), cudaMemcpyDeviceToHost));

    save_stbi(image_ldr_host.data(), width, height, n_channels, filename.c_str());
}

// template <uint32_t stride>
// __global__ void eval_image(uint32_t n_elements, cudaTextureObject_t texture,
//     float* __restrict__ xs_and_ys, float* __restrict__ result)
// {
//     uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i >= n_elements)
//         return;

//     uint32_t output_idx = i * stride;
//     uint32_t input_idx = i * 2;

//     float4 val = tex2D<float4>(texture, xs_and_ys[input_idx], xs_and_ys[input_idx + 1]);
//     result[output_idx + 0] = val.x;
//     result[output_idx + 1] = val.y;
//     result[output_idx + 2] = val.z;

//     for (uint32_t i = 3; i < stride; ++i) {
//         result[output_idx + i] = 1;
//     }
// }

vector<string> load_files(const string dirname, const string extname)
{
    DIR* dir;
    struct dirent* ent;
    std::vector<string> files;

    dir = opendir(dirname.c_str());
    if (dir == NULL) {
        tlog::error() << "Cannot open directory " << dirname;
        exit(EXIT_FAILURE);
    }

    while ((ent = readdir(dir)) != NULL) {
        char* name = ent->d_name;
        if (strcmp(name, ".") == 0 || strcmp(ent->d_name, "..") == 0)
            continue;

        if (strstr(name, extname.c_str()))
            files.push_back(dirname + "/" + string(name));
    }
    closedir(dir);

    return files;
}

void save_image_as_texture(GPUMemory<float> image, int width, int height, cudaTextureObject_t texture)
{
    // int width, height;
    // GPUMemory<float> image = load_image(argv[1], width, height);

    // Create a cuda texture out of this image.
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypePitch2D;
    resDesc.res.pitch2D.devPtr = image.data();
    resDesc.res.pitch2D.desc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
    resDesc.res.pitch2D.width = width;
    resDesc.res.pitch2D.height = height;
    resDesc.res.pitch2D.pitchInBytes = width * 4 * sizeof(float);

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.normalizedCoords = true;
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.addressMode[2] = cudaAddressModeClamp;

    CUDA_CHECK_THROW(cudaCreateTextureObject(&texture, &resDesc, &texDesc, nullptr));
}

GPUMemory<float> load_image_and_depth(const std::string& image_filename,
    const std::string& depth_filename, int& width, int& height)
{
    // width * height * RGBA
    float* image_out = load_stbi(&width, &height, image_filename.c_str());

    int depth_width, depth_height;
    float* depth_out = load_stbi(&depth_width, &depth_height, depth_filename.c_str());
    if (width != depth_width || height != depth_height) {
        throw std::runtime_error{fmt::format("Image {} size is not same as depth {}", 
            image_filename, depth_filename)};
    }
    float *src = depth_out;
    float *dst = image_out;
    for (int i = 0; i < width * height; i++) {
        if (src[3] < 0.5f) { // Image masked, depth is far ...
            dst[3] = MAX_DEPTH;
        } else { // The feature of depth is more near, more bright
            dst[3] = (1.0f - src[0]) * 256.0f + (1.0f - src[1]) + (1.0f - src[2])/256.0f;
        }
        src += 4; dst += 4;
    }
    free(depth_out); // release memory of depth data

    GPUMemory<float> result(width * height * 4);
    result.copy_from_host(image_out);
    free(image_out); // release memory of image data

    return result;    
}
