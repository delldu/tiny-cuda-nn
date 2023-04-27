/************************************************************************************
***
***     Copyright 2023 Dell Du(18588220928@163.com), All Rights Reserved.
***
***     File Author: Dell, 2023年 03月 07日 星期二 18:29:34 CST
***
************************************************************************************/
#include <dirent.h>
#include <sys/stat.h> // dir
#include <sys/types.h>

#include <stbi/stb_image.h>
#include <stbi/stbi_wrapper.h>
#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/config.h>
using namespace tcnn;

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace std;

#include <Eigen/Dense> // Version 3.4.9, eigen.tgz under dependencies
using namespace Eigen;

#include "../include/meshbox.h"
#include "tinylog.h"

#define MAX_POINTS (5 * 1024 * 1024)

#define MAX_DEPTH 16384.0f
#define MAX_IMAGES 512
#define MEGA_BYTES 1000000

struct Camera {
    // transform matrix = K*[R | T]
    Camera()
        : K(Matrix3f::Identity())
        , R(Matrix3f::Identity())
        , T(Vector3f::Zero())
    {
        update();
    }

    void update()
    {
        KR = K * R;
        O = K * T;
        FWD_norm = R.col(2).normalized();
        R_K_inv = R.inverse() * K.inverse();
        R_inv = R.inverse();

        // C3
        Matrix<float, 3, 4> P;
        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < 3; c++)
                P(r, c) = KR(r, c);
            P(r, 3) = O(r, 0);
        }
        Matrix3f M1 = P(Eigen::placeholders::all, { 1, 2, 3 }); // x
        Matrix3f M2 = P(Eigen::placeholders::all, { 0, 2, 3 }); // y
        Matrix3f M3 = P(Eigen::placeholders::all, { 0, 1, 3 }); // z
        Matrix3f M4 = P(Eigen::placeholders::all, { 0, 1, 2 }); // t

        C3(0) = M1.determinant();
        C3(1) = -M2.determinant();
        C3(2) = M3.determinant();
        float t = -M4.determinant();
        C3(0) = C3(0) / t;
        C3(1) = C3(1) / t;
        C3(2) = C3(2) / t;
    }

    void load(const string filename);
    void dump();

    Matrix3f K;
    Matrix3f R;
    Vector3f T;

    // private for past
    Matrix3f KR;
    Vector3f O;
    Vector3f FWD_norm; // Forward normal
    Matrix3f R_inv; // R_inverse
    Matrix3f R_K_inv; // R_inverse * K_inverse
    Vector3f C3;
};

struct Ray {
    Ray()
        : o(Vector3f::Zero())
        , d(Vector3f::Zero())
    {
    }

    Eigen::Vector3f o;
    Eigen::Vector3f d;
};

struct Point {
    Point()
        : xyz(Vector3f::Zero())
        , rgba(Vector4f::Zero())
    {
    }
    Vector3f xyz;
    Vector4f rgba; // w == valid ? 1.0f, 0.0f
};

// 0) Device
// 1) Camera
// 2) Image
// 3) Points

// 0) Device
/************************************************************************************/
void dump_gpu_memory()
{
    size_t avail, total, used;
    cudaMemGetInfo(&avail, &total);

    used = total - avail;
    tlog::info() << "GPU used " << (float)used / MEGA_BYTES << " M"
                 << ", free " << (float)avail / MEGA_BYTES << " M";
}

bool has_cuda_device()
{
    int i, count = 0;

    cudaGetDeviceCount(&count);
    if (count == 0) {
        tlog::error() << "NO GPU Device";
        return false;
    }

    for (i = 0; i < count; i++) {
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
            if (prop.major >= 1)
                break;
        }
    }
    if (i == count) {
        tlog::error() << "NO GPU supporting CUDA";
        return false;
    }

    tlog::info() << "Running on GPU " << i << " ... ";
    cudaSetDevice(i);
    cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 1024 * 128);

    return true;
}

// 1) Camera
/************************************************************************************/
// xxxx8888
__device__ float l2_distance(const Vector3f& a, const Vector3f& b)
{
    Vector3f c = a - b;
    return sqrtf(c.x() * c.x() + c.y() * c.y() + c.z() * c.z());
}

__device__ float get_disparity(
    const Camera& camera1, const Camera& camera2, const float depth1, const float depth2)
{
    float baseline = l2_distance(camera1.C3, camera2.C3);
    // camera.K(0,0) -- focal length
    // depth/baseline=focal_length/x ==> x = focal_leng * baseline/depth
    // return fabs(camera1.K(0, 0) * baseline / depth1 - camera2.K(0, 0) * baseline / depth2);
    return fabs(baseline / depth1 - baseline / depth2);
}

__device__ void image_to_world(
    const int u, const int v, const float depth,
    const Camera& camera, Vector3f* __restrict__ xyz)
{
    Vector3f pt = Vector3f { depth * u, depth * v, depth } - camera.O;
    *xyz = camera.R_K_inv * pt;
}

__device__ void world_to_image(const Vector3f xyz, const Camera& camera,
    int* __restrict__ u, int* __restrict__ v)
{
    Vector3f temp = camera.K * camera.R * xyz + camera.O;
    // depth = temp.z();
    *u = (int)(temp.x() / (temp.z() + 1.0e-10f));
    *v = (int)(temp.y() / (temp.z() + 1.0e-10f));
}

__host__ __device__ void uv_to_ray(
    const int u, const int v,
    const uint32_t width, const uint32_t height,
    const Camera camera, const float depth,
    Vector3f* __restrict__ endpoint)
{
    Vector3f dir = Vector3f {
        ((float)u - (float)width / 2.0f) / camera.K(0, 0), // focal_length_x
        ((float)v - (float)height / 2.0f) / camera.K(1, 1), // focal_length_y,
        1.0f
    };
    dir = camera.R * dir;
    float cos_theta = dir.dot(camera.FWD_norm) + 1.0e-10f;
    *endpoint = camera.T + depth / cos_theta * dir; // ray.o + (depth/cos_theta) * ray.dir
}

static void read_camera(const string filename, Camera& camera)
{
    int i, j;
    char line[512], *p;
    ifstream fp;

    fp.open(filename.c_str(), ifstream::in);

    for (i = 0; i < 3; i++) {
        if (fp.eof())
            break;

        fp.getline(line, 512);
        for (j = 0, p = strtok(line, " "); p; p = strtok(NULL, " "), j++) {
            if (j >= 0 && j < 3) {
                camera.K(i, j) = (float)atof(p);
            } else if (j >= 3 && j < 6) {
                camera.R(i, j - 3) = (float)atof(p);
            } else if (j >= 6 && j < 7) {
                camera.T(i, j - 6) = (float)atof(p);
            } else {
                ; // comments, skip ...
            }
        }
    }
    camera.update();

    fp.close();
}

void Camera::load(const string filename) { read_camera(filename, *this); }

void Camera::dump()
{
    cout << "Camera:" << endl;
    cout << "K:" << endl
         << this->K << endl;
    cout << "R:" << endl
         << this->R << endl;
    cout << "T:" << endl
         << this->T << endl;
    cout << "KR:" << endl
         << this->K * this->R << endl;
    cout << "R_K_inv:" << endl
         << this->R_K_inv << endl;
}

// 2) Image
/************************************************************************************/
inline void depth_rgb(float depth, uint8_t* R, uint8_t* G, uint8_t* B)
{
    uint32_t rgb = (uint32_t)(depth * 512.0f);
    *R = (rgb & 0xff0000) >> 16;
    *G = (rgb & 0x00ff00) >> 8;
    *B = (rgb & 0xff);
}

inline void rgb_depth(uint8_t R, uint8_t G, uint8_t B, float* depth)
{
    uint32_t rgb = ((uint32_t)R << 16) | ((uint32_t)G << 8) | ((uint32_t)B);
    *depth = (float)rgb / 512.0f;
}

float* load_image_with_depth(const string& image_filename, int& width, int& height)
{
    // width * height * RGBA
    float* image_float_data = load_stbi(&width, &height, image_filename.c_str());
    return image_float_data;
}

float* load_image_and_depth(
    const string& image_filename, const string& depth_filename, int& width, int& height)
{
    int n;
    // width * height * RGBA
#if 0   
    float* image_float_data = load_stbi(&width, &height, image_filename.c_str());
#endif
    uint8_t* image_data = stbi_load(image_filename.c_str(), &width, &height, &n, 0);
    int depth_width, depth_height;
    uint8_t* depth_data = stbi_load(depth_filename.c_str(), &depth_width, &depth_height, &n, 0);
    if (width != depth_width || height != depth_height) {
        tlog::error() << "Image " << image_filename << " size is not same as depth " << depth_filename;
    }

    float* image_float_data = (float*)malloc(width * height * sizeof(float) * 4);
    uint8_t* src1 = image_data;
    uint8_t* src2 = depth_data;
    float* dst = image_float_data;
    for (int i = 0; i < width * height; i++) {
        if (src1[3] < 128) {
            dst[3] = MAX_DEPTH; // Image masked, depth is far ...
        } else {
            dst[0] = (float)src1[0];
            dst[1] = (float)src1[1];
            dst[2] = (float)src1[2];
            rgb_depth(src2[0], src2[1], src2[2], &dst[3]);
        }
        src1 += 4;
        src2 += 4;
        dst += 4;
    }
    free(image_data); // release memory of color data
    free(depth_data); // release memory of depth data

    return image_float_data;
}

// 3) Point
/************************************************************************************/
__global__ void fusion_point_kernel(
    const uint32_t n_images,
    const uint32_t image_k,
    const uint32_t image_width,
    const uint32_t image_height,
    const cudaTextureObject_t* __restrict__ gpu_textures,
    const Camera* __restrict__ gpu_cameras,
    Point* __restrict__ one_image_gpu_points)
{
    uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x >= image_width || y >= image_height)
        return;

    const uint32_t center = y * image_width + x;
    float4 sum_rgba = tex2D<float4>(gpu_textures[image_k], x + 0.5f, y + 0.5f);
    float depth = sum_rgba.w;
    if (depth >= MAX_DEPTH)
        return;

    Vector3f sum_xyz; // save ray end point (world coordinate)
    uv_to_ray(x, y, image_width, image_height, gpu_cameras[image_k], depth, &sum_xyz);
    Vector3f k_xyz = sum_xyz;

#if 1
    one_image_gpu_points[center].xyz = sum_xyz;
    one_image_gpu_points[center].rgba
        = Vector4f { sum_rgba.x, sum_rgba.y, sum_rgba.z, 1.0f }; // mark w == 1.0f
#else
    int match_count = 0;

    Vector3f xyz;
    image_to_world(x, y, depth, gpu_cameras[image_k], &xyz);

    int i_u, i_v;
    Vector3f i_xyz; // Consistent ray end point on other view

    for (uint32_t image_i = 0; image_i < n_images && match_count < 6; image_i++) {
        if (image_i == image_k)
            continue;

        // Project 3d point xyz on camera i
        world_to_image(xyz, gpu_cameras[image_i], &i_u, &i_v);
        if (i_u < 0 || i_u >= image_width || i_v < 0 || i_v >= image_height)
            continue;

        float4 i_rgba = tex2D<float4>(gpu_textures[image_i], i_u + 0.5f, i_v + 0.5f);
        if (i_rgba.w > MAX_DEPTH)
            continue;

        float point_distance = l2_distance(k_xyz, i_xyz);
        float depth_disp = get_disparity(gpu_cameras[image_k], gpu_cameras[image_i], depth, i_rgba.w);
        if (x % 100 == 0 && y % 100 == 0) {
            printf("depth_disp = %.4f, point_distance=%.4f\n", depth_disp, point_distance);
        }

        // check on depth disparity
        if (depth_disp < 0.1f && point_distance < 0.1f) {
            // depth_threshold == 0.25
            printf("!!!!! \n");

            uv_to_ray(i_u, i_v, image_width, image_height, gpu_cameras[image_i], i_rgba.w, &i_xyz);

            sum_xyz.x() = sum_xyz.x() + i_xyz.x();
            sum_xyz.y() = sum_xyz.y() + i_xyz.y();
            sum_xyz.z() = sum_xyz.z() + i_xyz.z();

            sum_rgba.x += i_rgba.x;
            sum_rgba.y += i_rgba.y;
            sum_rgba.z += i_rgba.z;
            // sum_rgba.w += i_rgba.w;

            match_count += 1;
        }
    }

    if (match_count >= 0) {
        // Average normals and points
        float fc = (float)match_count + 1.0f;

        one_image_gpu_points[center].xyz
            = Vector3f { sum_xyz.x() / fc, sum_xyz.y() / fc, sum_xyz.z() / fc };
        one_image_gpu_points[center].rgba
            = Vector4f { sum_rgba.x / fc, sum_rgba.y / fc, sum_rgba.z / fc, 1.0f }; // valid w == 1.0f
    }
#endif
}

vector<string> load_files(const string dirname, const string extname)
{
    DIR* dir;
    struct dirent* ent;
    vector<string> files;

    dir = opendir(dirname.c_str());
    if (dir == NULL) {
        tlog::error() << "Cannot open directory " << dirname;
    } else {
        while ((ent = readdir(dir)) != NULL) {
            char* name = ent->d_name;
            if (strcmp(name, ".") == 0 || strcmp(ent->d_name, "..") == 0)
                continue;

            if (strstr(name, extname.c_str()))
                files.push_back(dirname + "/" + string(name));
        }
        closedir(dir);
    }

    return files;
}

void save_point_cloud(const string& filename, const vector<Point>& pc)
{
    uint32_t n_pc = pc.size();
    if (n_pc >= MAX_POINTS)
        n_pc = MAX_POINTS;

    cout << "Save " << n_pc << "/" << pc.size() << " points to " << filename << " ..." << endl;

    FILE* fp = fopen(filename.c_str(), "wb");

    /*write header */
    fprintf(fp, "ply\n");
    fprintf(fp, "format binary_little_endian 1.0\n");
    fprintf(fp, "element vertex %d\n", n_pc);
    fprintf(fp, "property float x\n");
    fprintf(fp, "property float y\n");
    fprintf(fp, "property float z\n");
    fprintf(fp, "property uchar red\n");
    fprintf(fp, "property uchar green\n");
    fprintf(fp, "property uchar blue\n");
    fprintf(fp, "end_header\n");

    // write data
#pragma omp parallel for
    for (size_t i = 0; i < n_pc; i++) {
        const Point& p = pc[i];

        const char color_r = (uint8_t)(p.rgba.x());
        const char color_g = (uint8_t)(p.rgba.y());
        const char color_b = (uint8_t)(p.rgba.z());

#pragma omp critical
        {
            fwrite(&p.xyz.x(), sizeof(float), 1, fp);
            fwrite(&p.xyz.y(), sizeof(float), 1, fp);
            fwrite(&p.xyz.z(), sizeof(float), 1, fp);
            fwrite(&color_r, sizeof(char), 1, fp);
            fwrite(&color_g, sizeof(char), 1, fp);
            fwrite(&color_b, sizeof(char), 1, fp);
        }
    }
    fclose(fp);
}

int eval_points(char* input_folder)
{
    size_t i, n_filenames;
    char output_folder[1024], file_name[2048];

    if (has_cuda_device()) {
        dump_gpu_memory();
    }

    // Create output for saving points
    sprintf(output_folder, "%s/point", input_folder);
    mkdir(output_folder, 0777);

    vector<string> image_filenames;
    {
        // Loading image files
        snprintf(file_name, sizeof(file_name), "%s/image", input_folder);
        image_filenames = load_files(file_name, ".png");
        if (image_filenames.size() < 1) {
            tlog::error() << "NO images under folder '" << input_folder << "'";
            return -1;
        }

        sort(image_filenames.begin(), image_filenames.end());
        n_filenames = image_filenames.size();
        if (n_filenames >= MAX_IMAGES) {
            tlog::error() << "Too many images (>=" << MAX_IMAGES << ") under folder '" << input_folder << "'";
            return -1;
        }
    }

    vector<string> camera_filenames;
    {
        // Loading camera files
        snprintf(file_name, sizeof(file_name), "%s/camera", input_folder);
        camera_filenames = load_files(file_name, ".txt");
        if (camera_filenames.size() < 1) {
            tlog::error() << "NO camera files under folder '" << input_folder << "'";
            return -1;
        }
        sort(camera_filenames.begin(), camera_filenames.end());
        if (image_filenames.size() != camera_filenames.size()) {
            tlog::error() << "image/camera files DOES NOT match under folder '" << input_folder << "'";
            return -1;
        }
    }

    vector<string> depth_filenames;
    {
        // Loading depth files
        snprintf(file_name, sizeof(file_name), "%s/depth", input_folder);
        depth_filenames = load_files(file_name, ".png");
        // if depth_filnames == 0, suppose image including depth information in A channel
        if (depth_filenames.size() > 0 && depth_filenames.size() != n_filenames) {
            tlog::error() << "image/depth files DOES NOT match under folder '" << input_folder << "'";
            return -1;
        }
        sort(depth_filenames.begin(), depth_filenames.end());
    }

    if (n_filenames >= 100)
        n_filenames = 100; //
    uint32_t image_width, image_height;
    {
        // image/depth files have same size ?
        int x, y, n, ok;
        ok = stbi_info(image_filenames[0].c_str(), &x, &y, &n);
        if (ok != 1) {
            tlog::error() << "image '" << image_filenames[0] << "' pixel size is not valid";
            return -1;
        }
        image_width = x;
        image_height = y;
        for (i = 1; i < n_filenames; i++) {
            ok = stbi_info(image_filenames[i].c_str(), &x, &y, &n);
            if (ok != 1 || x != image_width || y != image_height) {
                tlog::error() << "image pixel size IS NOT same under '" << input_folder << "'";
                return -1;
            }
        }
        for (i = 0; i < n_filenames; i++) {
            ok = stbi_info(depth_filenames[i].c_str(), &x, &y, &n);
            if (ok != 1 || x != image_width || y != image_height) {
                tlog::error() << "image/depth pixel size IS NOT same under '" << input_folder << "'";
                return -1;
            }
        }
    }

    GPUMemory<Camera> gpu_cameras(n_filenames);
    {
        // Loading cameras ...
        auto load_camera_logger = tlog::Logger("Loading cameras ...");
        auto progress = load_camera_logger.progress(n_filenames);
        vector<Camera> cpu_cameras(n_filenames);

        for (i = 0; i < n_filenames; i++) {
            progress.update(i);
            cpu_cameras[i].load(camera_filenames[i]);
        }
        gpu_cameras.copy_from_host(cpu_cameras);
        cpu_cameras.clear();

        load_camera_logger.success("OK !");
    }

    cudaTextureObject_t* gpu_textures;
    CUDA_CHECK_THROW(cudaMalloc(&gpu_textures, n_filenames * sizeof(cudaTextureObject_t)));
    {
        // Loading image/depth and save to textures
        int width, height;
        float* image_float_data;
        cudaTextureObject_t cpu_textures[MAX_IMAGES];

        auto load_texture_logger = tlog::Logger("Loading texture ...");
        auto progress = load_texture_logger.progress(n_filenames);

        for (i = 0; i < n_filenames; i++) {
            progress.update(i);

            if (depth_filenames.size() == 0) {
                image_float_data = load_image_with_depth(image_filenames[i], width, height);
            } else {
                image_float_data = load_image_and_depth(image_filenames[i], depth_filenames[i], width, height);
            }

            cudaArray* cuArray;
            cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
            CUDA_CHECK_THROW(cudaMallocArray(&cuArray, &channelDesc, width, height));
            CUDA_CHECK_THROW(cudaMemcpy2DToArray(cuArray /*dst*/, 0 /*wOffset*/, 0 /*hOffset*/,
                image_float_data /*src*/, width * sizeof(float4) /*spitch bytes*/,
                width * sizeof(float4) /*width bytes*/, height /*rows*/, cudaMemcpyHostToDevice));
            CUDA_CHECK_THROW(cudaDeviceSynchronize());
            free(image_float_data); // release memory of image data
            // cudaFreeArray(cuArray);

            // Specify texture
            struct cudaResourceDesc resDesc;
            memset(&resDesc, 0, sizeof(resDesc));
            resDesc.resType = cudaResourceTypeArray;
            resDesc.res.array.array = cuArray;

            // Specify texture object parameters
            struct cudaTextureDesc texDesc;
            memset(&texDesc, 0, sizeof(texDesc));
            texDesc.addressMode[0] = cudaAddressModeWrap;
            texDesc.addressMode[1] = cudaAddressModeWrap;
            texDesc.filterMode = cudaFilterModeLinear;
            texDesc.readMode = cudaReadModeElementType;
            texDesc.normalizedCoords = 0; // (u, v)

            // Create texture object
            CUDA_CHECK_THROW(cudaCreateTextureObject(&(cpu_textures[i]), &resDesc, &texDesc, nullptr));
            // cudaDestroyTextureObject(&(cpu_textures[i])
        }
        CUDA_CHECK_THROW(cudaMemcpy(gpu_textures, cpu_textures, n_filenames * sizeof(cudaTextureObject_t),
            cudaMemcpyHostToDevice));

        load_texture_logger.success("OK !");
    }
    CUDA_CHECK_THROW(cudaDeviceSynchronize());

    vector<Point> all_cpu_points;
    {
        dump_gpu_memory();
        vector<Point> one_image_cpu_points(image_width * image_height);
        GPUMemory<Point> one_image_gpu_points(image_width * image_height);

        auto fusion_points_logger = tlog::Logger("Fusion points ...");
        auto progress = fusion_points_logger.progress(n_filenames);

        for (i = 0; i < n_filenames; i++) {
            progress.update(i);

            // process one_image_gpu_points
            const dim3 threads = { 32, 32, 1 };
            const dim3 blocks = { div_round_up((unsigned int)image_width, threads.x),
                div_round_up((unsigned int)image_height, threads.y), 1 };

            one_image_gpu_points.memset(0);
            fusion_point_kernel<<<blocks, threads>>>(
                (uint32_t)n_filenames,
                (uint32_t)i,
                image_width,
                image_height,
                gpu_textures,
                gpu_cameras.data(),
                one_image_gpu_points.data());
            CUDA_CHECK_THROW(cudaDeviceSynchronize());

            one_image_gpu_points.copy_to_host(one_image_cpu_points);
            CUDA_CHECK_THROW(cudaDeviceSynchronize());

            // save valid points
            // float b = 1.5f;
            for (size_t j = 0; j < one_image_cpu_points.size(); j++) {
                Point pc = one_image_cpu_points[j];
                // if (pc.xyz.minCoeff() < -b || pc.xyz.maxCoeff() > b || pc.xyz.cwiseAbs().sum() < 0.00001f)
                //     continue;
                if (pc.rgba.w() > 0.5f)
                    all_cpu_points.push_back(pc);
            }
        }
        fusion_points_logger.success("OK !");

        one_image_gpu_points.free_memory();
        one_image_cpu_points.clear();
    }
    CUDA_CHECK_THROW(cudaDeviceSynchronize());
    std::random_shuffle(all_cpu_points.begin(), all_cpu_points.end());

    snprintf(file_name, sizeof(file_name), "%s/pc.ply", output_folder);
    save_point_cloud(file_name, all_cpu_points);
    all_cpu_points.clear();

    return 0;
}
