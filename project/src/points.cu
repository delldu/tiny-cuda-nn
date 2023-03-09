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

#include "../include/meshbox.h"
#include "tinylogger.h"

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
    cout << "GPU memory used " << (float)used / MEGA_BYTES << " M"
         << ", free " << (float)avail / MEGA_BYTES << " M" << endl;
}

bool has_cuda_device()
{
    int i, count = 0;

    cudaGetDeviceCount(&count);
    if (count == 0)
        throw std::runtime_error{ fmt::format("NO GPU Device") };

    for (i = 0; i < count; i++) {
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
            if (prop.major >= 1)
                break;
        }
    }
    if (i == count)
        throw std::runtime_error{ fmt::format("NO GPU supporting CUDA") };

    std::cout << "Running on GPU " << i << " ... " << std::endl;
    cudaSetDevice(i);
    cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 1024 * 128);

    return true;
}

// 1) Camera
/************************************************************************************/
__device__ float get_disparity(
    const Camera& camera1, const Camera& camera2, const float depth1, const float depth2)
{
    Vector3f a = camera2.O - camera1.O;
    float baseline = sqrtf(a.x() * a.x() + a.y() * a.y() + a.z() * a.z());
    // camera.K(0,0) -- focal length
    // depth/baseline=focal_length/? ==> ? = focal_leng * baseline/depth
    return camera1.K(0, 0) * baseline / depth1 - camera2.K(0, 0) * baseline / depth2;
}

__device__ void image_to_world(const float u, const float v, const float depth,
    const Camera camera, Vector3f* __restrict__ xyz)
{
    Vector3f pt
        = Vector3f{ depth * u - camera.KT.x(), depth * v - camera.KT.y(), depth - camera.KT.z() };
    *xyz = camera.R_K_inv * pt;
}

__device__ void world_to_image(const Vector3f xyz, const Camera camera,
    float* __restrict__ u, float* __restrict__ v)
{
    Vector3f temp = camera.K * camera.R * xyz + camera.K * camera.T;
    // // depth = temp.z();
    *u = temp.x() / (temp.z() + 1e-10);
    *v = temp.y() / (temp.z() + 1e-10);
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
    cout << "K:" << endl << this->K << endl;
    cout << "R:" << endl << this->R << endl;
    cout << "T:" << endl << this->T << endl;
    cout << "O:" << endl << this->O << endl;
    cout << "KR:" << endl << this->K * this->R << endl;
    cout << "KT:" << endl << this->K * this->T << endl;
    cout << "R_K_inv:" << endl << this->R_K_inv << endl;
}

// 2) Image
void depth_rgb(float depth, uint8_t *R, uint8_t *G, uint8_t *B)
{
    if (depth < MIN_DEPTH)
        depth = MIN_DEPTH;
    if (depth > MAX_DEPTH)
        depth = MAX_DEPTH;

    uint32_t rgb = (uint32_t)((MAX_DEPTH - depth) * 256.0f);
    *R = (rgb & 0xff0000) >> 16;
    *G = (rgb & 0x00ff00) >> 8;
    *B = (rgb & 0xff);
}

void rgb_depth(uint8_t R, uint8_t G, uint8_t B, float *depth)
{
    uint32_t rgb = (R << 16) | (G << 8) | (B);
    *depth = MAX_DEPTH - (float)rgb / 256.0f;
}

/************************************************************************************/
float* load_image(const string& filename, int& width, int& height)
{
    // width * height * RGBA
    return load_stbi(&width, &height, filename.c_str());
}


float* load_image_and_depth(
    const string& image_filename, const string& depth_filename, int& width, int& height)
{
    // width * height * RGBA
    float* image_data = load_image(image_filename, width, height);

    int depth_width, depth_height;
    float* depth_data = load_image(depth_filename, depth_width, depth_height);
    if (width != depth_width || height != depth_height) {
        throw std::runtime_error{ fmt::format(
            "Image {} size is not same as depth {}", image_filename, depth_filename) };
    }
    float* src = depth_data;
    float* dst = image_data;
    float temp_depth;
    for (int i = 0; i < width * height; i++) {
        if (src[3] < 0.5f) { // Image masked, depth is far ...
            dst[3] = MAX_DEPTH;
        } else { // The feature of depth is more near, more bright
            rgb_depth((uint8_t)(src[0] * 255.0f), (uint8_t)(src[1] * 255.0f), 
                (uint8_t)(src[2] * 255.0f), &temp_depth);
            dst[3] = temp_depth;
        }
        src += 4;
        dst += 4;
    }
    free(depth_data); // release memory of depth data

    return image_data;
}

// 3) Point
/************************************************************************************/
__global__ void fusion_point_kernel(
    const uint32_t n_images, 
    const uint32_t image_k,
    const uint32_t image_width,
    const uint32_t image_height,
    // const cudaTextureObject_t* __restrict__ gpu_textures, 
    // const Camera* __restrict__ gpu_cameras,
    // Point* __restrict__ one_image_gpu_points)
    cudaTextureObject_t* gpu_textures, 
    Camera* gpu_cameras,
    Point* one_image_gpu_points)
{
    uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x >= image_width || y >= image_height)
        return;

    const uint32_t center = y * image_width + x;
    float4 sum_rgba = tex2D<float4>(gpu_textures[image_k], x + 0.5f, y + 0.5f);

    if (x % 100 == 0 && y % 100 == 0) {
        printf("image_k = %d, x = %d, y = %d, image_size = (%d, %d), n_images = %d, rgba: (%.4f, %.4f. %.4f. %.4f)\n", 
            image_k, x, y, image_width, image_height, n_images, sum_rgba.x, sum_rgba.y, sum_rgba.z, sum_rgba.w);
    }

    float depth = sum_rgba.w;
    if (depth < MIN_DEPTH || depth >= MAX_DEPTH)
        return;

    Vector3f xyz;
    image_to_world((float)x, (float)y, depth, gpu_cameras[image_k], &xyz);
    int count = 0;
    Vector3f sum_xyz = xyz;
    for (uint32_t image_i = 0; image_i < n_images && count < 6; image_i++) {
        if (image_i == image_k)
            continue;

        // Project 3d point xyz on camera i
        float i_u, i_v;
        world_to_image(xyz, gpu_cameras[image_i], &i_u, &i_v);

        // Boundary check
        if ((int)i_u < 0 || (int)i_u >= image_width || (int)i_v < 0
            || (int)i_v >= image_height)
            continue;

        float4 i_rgba = tex2D<float4>(gpu_textures[image_i], i_u + 0.5f, i_v + 0.5f);
        if (i_rgba.w < MIN_DEPTH || i_rgba.w > MAX_DEPTH)
            continue;

        float depth_disp = get_disparity(gpu_cameras[image_k], gpu_cameras[image_i], depth, i_rgba.w);

        // check on depth disparity
        if (fabsf(depth_disp) < 0.05f) {
            // depth_threshold == 0.25
            Vector3f i_xyz; // 3d point of consistent point on other view
            image_to_world(i_v, i_u, i_rgba.w, gpu_cameras[image_i], &i_xyz);

            sum_xyz = Vector3f{ sum_xyz.x() + i_xyz.x(), sum_xyz.y() + i_xyz.y(),
                sum_xyz.z() + i_xyz.z() };
            sum_rgba.x += i_rgba.x;
            sum_rgba.y += i_rgba.y;
            sum_rgba.z += i_rgba.z;
            sum_rgba.w += i_rgba.w;

            count++;
        }
    }
        

    if (count >= 3) {
        // Average normals and points
        float fc = (float)count + 1.0f;
        sum_xyz = Vector3f{ sum_xyz.x() / fc, sum_xyz.y() / fc, sum_xyz.z() / fc };
        sum_rgba.x /= fc;
        sum_rgba.y /= fc;
        sum_rgba.z /= fc;
        sum_rgba.w /= fc;

        one_image_gpu_points[center].xyzw
            = Vector4f{ sum_xyz.x(), sum_xyz.y(), sum_xyz.z(), 1.0f }; // mark w == 1.0f
        one_image_gpu_points[center].rgba
            = Vector4f{ sum_rgba.x, sum_rgba.y, sum_rgba.z, sum_rgba.w };

        // one_image_gpu_points[center].xyzw.x() = sum_xyz.x();
        // one_image_gpu_points[center].xyzw.y() = sum_xyz.y();
        // one_image_gpu_points[center].xyzw.z() = sum_xyz.z();
        // one_image_gpu_points[center].xyzw.w() = 1.0f;

        // one_image_gpu_points[center].rgba.x() = sum_rgba.x;
        // one_image_gpu_points[center].rgba.y() = sum_rgba.y;
        // one_image_gpu_points[center].rgba.z() = sum_rgba.z;
        // one_image_gpu_points[center].rgba.w() = sum_rgba.w;
    }
}

vector<string> load_files(const string dirname, const string extname)
{
    DIR* dir;
    struct dirent* ent;
    vector<string> files;

    dir = opendir(dirname.c_str());
    if (dir == NULL) {
        throw std::runtime_error{fmt::format("Cannot open directory {}.", dirname)};
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

void save_point_cloud(const string& filename, const vector<Point>& pc)
{
    uint32_t n_pc = pc.size();
    if (n_pc >= 1*1024*1024) {
        n_pc = 1*1024*1024;
        // std::random_shuffle(pc.begin(), pc.end());
    }

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

        const char color_r = (int)(p.rgba.x() * 255.0);
        const char color_g = (int)(p.rgba.y() * 255.0);
        const char color_b = (int)(p.rgba.z() * 255.0);

#pragma omp critical
        {
            fwrite(&p.xyzw.x(), sizeof(float), 1, fp);
            fwrite(&p.xyzw.y(), sizeof(float), 1, fp);
            fwrite(&p.xyzw.z(), sizeof(float), 1, fp);
            fwrite(&color_r, sizeof(char), 1, fp);
            fwrite(&color_g, sizeof(char), 1, fp);
            fwrite(&color_b, sizeof(char), 1, fp);
        }
    }
    fclose(fp);
}

int eval_points(char *input_folder)
{
    size_t i, n_filenames;
    char output_folder[1024], file_name[2048];
    // cudaStream_t fusion_stream;

    if (has_cuda_device()) {
        dump_gpu_memory();
    }
    // CUDA_CHECK_THROW(cudaStreamCreate(&fusion_stream));
    {
        // Create output for saving points
        sprintf(output_folder, "%s/point", input_folder);
        mkdir(output_folder, 0777);
    }

    vector<string> image_filenames;
    {
        // Loading image files
        snprintf(file_name, sizeof(file_name), "%s/image", input_folder);
        image_filenames = load_files(file_name, ".png");
        if (image_filenames.size() < 1) {
            throw std::runtime_error{ fmt::format("NOT images under folder '{}'", input_folder) };
        }

        sort(image_filenames.begin(), image_filenames.end());
        n_filenames = image_filenames.size();
        if (n_filenames >= MAX_IMAGES) {
            throw std::runtime_error{ fmt::format(
                "Too many images (>={}) under folder '{}'", MAX_IMAGES, input_folder) };
        }
    }

    vector<string> camera_filenames;
    {
        // Loading camera files
        snprintf(file_name, sizeof(file_name), "%s/camera", input_folder);
        camera_filenames = load_files(file_name, ".txt");
        if (camera_filenames.size() < 1) {
            throw std::runtime_error{ fmt::format(
                "NO camera files under folder '{}'", input_folder) };
        }
        sort(camera_filenames.begin(), camera_filenames.end());
        if (image_filenames.size() != camera_filenames.size()) {
            throw std::runtime_error{ fmt::format(
                "image/camera files DOES NOT match under folder '{}'", input_folder) };
        }
    }

    vector<string> depth_filenames;
    {
        // Loading depth files
        snprintf(file_name, sizeof(file_name), "%s/depth", input_folder);
        depth_filenames = load_files(file_name, ".png");
        // if depth_filnames == 0, suppose image including depth information in A channel
        if (depth_filenames.size() > 0 && depth_filenames.size() != n_filenames) {
            throw std::runtime_error{ fmt::format(
                "image/depth files DOES NOT match under folder '{}'", input_folder) };
        }
        sort(depth_filenames.begin(), depth_filenames.end());
    }

    // n_filenames = 10; // xxxx8888
    uint32_t image_width, image_height;
    {
        // image/depth files have same size ?
        int x, y, n, ok;
        ok = stbi_info(image_filenames[0].c_str(), &x, &y, &n);
        if (ok != 1) {
            throw std::runtime_error{ fmt::format(
                "image '{}' pixel size is not valid", image_filenames[0]) };
        }
        image_width = x;
        image_height = y;
        for (i = 1; i < n_filenames; i++) {
            ok = stbi_info(image_filenames[i].c_str(), &x, &y, &n);
            if (ok != 1 || x != image_width || y != image_height) {
                throw std::runtime_error{ fmt::format(
                    "image pixel size IS NOT same under '{}'", input_folder) };
            }
        }
        for (i = 0; i < n_filenames; i++) {
            ok = stbi_info(depth_filenames[i].c_str(), &x, &y, &n);
            if (ok != 1 || x != image_width || y != image_height) {
                throw std::runtime_error{ fmt::format(
                    "image/depth pixel size IS NOT same under '{}'", input_folder) };
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
            // cpu_cameras[i].dump();
        }
        gpu_cameras.copy_from_host(cpu_cameras);
        cpu_cameras.clear();

        load_camera_logger.success("OK !");
    }

    cudaTextureObject_t *gpu_textures;
    CUDA_CHECK_THROW(cudaMalloc(&gpu_textures, n_filenames * sizeof(cudaTextureObject_t)));
    {
        // Loading image/depth and save to textures
        int width, height;
        float* image_data;
        cudaTextureObject_t cpu_textures[MAX_IMAGES];

        auto load_texture_logger = tlog::Logger("Loading texture ...");
        auto progress = load_texture_logger.progress(n_filenames);

        for (i = 0; i < n_filenames; i++) {
            progress.update(i);

            if (depth_filenames.size() == 0) {
                image_data = load_image(image_filenames[i], width, height);
            } else {
                image_data
                    = load_image_and_depth(image_filenames[i], depth_filenames[i], width, height);
            }

            cudaArray* cuArray;
            cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
            CUDA_CHECK_THROW(cudaMallocArray(&cuArray, &channelDesc, width, height));
            CUDA_CHECK_THROW(cudaMemcpy2DToArray(cuArray /*dst*/, 0 /*wOffset*/, 0 /*hOffset*/,
                image_data /*src*/, width * sizeof(float4) /*spitch bytes*/, 
                width * sizeof(float4) /*width bytes*/, height /*rows*/, cudaMemcpyHostToDevice));
            CUDA_CHECK_THROW(cudaDeviceSynchronize());
            free(image_data); // release memory of image data

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
                one_image_gpu_points.data()
            );
            CUDA_CHECK_THROW(cudaDeviceSynchronize());

            one_image_gpu_points.copy_to_host(one_image_cpu_points);
            CUDA_CHECK_THROW(cudaDeviceSynchronize());

            // save valid points
            for (size_t j = 0; j < one_image_cpu_points.size(); j++) {
                Point pc = one_image_cpu_points[j];
                if (pc.xyzw.w() > 0.5f && pc.rgba.w() < MAX_DEPTH)
                    all_cpu_points.push_back(pc);
            }
        }
        fusion_points_logger.success("OK !");

        // one_image_gpu_points.free_memory();
        // one_image_cpu_points.clear();
    }
    CUDA_CHECK_THROW(cudaDeviceSynchronize());

    snprintf(file_name, sizeof(file_name), "%s/pc.ply", output_folder);
    save_point_cloud(file_name, all_cpu_points);
    // all_cpu_points.clear();

    return 0;
}
