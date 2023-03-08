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

// 0) Device
// 1) Camera
// 2) Image
// 3) Points

// 0) Camera
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
    const Camera& camera, Vector3f* __restrict__ xyz)
{
    Vector3f pt
        = Vector3f{ depth * u - camera.KT.x(), depth * v - camera.KT.y(), depth - camera.KT.z() };
    *xyz = camera.R_K_inv * pt;
}

__device__ void world_to_image(const Vector3f& xyz, const Camera& camera, float* u, float* v)
{
    Vector3f temp = camera.K * camera.R * xyz + camera.K * camera.T;
    // *depth = temp.z();
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
    cout << "KR:" << endl << this->K * this->R << endl;
    cout << "KT:" << endl << this->K * this->T << endl;
    cout << "R_K_inv:" << endl << this->R_K_inv << endl;
}

// 2) Image
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
    for (int i = 0; i < width * height; i++) {
        if (src[3] < 0.5f) { // Image masked, depth is far ...
            dst[3] = MAX_DEPTH;
        } else { // The feature of depth is more near, more bright
            dst[3] = (1.0f - src[0]) * 256.0f + (1.0f - src[1]) + (1.0f - src[2]) / 256.0f;
        }
        src += 4;
        dst += 4;
    }
    free(depth_data); // release memory of depth data

    return image_data;
}

// 3) Point
/************************************************************************************/
__global__ void fusion_point_kernel(const uint32_t image_k, const Vector2i& image_size,
    const cudaTextureObject_t* __restrict__ image_textures, const uint32_t n_cameras,
    const Camera* __restrict__ gpu_cameras, Point* __restrict__ one_image_gpu_points)
{
    uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x >= image_size.x() || y >= image_size.y())
        return;

    const uint32_t center = y * image_size.x() + x;
    float4 sum_rgba = tex2D<float4>(image_textures[image_k], x + 0.5f, y + 0.5f);
    float depth = sum_rgba.w;
    if (depth < MIN_DEPTH || depth >= MAX_DEPTH)
        return;

    Vector3f xyz;
    image_to_world((float)x, (float)y, depth, gpu_cameras[image_k], &xyz);

    int count = 0;
    Vector3f sum_xyz = xyz;
    for (uint32_t i = 0; i < n_cameras && count < 6; i++) {
        if (i == image_k)
            continue;

        // Project 3d point xyz on camera i
        float i_u, i_v;
        world_to_image(xyz, gpu_cameras[i], &i_u, &i_v);

        // Boundary check
        if ((int)i_u < 0 || (int)i_u >= image_size.x() || (int)i_v < 0
            || (int)i_v >= image_size.y())
            continue;

        float4 i_rgba = tex2D<float4>(image_textures[i], i_u + 0.5f, i_v + 0.5f);
        if (i_rgba.w < MIN_DEPTH || i_rgba.w > MAX_DEPTH)
            continue;

        float depth_disp = get_disparity(gpu_cameras[image_k], gpu_cameras[i], depth, i_rgba.w);

        // check on depth
        if (fabsf(depth_disp) < 0.25f) {
            // depth_threshold == 0.25
            Vector3f i_xyz; // 3d point of consistent point on other view
            image_to_world(i_v, i_u, i_rgba.w, gpu_cameras[i], &i_xyz);

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
    cout << "Save " << pc.size() << " points to " << filename << " ..." << endl;

    FILE* fp = fopen(filename.c_str(), "wb");

    /*write header */
    fprintf(fp, "ply\n");
    fprintf(fp, "format binary_little_endian 1.0\n");
    fprintf(fp, "element vertex %ld\n", pc.size());
    fprintf(fp, "property float x\n");
    fprintf(fp, "property float y\n");
    fprintf(fp, "property float z\n");
    fprintf(fp, "property uchar red\n");
    fprintf(fp, "property uchar green\n");
    fprintf(fp, "property uchar blue\n");
    fprintf(fp, "end_header\n");

    // write data
#pragma omp parallel for
    for (size_t i = 0; i < pc.size(); i++) {
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

    if (has_cuda_device()) {
        dump_gpu_memory();
    }

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
        if (depth_filenames.size() > 0 && depth_filenames.size() != image_filenames.size()) {
            throw std::runtime_error{ fmt::format(
                "image/depth files DOES NOT match under folder '{}'", input_folder) };
        }
        sort(depth_filenames.begin(), depth_filenames.end());
    }

    Vector2i image_size;
    {
        // image/depth files have same size ?
        int x, y, n, ok;
        ok = stbi_info(image_filenames[0].c_str(), &x, &y, &n);
        if (ok != 1) {
            throw std::runtime_error{ fmt::format(
                "image '{}' pixel size is not valid", image_filenames[0]) };
        }
        image_size.x() = x;
        image_size.y() = y;
        for (i = 1; i < image_filenames.size(); i++) {
            ok = stbi_info(image_filenames[i].c_str(), &x, &y, &n);
            if (ok != 1 || x != image_size.x() || y != image_size.y()) {
                throw std::runtime_error{ fmt::format(
                    "image pixel size IS NOT same under '{}'", input_folder) };
            }
        }
        for (i = 0; i < depth_filenames.size(); i++) {
            ok = stbi_info(depth_filenames[i].c_str(), &x, &y, &n);
            if (ok != 1 || x != image_size.x() || y != image_size.y()) {
                throw std::runtime_error{ fmt::format(
                    "image/depth pixel size IS NOT same under '{}'", input_folder) };
            }
        }
    }

    GPUMemory<Camera> gpu_cameras(n_filenames);
    {
        // Loading cameras ...
        vector<Camera> cpu_cameras(n_filenames);
        for (i = 0; i < n_filenames; i++) {
            Camera camera;
            camera.load(camera_filenames[i]);
            cpu_cameras.push_back(camera);
        }
        gpu_cameras.copy_from_host(cpu_cameras);
        cpu_cameras.clear();
    }

    cudaTextureObject_t image_textures[MAX_IMAGES];
    {
        // Loading image/depth and save to textures
        int width, height;
        float* image_data;

        for (i = 0; i < n_filenames; i++) {
            if (depth_filenames.size() == 0) {
                image_data = load_image(image_filenames[i], width, height);
            } else {
                image_data
                    = load_image_and_depth(image_filenames[i], depth_filenames[i], width, height);
            }
            GPUMemory<float> gpu_image_data(width * height * 4);
            gpu_image_data.copy_from_host(image_data);
            free(image_data); // release memory of image data

            // Create a cuda texture out of this image.
            cudaResourceDesc resDesc;
            memset(&resDesc, 0, sizeof(resDesc));
            resDesc.resType = cudaResourceTypePitch2D;
            resDesc.res.pitch2D.devPtr = gpu_image_data.data();
            resDesc.res.pitch2D.desc
                = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
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

            CUDA_CHECK_THROW(
                cudaCreateTextureObject(&image_textures[i], &resDesc, &texDesc, nullptr));
        }
    }

    vector<Point> all_cpu_points;
    {
        vector<Point> one_image_cpu_points(image_size.x() * image_size.y());
        GPUMemory<Point> one_image_gpu_points(image_size.x() * image_size.y());

        for (i = 0; i < n_filenames; i++) {
            // process one_image_gpu_points
            const dim3 threads = { 32, 32, 1 };
            const dim3 blocks = { div_round_up((unsigned int)image_size.x(), threads.x),
                div_round_up((unsigned int)image_size.y(), threads.y), 1 };

            fusion_point_kernel<<<blocks, threads, 0, nullptr>>>(i, image_size, image_textures,
                n_filenames, gpu_cameras.data(), one_image_gpu_points.data());

            one_image_gpu_points.copy_to_host(one_image_cpu_points);

            // save valid points
            for (size_t j = 0; j < one_image_cpu_points.size(); j++) {
                Point pc = one_image_cpu_points[j];
                if (pc.xyzw.w() > 0.5f)
                    all_cpu_points.push_back(pc);
            }
        }
        one_image_gpu_points.free_memory();
        one_image_cpu_points.clear();
    }

    snprintf(file_name, sizeof(file_name), "%s/pc.ply", output_folder);
    save_point_cloud(file_name, all_cpu_points);
    all_cpu_points.clear();

    return 0;
}
