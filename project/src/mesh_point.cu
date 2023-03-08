/************************************************************************************
***
***     Copyright 2023 Dell Du(18588220928@163.com), All Rights Reserved.
***
***     File Author: Dell, 2023年 03月 07日 星期二 18:29:34 CST
***
************************************************************************************/
#include "../include/mesh_point.h"
#include "../include/meshbox.h"
#include "../include/mesh_common.h"

__device__ float get_disparity(
    const Camera & camera1,
    const Camera & camera2,
    const float depth1, 
    const float depth2)
{
    Vector3f a = camera2.O - camera1.O;
    float baseline = sqrtf(a.x()*a.x() + a.y()*a.y() + a.z()*a.z());
    // camera.K(0,0) -- focal length
    // depth/baseline=focal_length/? ==> ? = focal_leng * baseline/depth
    return camera1.K(0,0) * baseline / depth1 - camera2.K(0,0) * baseline / depth2;
}

__device__ void image_to_world(
    const float u, const float v, const float depth, const Camera& camera,
    Vector3f* __restrict__ xyz)
{
    Vector3f pt = Vector3f{ depth * u - camera.KT.x(),
        depth * v - camera.KT.y(), depth - camera.KT.z() };
    *xyz = camera.R_K_inv * pt;
}

__device__ void world_to_image(
    const Vector3f& xyz, const Camera& camera,
    float* u, float* v)
{
    Vector3f temp = camera.K * camera.R * xyz + camera.K * camera.T;
    // *depth = temp.z();
    *u = temp.x() / (temp.z() + 1e-10);
    *v = temp.y() / (temp.z() + 1e-10);
}

__global__ void fusion_point_cloud_kernel(
    const uint32_t image_k,
    const Vector2i& image_size,
    const cudaTextureObject_t* __restrict__ image_textures,
    const uint32_t n_cameras,
    const Camera* __restrict__ gpu_cameras,
    Point* __restrict__ one_image_gpu_points)
{
    uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x >= image_size.x() || y >= image_size.y())
        return;

    const uint32_t center = y * image_size.x() + x;
    float4 sum_rgba = tex2D < float4 > (image_textures[image_k], x + 0.5f, y + 0.5f);
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
        if ((int)i_u < 0 || (int)i_u >= image_size.x() || (int)i_v < 0 || (int)i_v >= image_size.y())
            continue;

        float4 i_rgba = tex2D < float4 > (image_textures[i], i_u + 0.5f, i_v + 0.5f);
        if (i_rgba.w < MIN_DEPTH || i_rgba.w > MAX_DEPTH)
            continue;

        float depth_disp = get_disparity(gpu_cameras[image_k], gpu_cameras[i], depth, i_rgba.w);

        // check on depth
        if (fabsf(depth_disp) < 0.25f) {
            // depth_threshold == 0.25
            Vector3f i_xyz;       // 3d point of consistent point on other view
            image_to_world(i_v, i_u, i_rgba.w, gpu_cameras[i], &i_xyz);

            sum_xyz = Vector3f{sum_xyz.x() + i_xyz.x(), sum_xyz.y() + i_xyz.y(), sum_xyz.z() + i_xyz.z()};
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
        sum_xyz = Vector3f{sum_xyz.x()/fc, sum_xyz.y()/fc, sum_xyz.z()/fc};
        sum_rgba.x /= fc;
        sum_rgba.y /= fc;
        sum_rgba.z /= fc;
        sum_rgba.w /= fc;

        one_image_gpu_points[center].xyzw = Vector4f{sum_xyz.x(), sum_xyz.y(), sum_xyz.z(), 1.0f }; // mark w == 1.0f
        one_image_gpu_points[center].rgba = Vector4f{sum_rgba.x, sum_rgba.y, sum_rgba.z, sum_rgba.w};
    }
}

void save_point_cloud(const string& filename, const vector<Point>& pc)
{
    tlog::info() << "Save " << pc.size() << " points to " << filename << " ...";

    FILE *fp = fopen(filename.c_str(), "wb");

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

    //write data
#pragma omp parallel for
    for (size_t i = 0; i < pc.size(); i++) {
        const Point& p = pc[i];

        const char color_r = (int) (p.rgba.x() * 255.0);
        const char color_g = (int) (p.rgba.y() * 255.0);
        const char color_b = (int) (p.rgba.z() * 255.0);

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


int run_fusibile(char* input_folder)
{
    size_t i, n_filenames;
    char output_folder[1024], file_name[2048];

    Vector2i image_size;
    vector<string> image_filenames;
    vector<string> camera_filenames;
    vector<string> depth_filenames;

    sprintf(output_folder, "%s/point", input_folder);
    mkdir(output_folder, 0777);

    {
        // Loading image files
        snprintf(file_name, sizeof(file_name), "%s/image", input_folder);
        image_filenames = load_files(file_name, ".png");
        if (image_filenames.size() < 1) {
            std::cout << "Error: NOT found images under folder '" << file_name << "'" << std::endl;
            return -1;
        }
        std::sort(image_filenames.begin(), image_filenames.end());
        n_filenames = image_filenames.size();
        if (n_filenames >= MAX_IMAGES) {
            std::cout << "Error: Too many image files under folder '" << file_name << "'" << std::endl;
            return -1;
        }
    }

    {
        // Loading camera files
        snprintf(file_name, sizeof(file_name), "%s/camera", input_folder);
        camera_filenames = load_files(file_name, ".txt");
        if (camera_filenames.size() < 1) {
            std::cout << "Error: NOT found camera files under folder '" << file_name << "'" << std::endl;
            return -1;
        }
        std::sort(camera_filenames.begin(), camera_filenames.end());
        if (image_filenames.size() != camera_filenames.size()) {
            std::cout << "Error: image/camera files DOES NOT match under '" << input_folder << "'" << std::endl;
            return -1;
        }
    }

    {
        // Loading depth files
        snprintf(file_name, sizeof(file_name), "%s/depth", input_folder);
        depth_filenames = load_files(file_name, ".png");
        // if depth_filnames == 0, suppose image including depth information in A channel
        if (depth_filenames.size() > 0 && depth_filenames.size() != image_filenames.size()) {
            std::cout << "Error: image/depth files DOES NOT match under '" << input_folder << "'" << std::endl;
            return -1;
        }
        std::sort(depth_filenames.begin(), depth_filenames.end());
    }

    {
        // image/depth files have same size ?
        int x, y, n, ok;
        ok = stbi_info(image_filenames[0].c_str(), &x, &y, &n);
        if (ok != 1) {
            std::cout << "Error: image file " << image_filenames[0] << " NOT valid" << std::endl;
            return -1;
        }
        image_size.x() = x;
        image_size.y() = y;
        for (i = 1; i < image_filenames.size(); i++) {
            ok = stbi_info(image_filenames[i].c_str(), &x, &y, &n);
            if (ok != 1 || x != image_size.x() || y != image_size.y()) {
                std::cout << "Error: image pixel size IS NOT same under '" << input_folder << "'" << std::endl;
                return -1;
            }
        }
        for (i = 0; i < depth_filenames.size(); i++) {
            ok = stbi_info(depth_filenames[i].c_str(), &x, &y, &n);
            if (ok != 1 || x != image_size.x() || y != image_size.y()) {
                std::cout << "Error: depth/image pixel size IS NOT same under '" << input_folder << "'" << std::endl;
                return -1;
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
        for (i = 0; i < n_filenames; i++) {
            if (depth_filenames.size() == 0) {
                GPUMemory<float> image = load_image(image_filenames[i], width, height);
                save_image_as_texture(image, width, height, image_textures[i]);
            } else {
                GPUMemory<float> image = load_image_and_depth(image_filenames[i],
                    depth_filenames[i], width, height);
                save_image_as_texture(image, width, height, image_textures[i]);
            }
        }
    }

    vector<Point> all_cpu_points;
    {
        vector<Point> one_image_cpu_points(image_size.x() * image_size.y());
        GPUMemory<Point> one_image_gpu_points(image_size.x() * image_size.y());

        for (i = 0; i < n_filenames; i++) {
            // process one_image_gpu_points
            const dim3 threads = { 32, 32, 1 };
            const dim3 blocks = { 
                div_round_up((unsigned int)image_size.x(), threads.x),
                div_round_up((unsigned int)image_size.y(), threads.y),
                1 };

            fusion_point_cloud_kernel<<<blocks, threads, 0, nullptr>>>(
                i, image_size,
                image_textures,
                n_filenames,
                gpu_cameras->data(),
                one_image_gpu_points->data()
            );

            one_image_gpu_points.copy_to_host(one_image_cpu_points);
            // save valid points to all_cpu_points
            for (size_t j = 0; j < one_image_cpu_points.size(); j++) {
                Point pc = one_image_cpu_points[j];
                if (pc.xyzw.w > 0.5f) {
                    all_cpu_points.push_back(pc);
                }
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
