/************************************************************************************
***
***     Copyright 2023 Dell Du(18588220928@163.com), All Rights Reserved.
***
***     File Author: Dell, 2023年 03月 07日 星期二 18:29:34 CST
***
************************************************************************************/

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
using namespace std;

#include <Eigen/Dense> // Version 3.4.9, eigen.tgz under dependencies
using namespace Eigen;

#include "tinylog.h"
// #include "nanoflann.hpp"

#include "mesh.h"

#define D_EPISON 1.0e-03 // distance epision
// 1.0 - math.cos(15.0 * 3.1415926/180.0) -- 0.03407
// 1.0 - cos(10.0/180.0 * 3.1415926) -- 0.01519
#define T_EPISON_10 0.01519f // cosine etheta epision, cos(10.0/180.0 * 3.1415926) -- 0.9848077535291953
#define T_EPISON_15 0.03407f

// using Point = Eigen::Vector3f;
// using Normal = Eigen::Vector3f;
// using UVMap = Eigen::Vector2f;
// using Face = std::vector<size_t>;
// using Points = std::vector<Point>;



// std::ostream& operator<<(std::ostream& os, const Point& point)
// {
//     os << std::fixed << "(" << point.x() << ", " << point.y() << ", " << point.z() << ")";
//     return os;
// }

// struct GridKey
// {
//     GridKey(uint32_t f, uint32_t s, uint32_t t) : i(f), j(s), k(t){}

//     friend std::ostream& operator<<(std::ostream& os, const GridKey& key)
//     {
//         os << std::fixed << "(" << key.i << ", " << key.j << ", " << key.k << ")";
//         return os;
//     }

//     uint32_t i, j, k;
// };

// struct HashFunc
// {
//     std::size_t operator()(const GridKey &key) const
//     {
//         return ((std::hash<uint32_t>()(key.i) ^ (std::hash<uint32_t>()(key.j) << 1)) >> 1) ^ (std::hash<uint32_t>()(key.k) << 1);
//     }
// };

// struct EqualKey
// {
//     bool operator () (const GridKey &lhs, const GridKey &rhs) const {
//         return lhs.i == rhs.i && lhs.j == rhs.j && lhs.k == rhs.k;
//     }
// };
// using GridIndex = std::unordered_map<GridKey, std::vector<uint32_t>, HashFunc, EqualKey>;

// struct Grid {
//     Grid(Points V, size_t N) {
//         BoundingBox aabb;
//         for (Point p: V)
//             aabb.update(p);
//         // aabb.xxxx

//     }

//     uint32_t nx, ny, nz;
//     GridIndex m_data;
// };

// struct Color {
//     friend std::ostream& operator<<(std::ostream& os, const Color& color)
//     {
//         os << std::fixed << color.r << " " << color.g << " " << color.b;
//         return os;
//     }

// public:
//     float r, g, b;
// };
// using GridColor = std::unordered_map<GridKey, Color>;
// using GridDensity = std::unordered_map<GridKey, float>;

// struct Material {
//     friend std::ostream& operator<<(std::ostream& os, const Material& material)
//     {
//         os << std::fixed;
//         os << "newmtl " << material.name << std::endl;
//         os << "Ns " << material.Ns << std::endl;
//         os << "Ka " << material.Ka << std::endl;
//         os << "Kd " << material.Kd << std::endl;
//         os << "Ks " << material.Ks << std::endl;
//         os << "Ke " << material.Ke << std::endl;
//         os << "Ni " << material.Ni << std::endl;
//         os << "d " << material.d << std::endl;
//         os << "illum " << material.illum << std::endl;
//         return os;
//     }

//     bool save(const char* filename)
//     {
//         ofstream out(filename);
//         if (!out.good()) {
//             tlog::error() << "Create file '" << filename << "'";
//             return false;
//         }
//         out << *this;
//         out.close();
//         return true;
//     }

// public:
//     std::string name = "Material"; // Texture name

//     Color Ka { 1.0, 1.0, 1.0 }; // Ambient color
//     Color Kd { 0.0, 0.0, 0.0 }; // Diffuse color
//     Color Ks { 0.5, 0.5, 0.5 }; // Specular color
//     Color Ke { 0.0, 0.0, 0.0 }; // Emit light color

//     float Ni = 1.45; // Density
//     float d = 1.0; // Transparency
//     float Ns = 250.0; // Shininess
//     int illum = 2; // Illumination model
// };

struct Plane {
    Plane(Point o, Normal n)
        : o(o)
        , n(n)
    {
        n = n.normalized();
    }

    Plane(const Point p1, const Point p2, const Point p3)
    {
        o = (p1 + p2 + p3) / 3.0f;
        n = (p2 - p1).cross(p3 - p1);
        n = n.normalized();
    }

    // p on plane ?
    bool contains(const Point p, float e = D_EPISON) const
    {
        return fabs(n.dot(p - o)) < e; // (p - o) _|_ n
    }

    Point project(const Point p)
    {
        float t = n.dot(p - o);
        return p - t * n;
    }

    float distance(const Point p)
    {
        return fabs(n.dot(p - o));
    }

    // fitting via ref_points ...
    void refine()
    {
        size_t size = ref_points.size();
        if (size > 3) {
            Eigen::MatrixXf A = Eigen::MatrixXf::Ones(size, 3);
            Eigen::VectorXf b = Eigen::VectorXf::Zero(size);

            for (size_t i = 0; i < size; i++) {
                A(i, 0) = ref_points[i].x();
                A(i, 1) = ref_points[i].y();
                b(i) = ref_points[i].z();
            }
            Eigen::VectorXf x = A.colPivHouseholderQr().solve(b);
            n.x() = x[0];
            n.y() = x[1];
            n.z() = -1.0f;

            x = A.colwise().mean();
            o.x() = x[0];
            o.y() = x[1];
            o.z() = b.colwise().mean().x();

            n = n.normalized();
        }
    }

    bool coincide(const Plane& other, float e = D_EPISON, float t = T_EPISON_15)
    {
        return this->contains(other.o, e) && other.contains(this->o, e) && (fabs(this->n.dot(other.n)) > 1.0f - t);
    }

    friend std::ostream& operator<<(std::ostream& os, const Plane& plane)
    {
        os << "Plane " << std::fixed;
        os << "[o: " << plane.o.x() << ", " << plane.o.y() << ", " << plane.o.z() << "; ";
        os << "n: " << plane.n.x() << ", " << plane.n.y() << ", " << plane.n.z() << "]";
        return os;
    }

public:
    Point o = Vector3f::Zero(); // orignal
    Normal n = Vector3f::Ones(); // normal, should be normalized !!!

    Points ref_points;
    std::vector<size_t> ref_indexs;
};

// struct Normals
// {
//     using coord_t = float;  //!< The type of each coordinate

//     // Must return the number of normals
//     inline size_t kdtree_get_point_count() const { return data.size(); }

//     // Returns the dim'th component of the idx'th point in the class:
//     // Since this is inlined and the "dim" argument is typically an immediate
//     // value, the
//     //  "if/else's" are actually solved at compile time.
//     inline float kdtree_get_pt(const size_t idx, const size_t dim) const
//     {
//         if (dim == 0)
//             return data[idx].x();
//         else if (dim == 1)
//             return data[idx].y();
//         else
//             return data[idx].z();
//     }

//     // Optional bounding-box computation: return false to default to a standard
//     // bbox computation loop.
//     //   Return true if the BBOX was already computed by the class and returned
//     //   in "bb" so it can be avoided to redo it again. Look at bb.size() to
//     //   find out the expected dimensionality (e.g. 2 or 3 for point clouds)
//     template <class BBOX>
//     bool kdtree_get_bbox(BBOX& /* bb */) const
//     {
//         return false;
//     }

// public:
//     std::vector<Normal> data;
// };

// struct PointCloud
// {
//     using coord_t = float;  //!< The type of each point

//     // Must return the number of points
//     inline size_t kdtree_get_point_count() const { return data.size(); }

//     //  "if/else's" are actually solved at compile time.
//     inline float kdtree_get_pt(const size_t idx, const size_t dim) const
//     {
//         if (dim == 0)
//             return data[idx].x();
//         else if (dim == 1)
//             return data[idx].y();
//         else
//             return data[idx].z();
//     }

//     // Optional bounding-box computation: return false to default to a standard
//     // bbox computation loop.
//     //   Return true if the BBOX was already computed by the class and returned
//     //   in "bb" so it can be avoided to redo it again. Look at bb.size() to
//     //   find out the expected dimensionality (e.g. 2 or 3 for point clouds)
//     template <class BBOX>
//     bool kdtree_get_bbox(BBOX& /* bb */) const
//     {
//         return false;
//     }

// public:
//     std::vector<Point> data;
// };

struct BoundingBox {
    BoundingBox() { }

    BoundingBox(const Points V)
    {
        for (Point p : V)
            update(p);
    }

    BoundingBox(const Point& a, const Point& b)
        : min { a }
        , max { b }
    {
    }

    void update(const Point& point)
    {
        min = min.cwiseMin(point);
        max = max.cwiseMax(point);
    }

    void inflate(float amount)
    {
        min -= Point::Constant(amount);
        max += Point::Constant(amount);
    }

    Point diag() { return max - min; }

    void voxel(size_t N)
    {
        radius = diag().maxCoeff() / N;
        Eigen::Vector3f f = diag() / radius;
        dim.x() = (int)ceilf(f.x());
        dim.y() = (int)ceilf(f.y());
        dim.z() = (int)ceilf(f.z());
        radius = radius / 2.0f;
    }

    Point relative_pos(const Point& point) { return (point - min).cwiseQuotient(diag()); }

    GridKey grid_key(const Point& point, float step)
    {
        Point pos = (point - min) / step;
        return GridKey { (uint32_t)pos.x(), (uint32_t)pos.y(), (uint32_t)pos.z() };
    }

    Point center() { return 0.5f * (max + min); }

    BoundingBox intersection(const BoundingBox& other)
    {
        BoundingBox result = *this;
        result.min = result.min.cwiseMax(other.min);
        result.max = result.max.cwiseMin(other.max);
        return result;
    }

    bool intersects(const BoundingBox& other) { return !intersection(other).is_empty(); }

    bool is_empty() const { return (max.array() < min.array()).any(); }

    bool contains(const Point& p)
    {
        return p.x() >= min.x() && p.x() <= max.x() && p.y() >= min.y() && p.y() <= max.y()
            && p.z() >= min.z() && p.z() <= max.z();
    }

    friend std::ostream& operator<<(std::ostream& os, const BoundingBox& bb)
    {
        os << "[";
        os << "min=[" << bb.min.x() << "," << bb.min.y() << "," << bb.min.z() << "], ";
        os << "max=[" << bb.max.x() << "," << bb.max.y() << "," << bb.max.z() << "], ";
        os << "dim=[" << bb.dim.x() << "," << bb.dim.y() << "," << bb.dim.z() << "], ";
        os << "radius=" << bb.radius;
        os << "]";
        return os;
    }

    Point min = Point::Constant(std::numeric_limits<float>::infinity());
    Point max = Point::Constant(-std::numeric_limits<float>::infinity());

    // Helper
    float radius = 1.0;
    Eigen::Vector3i dim = Vector3i { 1, 1, 1 };
};

#if 0
struct Mesh {
    void reset()
    {
        V.clear();
        C.clear();
        F.clear();
    }


    void dump() {
        std::cout << "Mesh vertext: " << std::fixed << V.size() << ", face: " << F.size() << std::endl;
    }

    friend std::ostream& operator<<(std::ostream& os, const Mesh& mesh)
    {
        os << std::fixed;
        os << "# Vertext: " << std::fixed << mesh.V.size() << ", Face: " << mesh.F.size() << std::endl;

        // v x y z or v x y z r g b
        bool has_color = (mesh.V.size() == mesh.C.size());
        for (size_t i = 0; i < mesh.V.size(); i++) {
            os << "v " << mesh.V[i].x() << " " << mesh.V[i].y() << " " << mesh.V[i].z();
            if (has_color)
                os << " " << mesh.C[i].r << " " << mesh.C[i].g << " " << mesh.C[i].b;
            os << std::endl;
        }

        for (size_t i = 0; i < mesh.F.size(); i++) {
            os << "f";
            for (size_t f : mesh.F[i])
                os << " " << f + 1; // face index list start from 1 in obj format
            os << std::endl;
        }

        return os;
    }


    Mesh grid(size_t N) {
        Mesh outmesh;
        // GridDensity density;
        GridIndex grid_index;

        BoundingBox aabb(V);
        aabb.voxel(N);
        float step = aabb.radius * 2.0f;

        // for (Point point: V) {
        //     Point ijk = (point - aabb.min)/step;
        //     uint32_t i = (uint32_t)ijk.x();
        //     uint32_t j = (uint32_t)ijk.y();
        //     uint32_t k = (uint32_t)ijk.z();
        //     GridKey key = GridKey{i, j, k};

        //     // uint32_t key = grid_key(i, j, k);
        //     // if (density.find(key) != density.end()) {
        //     //     density[key] += 1.0f;
        //     // } else {
        //     //     density[key] = 1.0f;
        //     // }

        // }
        auto grid_logger = tlog::Logger("Grid ...");
        auto progress = grid_logger.progress(V.size());
        for (size_t n = 0; n < V.size(); n++) {
            progress.update(n);
            GridKey key = aabb.grid_key(V[n], step);
            grid_index[key].push_back(n);
        }
        grid_logger.success("OK !");

        for (auto n:grid_index) {
            const std::vector<uint32_t> &index_list = n.second;
            for (size_t i = 0; i < index_list.size(); i++) {
                outmesh.V.push_back(V[index_list[i]]);
                break;
            }
        }



        // PointCloud pc;
        // for (Point point:V)
        //     pc.data.push_back(point);

        // float query_point[3];
        // using pc_kd_tree_t = nanoflann::KDTreeSingleIndexAdaptor<
        //     nanoflann::L2_Simple_Adaptor<float, PointCloud>,
        //     PointCloud, 3 /* dim */>;
        // pc_kd_tree_t index(3 /*dim*/, pc, {10 /* max leaf */});

        // float squaredRadius = aabb.radius * aabb.radius;
        // std::vector<nanoflann::ResultItem<size_t, float>> ret_indices;
        // nanoflann::RadiusResultSet<float, size_t> resultSet(squaredRadius, ret_indices);

        size_t search_count = 0;

        // for (int i = 0; i < aabb.dim.x(); i++) {
        //     query_point[0] = aabb.min.x() + (i + 0.5) * step;
        //     for (int j = 0; j < aabb.dim.y(); j++) {
        //         query_point[1] = aabb.min.y() + (j + 0.5) * step;
        //         for (int k = 0; k < aabb.dim.z(); k++) {
        //             query_point[2] = aabb.min.z() + (k + 0.5) * step;

        //             // do radius search and create density ...
        //             resultSet.clear();
        //             index.findNeighbors(resultSet, query_point); // nanoflann::SearchParams(10)
        //             if (resultSet.size() == 0)
        //                 continue;
        //             density[grid_key(i, j, k)] = resultSet.size();
        //             search_count += resultSet.size();
        //         }
        //     }
        // }

        for (auto d:grid_index) {
            search_count += (int) d.second.size();
            GridKey key = d.first;
            if (key.k % 100 == 0)
                 std::cout << key << " -- " << d.second.size() << std::endl;
        }

        std::cout << "aabb:" << aabb << std::endl;
        std::cout << "search_count: " << search_count << std::endl;

        return outmesh;
    }

    bool loadOBJ(const char* filename)
    {
        int count;
        char line[1024], *tv[8], *f_tv[8];
        Face one_face_index;

        FILE* fp = fopen(filename, "r");
        if (fp == NULL) {
            fprintf(stderr, "Error: %s could not be opened for reading ...", filename);
            return false;
        }

        reset();

        while (fgets(line, sizeof(line), fp) != NULL) {
            count = get_token(line, ' ', 8, tv);

            if (strcmp(tv[0], "v") == 0 && (count == 4 || count == 7)) {
                V.push_back(Point { (float)atof(tv[1]), (float)atof(tv[2]), (float)atof(tv[3]) });
                if (count == 7)
                    C.push_back(Color { (float)atof(tv[4]), (float)atof(tv[5]), (float)atof(tv[6]) });
            } else if (strcmp(tv[0], "f") == 0) {
                // f 1/1/1 2/2/2 4/4/3 3/3/4 -- face/texture/normal index
                one_face_index.clear();

                for (int j = 1; j < count; j++) {
                    get_token(tv[j], '/', 3, f_tv);
                    one_face_index.push_back(atoi(f_tv[0]) - 1);
                }
                if (one_face_index.size() >= 3)
                    F.push_back(one_face_index);
            }
        }
        fclose(fp);

        return true;
    }

    bool saveOBJ(const char* filename)
    {
        ofstream out(filename);
        if (! out.good()) {
            fprintf(stderr, "Error: create OBJ file '%s' for saving ... \n", filename);
            return false;
        }
        out << *this;
        out.close();
        return true;
    }

    void snap(float e=D_EPISON, float t=T_EPISON_15) {
        std::vector<Plane> planes;

        auto snap_logger = tlog::Logger("Create face normals ...");
        auto progress = snap_logger.progress(F.size());
        for (Face f : F) {
            progress.update(planes.size());
            if (f.size() >= 3) {
                Plane face_plane(V[f[0]], V[f[1]], V[f[2]]);
                face_plane.ref_points.clear();
                face_plane.ref_points.push_back(V[f[0]]);
                face_plane.ref_points.push_back(V[f[1]]);
                face_plane.ref_points.push_back(V[f[2]]);

                face_plane.ref_indexs.clear();
                face_plane.ref_indexs.push_back(f[0]);
                face_plane.ref_indexs.push_back(f[1]);
                face_plane.ref_indexs.push_back(f[2]);

                for (size_t i = 3; i < f.size(); i++) {
                    face_plane.ref_points.push_back(V[f[i]]);
                    face_plane.ref_indexs.push_back(f[i]);
                }

                if (f.size() > 3)
                    face_plane.refine();
                planes.push_back(face_plane);
            }
        }
        snap_logger.success("OK !");

        Normals normals;
        for (Plane plane:planes)
            normals.data.push_back(plane.n);

        // construct a kd-tree index:
        using normal_kd_tree_t = nanoflann::KDTreeSingleIndexAdaptor<
            nanoflann::L2_Simple_Adaptor<float, Normals>,
            Normals, 3 /* dim */
        >;
        normal_kd_tree_t index(3 /*dim*/, normals, {10 /* max leaf */});
        float query_point[3];

        const size_t num_results = 2048;
        size_t ret_index[num_results];
        float out_dist_sqr[num_results];
        nanoflann::KNNResultSet<float> resultSet(num_results);
        resultSet.init(ret_index, out_dist_sqr);

        snap_logger = tlog::Logger("Create support faces ...");
        progress = snap_logger.progress(planes.size());

        std::vector<int> planes_hold(planes.size(), 1);
        for (size_t i = 0; i < planes.size(); i++) {
            progress.update(i);
            if (planes_hold[i] == 0)
                continue;

            // do a knn search
            query_point[0] = planes[i].n.x();
            query_point[1] = planes[i].n.y();
            query_point[2] = planes[i].n.z();
            index.findNeighbors(resultSet, &query_point[0]);

            // std::cout << "query point: ==> " << resultSet.size() << " -- " << planes[i] << std::endl;
            for (size_t k = 1; k < resultSet.size(); k++) {
                size_t j = ret_index[k];
                if (planes_hold[j] == 0 || j <= i)
                     continue;

                // std::cout << "dist " << j << ": " << sqrtf(out_dist_sqr[k]) << std::endl;
                // std::cout << planes[j] << std::endl;

                if (planes[i].coincide(planes[j], e, t)) {
                    // std::cout << "merge available ... " << i << " with ... " << j << std::endl;
                    planes[i].ref_points.insert(planes[i].ref_points.end(), planes[j].ref_points.begin(), planes[j].ref_points.end());
                    planes[i].ref_indexs.insert(planes[i].ref_indexs.end(), planes[j].ref_indexs.begin(), planes[j].ref_indexs.end());

                    planes[j].ref_points.clear();
                    planes[j].ref_indexs.clear();
                    planes_hold[j] = 0;
                }
            }
        }
        snap_logger.success("OK !");

        snap_logger = tlog::Logger("Refine support faces ...");
        progress = snap_logger.progress(planes.size());
        std::vector<Plane> support_planes;
        for (size_t i = 0; i < planes.size(); i++) {
            progress.update(i);
            if (planes_hold[i] == 1) {
                planes[i].refine();
                support_planes.push_back(planes[i]);
            }
        }
        planes.clear();
        planes_hold.clear();
        snap_logger.success("OK !");
        std::cout << "Support planes: " << support_planes.size() << std::endl;


        snap_logger = tlog::Logger("Snap points ...");
        progress = snap_logger.progress(support_planes.size());
        for (size_t i = 0; i < support_planes.size(); i++) {
            progress.update(i);
            Plane plane = support_planes[i];
            if (plane.ref_indexs.size() <= 3)
                continue;
            for (size_t j : plane.ref_indexs) {
                if (plane.contains(V[j], e))
                    V[j] = plane.project(V[j]);
            }
        }
        snap_logger.success("OK !");

        support_planes.clear();
    }

public:
    std::vector<Point> V; // v x y z -- Vertex Command
    std::vector<Color> C; // Vertex Color

    std::vector<Face> F; // Face Index List
};

#endif

void test_plane()
{
    // Points points;

    // Point p1 = Point { 0.0, 0.0, 0.0 };
    // Point p2 = Point { 1.0, 0.0, 0.0 };
    // Point p3 = Point { 0.0, 2.0, 0.0 };
    // Point p4 = Point { 0.0, 0.0, 3.0 };

    // points.push_back(p1);
    // points.push_back(p2);
    // points.push_back(p3);
    // // points.push_back(Eigen::Vector3f{0.01, 0.01, 0.01});

    // Plane plane(points);
    // std::cout << plane << std::endl;

    // Point p = Point { 0.000001, 0.000001, 0.000001 };
    // std::cout << p << " on plane (epision = default) ? " << plane.contains(p) << std::endl;
    // std::cout << p << " on plane (epision = 0.1f)? " << plane.contains(p, 0.1f) << std::endl;

    // BoundingBox aabb;
    // aabb.update(p1);
    // aabb.update(p2);
    // aabb.update(p3);
    // aabb.update(p4);
    // aabb.voxel(100);
    // std::cout << "aabb:" << aabb << std::endl;
    // std::cout << "aabb dim:" << aabb.dim << std::endl;

    // Plane B(plane.o - Point { D_EPISON / 2.0, D_EPISON / 2.0, D_EPISON / 2.0 },
    //     plane.n + Point { D_EPISON / 2.0, D_EPISON / 2.0, D_EPISON / 2.0 });

    // std::cout << B << std::endl;
    // std::cout << "Plane coincide with B ?" << plane.coincide(B) << std::endl;

    Mesh mesh;

    // mesh.savePLY("empty.ply");
    // mesh.loadOBJ("mesh.obj");
    mesh.loadPLY("lego/point/pc.ply");
    mesh.dump();
    mesh.saveOBJ("test.obj");

    // BoundingBox aabb(mesh.V);
    // aabb.voxel(512);
    // std::cout << "aabb(V):" << aabb << std::endl;

    // Mesh outmesh = mesh.grid(1024);
    // outmesh.saveOBJ("grid.obj");

    // mesh.snap(0.05f, 2*T_EPISON_10);
    // mesh.saveOBJ("snap.obj");

    // Material material;
    // material.save("test.mtl");

    // GridDensity grid_density;
    // GridColor grid_color;

    // grid_density[grid_key(0, 0, 0)] = 1.0f;
    // grid_density[grid_key(100, 100, 100)] = 2.0f;
    // grid_density[grid_key(101, 201, 301)] = 3.0f;

    // grid_color[grid_key(0, 0, 0)] = Color{1.0, 0.0, 0.0};
    // grid_color[grid_key(100, 100, 100)] = Color{0.0, 1.0, 0.0};
    // grid_color[grid_key(101, 201, 301)] = Color{0.0, 0.0, 1.0};

    // std::cout << std::fixed;

    // if (grid_density.find(grid_key(101, 201, 301)) == grid_density.end())
    //     std::cout << "not found 101" << std::endl;
    // else
    //     std::cout << "found 101" << std::endl;

    // for (auto d:grid_density) {
    //     std::cout << d.first << " -- " << d.second << std::endl;
    // }

    // for (auto d:grid_color) {
    //     std::cout << d.first << " -- " << d.second << std::endl;
    // }
}
