/************************************************************************************
***
***     Copyright 2023 Dell Du(18588220928@163.com), All Rights Reserved.
***
***     File Author: Dell, 2023年 03月 07日 星期二 18:29:34 CST
***
************************************************************************************/
#pragma once

#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>
using namespace std;

#include <Eigen/Dense> // Version 3.4.9, eigen.tgz under dependencies
using namespace Eigen;

#include "tinylog.h"

#define D_EPISON 1.0e-03 // distance epision
// 1.0 - math.cos(10.0/180.0 * 3.1415926) -- 0.01519
// 1.0 - math.cos(15.0 * 3.1415926/180.0) -- 0.03407
#define T_EPISON_10 0.01519f
#define T_EPISON_15 0.03407f

struct Color;
struct Mesh;
struct GridKey;
struct HashFunc;
struct EqualKey;

using Point = Eigen::Vector3f;
using Points = std::vector<Point>;
using Colors = std::vector<Color>;
using Face = std::vector<uint32_t>;
using Faces = std::vector<Face>;
using Normal = Eigen::Vector3f;
using Normals = std::vector<Normal>;
// using UVMap = Eigen::Vector2f;
using Mask = std::vector<bool>;
using IndexList = std::vector<uint32_t>;
using MeshList = std::vector<Mesh>;
using GridColor = std::unordered_map<GridKey, Color>;
using GridDensity = std::unordered_map<GridKey, float>;
using GridIndex = std::unordered_map<GridKey, IndexList, HashFunc, EqualKey>;


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
        n.normalize();
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

            n.normalize();
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
    IndexList ref_indexs;
};

struct Color {
    friend std::ostream& operator<<(std::ostream& os, const Color& color)
    {
        os << std::fixed << color.r << " " << color.g << " " << color.b;
        return os;
    }

public:
    float r, g, b;
};

struct GridKey {
    GridKey(uint32_t f, uint32_t s, uint32_t t)
        : i(f)
        , j(s)
        , k(t)
    {
    }

    friend std::ostream& operator<<(std::ostream& os, const GridKey& key)
    {
        os << std::fixed << "(" << key.i << ", " << key.j << ", " << key.k << ")";
        return os;
    }

    uint32_t i, j, k;
};

struct HashFunc {
    std::size_t operator()(const GridKey& key) const
    {
        return ((std::hash<uint32_t>()(key.i) ^ (std::hash<uint32_t>()(key.j) << 1)) >> 1) ^ (std::hash<uint32_t>()(key.k) << 1);
    }
};

struct EqualKey {
    bool operator()(const GridKey& lhs, const GridKey& rhs) const
    {
        return lhs.i == rhs.i && lhs.j == rhs.j && lhs.k == rhs.k;
    }
};

struct AABB {
    AABB() { }

    AABB(const Points V)
    {
        for (Point p : V)
            update(p);
    }

    AABB(const Point& a, const Point& b)
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

    void voxel(uint32_t N)
    {
        step = diag().maxCoeff() / N;
        Eigen::Vector3f f = diag() / step;
        dim.x() = (int)ceilf(f.x());
        dim.y() = (int)ceilf(f.y());
        dim.z() = (int)ceilf(f.z());
    }

    Point relative_pos(const Point& point) { return (point - min).cwiseQuotient(diag()); }

    GridKey grid_key(const Point& point)
    {
        Point pos = (point - min) / step;
        return GridKey { (uint32_t)pos.x(), (uint32_t)pos.y(), (uint32_t)pos.z() };
    }

    Point center() { return 0.5f * (max + min); }

    AABB intersection(const AABB& other)
    {
        AABB result = *this;
        result.min = result.min.cwiseMax(other.min);
        result.max = result.max.cwiseMin(other.max);
        return result;
    }

    bool intersects(const AABB& other) { return !intersection(other).is_empty(); }

    bool is_empty() const { return (max.array() < min.array()).any(); }

    bool contains(const Point& p)
    {
        return p.x() >= min.x() && p.x() <= max.x() && p.y() >= min.y() && p.y() <= max.y()
            && p.z() >= min.z() && p.z() <= max.z();
    }

    friend std::ostream& operator<<(std::ostream& os, const AABB& bb)
    {
        os << "[";
        os << "min=[" << bb.min.x() << "," << bb.min.y() << "," << bb.min.z() << "], ";
        os << "max=[" << bb.max.x() << "," << bb.max.y() << "," << bb.max.z() << "], ";
        os << "dim=[" << bb.dim.x() << "," << bb.dim.y() << "," << bb.dim.z() << "], ";
        os << "radius=" << bb.step / 2.0;
        os << "]";
        return os;
    }

    Point min = Point::Constant(std::numeric_limits<float>::infinity());
    Point max = Point::Constant(-std::numeric_limits<float>::infinity());

    // Helper
    float step = 1.0;
    Eigen::Vector3i dim = Vector3i { 1, 1, 1 };
};

struct Mesh {
    void reset()
    {
        V.clear();
        C.clear();
        // N.clear();
        F.clear();
    }

    void dump()
    {
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
            for (uint32_t f : mesh.F[i])
                os << " " << f + 1; // face index list start from 1 in obj format
            os << std::endl;
        }

        return os;
    }

    bool load(const char* filename);
    bool save(const char* filename);
    void snap(float e, float t);
    void clean(Mask mask);

    Mesh grid_sample(uint32_t N);
    MeshList fast_segment(uint32_t N, size_t outliers_threshold);
    void merge(MeshList cluster);

    Mesh grid_mesh(uint32_t N);
    void simplify(float ratio); // quadratic error metrics (QEM) ?

private:
    GridIndex grid_index(AABB& aabb);

    bool loadOBJ(const char* filename);
    bool saveOBJ(const char* filename);
    bool loadPLY(const char* filename);

public:
    Points V;
    Colors C;
    // Normals N;
    Faces F;
};

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
//     std::string name = "Material"; // name

//     Color Ka { 1.0, 1.0, 1.0 }; // Ambient color
//     Color Kd { 0.0, 0.0, 0.0 }; // Diffuse color
//     Color Ks { 0.5, 0.5, 0.5 }; // Specular color
//     Color Ke { 0.0, 0.0, 0.0 }; // Emit light color

//     float Ni = 1.45; // Density
//     float d = 1.0; // Transparency
//     float Ns = 250.0; // Shininess
//     int illum = 2; // Illumination model
// };
