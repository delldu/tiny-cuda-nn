/************************************************************************************
***
***     Copyright 2023 Dell Du(18588220928@163.com), All Rights Reserved.
***
***     File Author: Dell, 2023年 03月 07日 星期二 18:29:34 CST
***
************************************************************************************/
#pragma once

#include <algorithm>
#include <queue>
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
struct GridCell;

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
using GridIndex = std::unordered_map<GridKey, IndexList, HashFunc, EqualKey>;
// using GridPoint = std::unordered_map<GridKey, Point, HashFunc, EqualKey>;
// using GridColor = std::unordered_map<GridKey, Color, HashFunc, EqualKey>;
// using GridDensity = std::unordered_map<GridKey, float, HashFunc, EqualKey>;
using GridNormal = std::unordered_map<GridKey, GridCell, HashFunc, EqualKey>;

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

    Point project(const Point p) const
    {
        float t = n.dot(p - o);
        return p - t * n;
    }

    float distance(const Point p) const
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
    Color()
    {
    }

    Color(float r, float g, float b)
        : r(r)
        , g(g)
        , b(b)
    {
    }

    friend std::ostream& operator<<(std::ostream& os, const Color& color)
    {
        os << std::fixed << color.r << " " << color.g << " " << color.b;
        return os;
    }

public:
    float r, g, b;
};

struct GridCell {
    GridCell() { }

    GridCell(Point p, Color c, float d)
        : point(p)
        , color(c)
        , density(d)
    {
    }

public:
    Point point;
    Color color;
    float density;
};

struct GridKey {
    GridKey() { }

    GridKey(uint32_t f, uint32_t s, uint32_t t)
        : i(f)
        , j(s)
        , k(t)
    {
    }

    bool operator==(const GridKey& b)
    {
        return i == b.i && j == b.j && k == b.k;
    }

    bool operator<(const GridKey& b) const
    {
        if (i != b.i)
            return i < b.i;
        if (j != b.j)
            return j < b.j;
        return k < b.k;
    }

    friend std::ostream& operator<<(std::ostream& os, const GridKey& key)
    {
        os << std::fixed << "(" << key.i << ", " << key.j << ", " << key.k << ")";
        return os;
    }

    std::set<GridKey> nb3x3x3()
    {
        std::set<GridKey> s;
        for (int ii = -1; ii <= 1; ii++) {
            for (int jj = -1; jj <= 1; jj++) {
                for (int kk = -1; kk <= 1; kk++) {
                    if (!(ii == 0 && jj == 0 && kk == 0)) // skip self
                        s.insert(GridKey { i + ii, j + jj, k + kk });
                }
            }
        }
        return s;
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

    Point diag() const { return max - min; }

    void voxel(uint32_t N)
    {
        step = diag().maxCoeff() / N;
        Eigen::Vector3f f = diag() / step;
        dim.x() = (int)ceilf(f.x());
        dim.y() = (int)ceilf(f.y());
        dim.z() = (int)ceilf(f.z());
    }

    Point relative_pos(const Point& point) const { return (point - min).cwiseQuotient(diag()); }

    GridKey grid_key(const Point& point) const
    {
        Point pos = (point - min) / step; //  + Eigen::Vector3f {0.5f, 0.5f, 0.5f};
        return GridKey { (uint32_t)pos.x(), (uint32_t)pos.y(), (uint32_t)pos.z() };
    }

    Point key_point(GridKey key) const
    {
        return Point { min.x() + key.i * step, min.y() + key.j * step, min.z() + key.k * step };
    }

    Point center() const { return 0.5f * (max + min); }

    AABB intersection(const AABB& other) const
    {
        AABB result = *this;
        result.min = result.min.cwiseMax(other.min);
        result.max = result.max.cwiseMin(other.max);
        return result;
    }

    bool intersects(const AABB& other) { return !intersection(other).is_empty(); }

    bool is_empty() const { return (max.array() < min.array()).any(); }

    bool contains(const Point& p) const
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
    float step = 1.0f;
    Eigen::Matrix<uint32_t, 3, 1> dim { 1, 1, 1 };
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

    MeshList segment(uint32_t N, size_t outliers_threshold);
    void merge(MeshList cluster);

    Mesh grid_mesh(uint32_t N);
    void simplify(float ratio); // quadratic error metrics (QEM) ?

    GridIndex grid_index(const AABB& aabb);
    // GridNormal grid_normal(const GridIndex &gi);
    GridNormal grid_normal(GridIndex gi);

private:
    // GridColor grid_color(const GridIndex &gi);
    // GridDensity grid_density(const GridIndex &gi);

    void emit_triangle(int has_color, float v1[], float v2[], float v3[], float c1[], float c2[], float c3[])
    {
        static uint32_t n_triangles = 0;
        // std::cout << "---- n_triangles ---" << n_triangles << std::endl;

        Face new_face;

        Point p1 { v1[0], v1[1], v1[2] };
        Point p2 { v2[0], v2[1], v2[2] };
        Point p3 { v3[0], v3[1], v3[2] };
        V.push_back(p1);
        V.push_back(p2);
        V.push_back(p3);

        new_face = Face { n_triangles, n_triangles + 1, n_triangles + 2 };
        F.push_back(new_face);

        if (has_color) {
            Color color1 { c1[0], c1[1], c1[2] };
            Color color2 { c2[0], c2[1], c2[2] };
            Color color3 { c3[0], c3[1], c3[2] };
            C.push_back(color1);
            C.push_back(color2);
            C.push_back(color3);
        }

        n_triangles += 3;
    };
    int cube_mc(int has_color, float cell_point[], float cell_color[], float cell_density[], float borderval);

    bool loadOBJ(const char* filename);
    bool saveOBJ(const char* filename);
    bool loadPLY(const char* filename);

public:
    Points V;
    Colors C;
    // Normals N;
    Faces F;
};

class BitCube {
public:
    BitCube(size_t nx, size_t ny, size_t nz)
    {
        resize(nx, ny, nz);

        // init with zeros
        std::fill_n(m_data.begin(), m_data.size(), 0);
    }

    friend std::ostream& operator<<(std::ostream& os, const BitCube& bc)
    {
        os << "[";
        os << "dim=[" << bc.dim.x() << "," << bc.dim.y() << "," << bc.dim.z() << "], ";
        os << "size=" << bc.size();
        os << "]";
        return os;
    }

    inline void xyz_pos(uint32_t x, uint32_t y, uint32_t z, size_t* pos)
    {
        *pos = x * dim.y() * dim.z() + y * dim.z() + z;
    }

    inline void pos_xyz(size_t pos, uint32_t* x, uint32_t* y, uint32_t* z)
    {
        *z = pos % dim.z();
        pos /= dim.z();
        *y = pos % dim.y();
        pos /= dim.y();
        *x = pos;
    }

    inline bool get_pos(size_t pos) const
    {
        if (pos >= size())
            return false;

        size_t byte_loc = pos / 8;
        size_t offset = pos % 8;

        return (m_data[byte_loc] >> offset) & 0x1;
    }

    inline void set_pos(size_t pos, bool value = true)
    {
        if (pos >= size())
            return;

        size_t byte_loc = pos / 8;
        uint8_t offset = pos % 8;
        uint8_t bitfield = uint8_t(1 << offset);

        if (value) {
            m_data[byte_loc] |= bitfield;
        } else {
            m_data[byte_loc] &= (~bitfield);
        }
    }

    void resize(size_t nx, size_t ny, size_t nz)
    {
        dim.x() = nx;
        dim.y() = ny;
        dim.z() = nz;
        size_t nbits = nx * ny * nz;
        size_t num_bytes = (nbits < 8) ? 1 : 1 + (nbits - 1) / 8;

        m_data.resize(num_bytes);
    }

    size_t count() const
    {
        size_t c = 0;
        for (size_t pos = 0; pos < size(); pos++)
            c += get_pos(pos) ? 1 : 0;

        return c;
    }

    bool empty() const
    {
        for (size_t i = 0; i < m_data.size(); i++) {
            if (m_data[i] != 0)
                return false;
        }

        return true;
    }

    void clear()
    {
        std::fill_n(m_data.begin(), m_data.size(), 0);
    }

    void setall(bool value)
    {
        std::fill_n(m_data.begin(), m_data.size(), value ? 0xff : 0);
    }

    bool get(uint32_t x, uint32_t y, uint32_t z)
    {
        size_t pos;
        xyz_pos(x, y, z, &pos);
        return get_pos(pos);
    }

    // bool operator[](size_t pos) const
    // {
    //     return get_pos(pos);
    // }

    // bool operator()(const uint32_t x, const uint32_t y, const uint32_t z)
    // {
    //     return get(x, y, z);
    // }

    void set(uint32_t x, uint32_t y, uint32_t z, bool value)
    {
        size_t pos;
        xyz_pos(x, y, z, &pos);
        set_pos(pos, value);
    }

    size_t size() const
    {
        return dim.x() * dim.y() * dim.z();
    }

    std::vector<std::set<size_t>> segment();


private:
    std::vector<size_t> find_valid_nb3x3x3(size_t pos);

    const uint8_t* data() const
    {
        return m_data.data();
    }

    uint8_t* data()
    {
        return m_data.data();
    }

    Eigen::Matrix<uint32_t, 3, 1> dim { 1, 1, 1 };
    std::vector<uint8_t> m_data;
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
