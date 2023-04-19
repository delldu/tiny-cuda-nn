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
#include <vector>
using namespace std;

#include <Eigen/Dense> // Version 3.4.9, eigen.tgz under dependencies
using namespace Eigen;

#define EPISON 5.0e-4f
#define F(t) (float)atof(t)

using Point = Eigen::Vector3f;
using Normal = Eigen::Vector3f;
using UV = Eigen::Vector2f; // uv coordinate
using Face = std::vector<size_t>;
using Points = std::vector<Point>;

int get_token(char* buf, char deli, int maxcnt, char* tv[])
{
    int n = 0;

    if (!buf)
        return 0;
    tv[n] = buf;
    while (*buf && (n + 1) < maxcnt) {
        if (*buf == deli) {
            *buf = '\0';
            ++buf;
            while (*buf == deli)
                ++buf;
            tv[++n] = buf;
        }
        ++buf;
    }
    return (n + 1);
}

bool is_command(char* s, const char* cmd)
{
    return strncmp(s, cmd, strlen(cmd)) == 0;
}

std::ostream& operator<<(std::ostream& os, const Point& point)
{
    os << std::fixed << "(" << point.x() << ", " << point.y() << ", " << point.z() << ")";
    return os;
}

struct RGB {
    friend std::ostream& operator<<(std::ostream& os, const RGB& color)
    {
        os << std::fixed << color.r << " " << color.g << " " << color.b;
        return os;
    }

public:
    float r, g, b;
};

struct Material {
    friend std::ostream& operator<<(std::ostream& os, const Material& material)
    {
        os << std::fixed;
        os << "newmtl " << material.name << std::endl;
        os << "Ns " << material.Ns << std::endl;
        os << "Ka " << material.Ka << std::endl;
        os << "Kd " << material.Kd << std::endl;
        os << "Ks " << material.Ks << std::endl;
        os << "Ke " << material.Ke << std::endl;
        os << "Ni " << material.Ni << std::endl;
        os << "d " << material.d << std::endl;
        os << "illum " << material.illum << std::endl;
        return os;
    }

    bool save(const char* filename)
    {
        ofstream out(filename);
        if (!out.good()) {
            std::cerr << "Error: create file '" << filename << "'" << std::endl;
            return false;
        }
        out << *this;
        out.close();
        return true;
    }

public:
    std::string name = "Material"; // Texture name

    RGB Ka { 1.0, 1.0, 1.0 }; // Ambient color
    RGB Kd { 0.0, 0.0, 0.0 }; // Diffuse color
    RGB Ks { 0.5, 0.5, 0.5 }; // Specular color
    RGB Ke { 0.0, 0.0, 0.0 }; // Emit light color

    float Ni = 1.45; // Density
    float d = 1.0; // Transparency
    float Ns = 250.0; // Shininess
    int illum = 2; // Illumination model
};

struct Mesh {
    void reset()
    {
        V.clear();
        C.clear();
        F.clear();
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
                V.push_back(Point { F(tv[1]), F(tv[2]), F(tv[3]) });
                if (count == 7)
                    C.push_back(RGB { F(tv[4]), F(tv[5]), F(tv[6]) });
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

    bool loadPLY(const char* filename)
    {
        int count;
        char line[2048], *tv[256];
        float x[3];
        uint8_t rgb[3];
        size_t n_vertex, n_face;
        Face one_face_index;

        FILE* fp = fopen(filename, "r");
        if (fp == NULL) {
            fprintf(stderr, "Error: %s could not be opened for reading ...", filename);
            return false;
        }

        reset();

        // parsing ply header ...
        bool has_ply_magic = false;
        bool binary_format = false;
        bool has_color = false;

        n_vertex = n_face = 0;
        while (fgets(line, sizeof(line), fp) != NULL) {
            count = get_token(line, ' ', 256, tv);

            if (is_command(tv[0], "end_header"))
                break;

            if (is_command(tv[0], "ply"))
                has_ply_magic = true;

            if (is_command(tv[0], "format"))
                binary_format = (strstr(tv[1], "binary") != NULL);

            if (is_command(tv[0], "element")) {
                if (is_command(tv[1], "vertex"))
                    n_vertex = atoi(tv[2]);
                else if (is_command(tv[1], "face"))
                    n_face = atoi(tv[2]);
            }

            if (is_command(tv[0], "property") && is_command(tv[2], "red"))
                has_color = true;
        }

        if (!has_ply_magic) {
            fprintf(stderr, "Error: %s is not ply file...", filename);
            fclose(fp);
            return false;
        }

        if (binary_format) {
            // vertex ...
            while (n_vertex > 0 && !feof(fp)) {
                fread(x, sizeof(float), 3, fp);
                V.push_back(Point { x[0], x[1], x[2] });
                if (has_color) {
                    fread(rgb, sizeof(uint8_t), 3, fp);
                    C.push_back(RGB { (float)rgb[0] / 255.0f, (float)rgb[1] / 255.0f, (float)rgb[2] / 255.0f });
                }
                n_vertex--;
            }

            // face ...
            while (n_face > 0 && !feof(fp)) {
                one_face_index.clear();
                int j, f;
                uint8_t f_n; // ply format !!!
                fread(&f_n, sizeof(uint8_t), 1, fp);
                for (j = 0; j < (int)f_n; j++) {
                    fread(&f, sizeof(int), 1, fp);
                    one_face_index.push_back(f);
                }
                if (one_face_index.size() >= 3)
                    F.push_back(one_face_index);
                n_face--;
            } // end reading face
        } else { // ascii format
            // vertex ...
            while (n_vertex > 0 && fgets(line, sizeof(line), fp) != NULL) {
                count = get_token(line, ' ', 8, tv);
                if (has_color) {
                    if (count == 7) {
                        V.push_back(Point { F(tv[1]), F(tv[2]), F(tv[3]) });
                        C.push_back(RGB { F(tv[4]) / 255.0f, F(tv[5]) / 255.0f, F(tv[6]) / 255.0f });
                    }
                } else {
                    if (count == 4)
                        V.push_back(Point { F(tv[1]), F(tv[2]), F(tv[3]) });
                }
                n_vertex--;
            } // end reading vertex

            // face ...
            while (n_face > 0 && fgets(line, sizeof(line), fp) != NULL) {
                count = get_token(line, ' ', 256, tv);
                one_face_index.clear();
                for (int j = 1; j < atoi(tv[0]); j++)
                    one_face_index.push_back(atoi(tv[j]));
                if (one_face_index.size() >= 3)
                    F.push_back(one_face_index);
                n_face--;
            } // end reading face
        }

        fclose(fp);
        return true;
    }

    bool savePLY(const char* filename)
    {
        FILE* fp = fopen(filename, "wb");
        if (fp == NULL) {
            fprintf(stderr, "Error: %s could not be opened for writing ...", filename);
            return false;
        }

        /*write header */
        fprintf(fp, "ply\n");
        fprintf(fp, "format binary_little_endian 1.0\n");
        fprintf(fp, "element vertex %ld\n", V.size());
        fprintf(fp, "property float x\n");
        fprintf(fp, "property float y\n");
        fprintf(fp, "property float z\n");
        if (C.size() == V.size()) {
            fprintf(fp, "property uchar red\n");
            fprintf(fp, "property uchar green\n");
            fprintf(fp, "property uchar blue\n");
        }
        fprintf(fp, "element face %ld\n", F.size());
        fprintf(fp, "property list uchar int vertex_indices\n");
        fprintf(fp, "end_header\n");

        // write vertex
        for (size_t i = 0; i < V.size(); i++) {
            fwrite(&V[i].x(), sizeof(float), 1, fp);
            fwrite(&V[i].y(), sizeof(float), 1, fp);
            fwrite(&V[i].z(), sizeof(float), 1, fp);

            if (C.size() == V.size()) {
                uint8_t rgb[3];
                rgb[0] = (uint8_t)(C[i].r * 255.0);
                rgb[1] = (uint8_t)(C[i].g * 255.0);
                rgb[2] = (uint8_t)(C[i].b * 255.0);
                fwrite(rgb, sizeof(char), 3, fp);
            }
        }

        // write Face
        for (Face face : F) {
            uint8_t n = (uint8_t)face.size();
            fwrite(&n, sizeof(uint8_t), 1, fp);
            for (size_t f : face) {
                int index = (int)f;
                fwrite(&index, sizeof(int), 1, fp);
            }
        }

        fclose(fp);
        return true;
    }

public:
    std::vector<Point> V; // v x y z -- Vertex Command
    std::vector<RGB> C; // Vertex Color

    std::vector<Face> F; // Face Index List
};

struct Plane {
    Plane(Point o, Normal n)
        : o(o)
        , n(n)
    {
        n = n.normalized();
    }

    // p on plane ?
    bool contains(const Point p, float e = EPISON) const
    {
        return fabs(n.dot(p - o)) < e; // (p - o) _|_ n
    }

    Point project(const Point p)
    {
        float t = n.dot(p - o);
        return o + t * n;
    }

    // fitting ...
    Plane(const Points points)
    {
        size_t size = points.size();
        if (size >= 3) {
            Eigen::MatrixXf A = Eigen::MatrixXf::Ones(size, 3);
            Eigen::VectorXf b = Eigen::VectorXf::Zero(size);

            for (size_t i = 0; i < size; i++) {
                A(i, 0) = points[i].x();
                A(i, 1) = points[i].y();
                b(i) = points[i].z();
            }
            Eigen::VectorXf x = A.colPivHouseholderQr().solve(b);
            n.x() = x[0];
            n.y() = x[1];
            n.z() = -1.0f;

            x = A.colwise().mean();
            o.x() = x[0];
            o.y() = x[1];
            o.z() = b.colwise().mean().x();
        }
        n = n.normalized();
    }

    bool same(const Plane& other, float e = EPISON)
    {
        return this->contains(other.o, e) && other.contains(this->o, e)
            && (fabs(this->n.dot(other.n)) > 1.0f - e);
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
};

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

    Point relative_pos(const Point& pos) { return (pos - min).cwiseQuotient(diag()); }

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
        os << "max=[" << bb.max.x() << "," << bb.max.y() << "," << bb.max.z() << "]";
        os << "]";
        return os;
    }

    Point min = Point::Constant(std::numeric_limits<float>::infinity());
    Point max = Point::Constant(-std::numeric_limits<float>::infinity());

    // Helper
    float radius;
    Eigen::Vector3i dim;
};

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

    // Plane B(plane.o - Point { EPISON / 2.0, EPISON / 2.0, EPISON / 2.0 },
    //     plane.n + Point { EPISON / 2.0, EPISON / 2.0, EPISON / 2.0 });

    // std::cout << B << std::endl;
    // std::cout << "Plane is same as B ?" << plane.same(B) << std::endl;

    Mesh mesh;
 
    mesh.loadOBJ("001.obj");
    mesh.savePLY("001_test.ply");
    mesh.saveOBJ("001_test.obj");

    mesh.loadPLY("pc.ply");
    mesh.savePLY("pc_test.ply");
    mesh.saveOBJ("pc_test.obj");

    // Material material;
    // material.save("test.mtl");
}
