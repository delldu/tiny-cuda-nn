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

using Vertex = Eigen::Vector3f;
using Normal = Eigen::Vector3f;
using UV = Eigen::Vector2f; // uv coordinate
using Face = std::vector<size_t>;
using Point = Eigen::Vector3f;
using Points = std::vector<Point>;


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
        VC.clear();
        VN.clear();
        VT.clear();
        F.clear();
        FT.clear();
        FN.clear();        
    }

    friend std::ostream& operator<<(std::ostream& os, const Mesh& mesh)
    {
        os << std::fixed;
        os << "# Vertext: " << std::fixed << mesh.V.size() << ", Face: " << mesh.F.size() << std::endl;

        // Vertex Command -- v x y z or v x y z r g b
        bool has_color = (mesh.V.size() == mesh.VC.size());
        for (size_t i = 0; i < mesh.V.size(); i++) {
            os << "v " << mesh.V[i].x() << " " << mesh.V[i].y() << " " << mesh.V[i].z();
            if (has_color)
                os <<  " " << mesh.VC[i].r << " " << mesh.VC[i].g << " " << mesh.VC[i].b;
            os << std::endl;
        }

        // Vertex Normal Command -- vn x y z
        for (Normal vn : mesh.VN) {
            os << "vn " << vn.x() << " " << vn.y() << " " << vn.z() << std::endl;
        }

        // Vertex Texture Command -- vt u v
        for (UV uv : mesh.VT) {
            os << "vt " << uv.x() << " " << uv.y() << " " << std::endl;
        }

        size_t old_group = 0;
        bool has_face_group = (mesh.FG.size() == mesh.F.size());
        bool has_face_texture = (mesh.FT.size() == mesh.F.size());
        bool has_face_normal = (mesh.FT.size() == mesh.F.size());
        // All face index list start from 1
        for (size_t i = 0; i < mesh.F.size(); i++) {
            if (has_face_group && mesh.FG[i] != old_group) { // new group start
                os << "g " << mesh.FG[i] << std::endl;
                old_group = mesh.FG[i];
            }
            os << "f";
            for (size_t j = 0; j < mesh.F[i].size(); j++) {
                os << " " << mesh.F[i][j] + 1;
                if (has_face_normal) {
                    if (has_face_texture)
                        os << "/" << mesh.FT[i][j] + 1 << "/" << mesh.FN[i][j] + 1;
                    else
                        os << "//" << mesh.FN[i][j] + 1;
                } else if (has_face_texture) {
                    os << "/" << mesh.FT[i][j] + 1;
                }
            }
            os << std::endl;
        }

        return os;
    }

    bool loadOBJ(char* filename)
    {
        int count, i[3];
        float x[6];
        char line[2048], command[2048], word[2048];
        size_t group = 0;

        Face one_face_index;
        Face one_face_texture;
        Face one_face_normal;

        FILE* fp = fopen(filename, "r");
        if (fp == NULL) {
            fprintf(stderr, "Error: %s could not be opened for reading ...", filename);
            return false;
        }

        reset();

        while (fgets(line, sizeof(line), fp) != NULL) {
            if (sscanf(line, "%s", command) != 1)
                continue; // skip

            if (std::string(command) == "g") {
                group++;
                continue;
            }

            char* l = &line[strlen(command)]; // left line

            if (std::string(command) == "v") {
                count = sscanf(l, "%f %f %f %f %f %f\n", &x[0], &x[1], &x[2], &x[3], &x[4], &x[5]);
                if (count != 3 && count != 6)
                    continue; // skip invalid vertex

                V.push_back(Vertex { x[0], x[1], x[2] });
                if (count == 6)
                     VC.push_back(RGB{x[3], x[4], x[5]});
                continue;
            }

            if (std::string(command) == "vn") {
                count = sscanf(l, "%f %f %f\n", &x[0], &x[1], &x[2]);
                if (count == 3) // skip invalid normals
                    VN.push_back(Normal{x[0], x[1], x[2]});
                continue;
            }

            if (std::string(command) == "vt") {
                count = sscanf(l, "%f %f %f\n", &x[0], &x[1], &x[2]);
                if (count == 2 || count == 3) // skip invalid uv coordinates
                    VT.push_back(UV{x[0], x[1]}); // ignore vt - w
                continue;
            }

            if (std::string(command) == "f") {
                // f 1/1/1 2/2/2 4/4/3 3/3/4 -- face/texture/normal index
                int offset;
                one_face_index.clear();
                one_face_texture.clear();
                one_face_normal.clear();

                while (sscanf(l, "%s%n", word, &offset) == 1) {
                    l += offset;
                    if (sscanf(word, "%d/%d/%d", &i[0], &i[1], &i[2]) == 3) {
                        one_face_index.push_back(i[0] - 1);
                        one_face_texture.push_back(i[1] - 1);
                        one_face_normal.push_back(i[2] - 1);
                    } else if (sscanf(word, "%d/%d", &i[0], &i[1]) == 2) {
                        one_face_index.push_back(i[0] - 1);
                        one_face_texture.push_back(i[1] - 1);
                    } else if (sscanf(word, "%d//%d", &i[0], &i[1]) == 2) {
                        one_face_index.push_back(i[0] - 1);
                        one_face_normal.push_back(i[1] - 1);
                    } else if (sscanf(word, "%d", &i[0]) == 1) {
                        one_face_index.push_back(i[0] - 1);
                    } else {
                        ; // skip invalid face index ...
                    }
                }
                if (one_face_index.size() >= 3) {
                    F.push_back(one_face_index);
                    FG.push_back(group);
                }
                if (one_face_texture.size() >= 3 && one_face_texture.size() == one_face_index.size()) {
                    FT.push_back(one_face_texture);
                }
                if (one_face_normal.size() >= 3 && one_face_normal.size() == one_face_index.size()) {
                    FN.push_back(one_face_normal);
                }
                continue;
            }
        }
        fclose(fp);

        if (group == 0) // group command ?
            FG.clear();

        return true;
    }

    bool saveOBJ(const char* filename)
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

    bool loadPLY(char* filename)
    {
        int count;
        float x[3];
        char line[2048], command[2048], word[2048];
        size_t n, n_vertex, n_face;

        int offset;
        Face one_face_index;

        FILE* fp = fopen(filename, "r");
        if (fp == NULL) {
            fprintf(stderr, "Error: %s could not be opened for reading ...", filename);
            return false;
        }

        // parsing ply header ...
        bool is_ply_magic = false;
        bool binary_format = false;
        bool has_color = false;

        while (fgets(line, sizeof(line), fp) != NULL) {
            if (sscanf(line, "%s", command) != 1)
                continue; // skip
            if (std::string(command) == "end_header")
                break;

            if (std::string(command) == "ply")
                is_ply_magic = true;

            if (std::string(command) == "format") {
                binary_format = (strstr(line, "binary") != NULL);
            }

            if (std::string(command) == "element") {
                char* l = &line[strlen(command)]; // left line
                count = sscanf(l, "%s %ld\n", word, &n);
                if (count != 2)
                    continue; // skip invalid vertex/face ?
                if (std::string(word) == "vertex")
                    n_vertex = n;
                if (std::string(word) == "face")
                    n_face = n;
            }

            if (std::string(command) == "property") {
                if (strstr(line, "red") != NULL)
                    has_color = true;
            }
        }

        if (! is_ply_magic) {
            fprintf(stderr, "Error: %s is not ply file...", filename);
            fclose(fp);
            return false;
        }

        if (binary_format) {
            // Reading vertex ...
            for (size_t k = 0; k < n_vertex; k++) {
                fread(&x[0], sizeof(float), 1, fp);
                fread(&x[1], sizeof(float), 1, fp);
                fread(&x[2], sizeof(float), 1, fp);
                V.push_back(Vertex{x[0], x[1], x[2]});
                if (has_color) {
                    uint8_t r, g, b;
                    fread(&r, sizeof(uint8_t), 1, fp);
                    fread(&g, sizeof(uint8_t), 1, fp);
                    fread(&b, sizeof(uint8_t), 1, fp);
                    VC.push_back(RGB{(float)r/255.0f, (float)g/255.0f, (float)b/255.0f});
                }
            }
            // Reading face ...
            for (size_t k = 0; k < n_face; k++) {
                one_face_index.clear();
                int j, f, f_n;
                fread(&f_n, sizeof(int), 1, fp);
                for (j = 0; j < f_n; j++) {
                    fread(&f, sizeof(int), 1, fp);
                    one_face_index.push_back(f);
                }
                if (one_face_index.size() >= 3) {
                    F.push_back(one_face_index);
                }
            } // end reading face
        } else { // ascii format
            // Reading vertex ...
            for (size_t k = 0; k < n_vertex; k++) {
                if (fgets(line, sizeof(line), fp) == NULL)
                    continue;

                if (has_color) {
                    int r, g, b;
                    count = sscanf(line, "%f %f %f %d %d %d\n", &x[0], &x[1], &x[2], &r, &g, &b);
                    if (count == 6) {
                        V.push_back(Vertex{x[0], x[1], x[2]});
                        VC.push_back(RGB{(float)r/255.0f, (float)g/255.0f, (float)b/255.0f});
                    }
                } else {
                    count = sscanf(line, "%f %f %f\n", &x[0], &x[1], &x[2]);
                    if (count == 3)
                        V.push_back(Vertex{x[0], x[1], x[2]});
                }
            }  // end reading vertex

            // Reading face ...
            for (size_t k = 0; k < n_face; k++) {
                if (fgets(line, sizeof(line), fp) == NULL)
                    continue;

                char *l = line;
                if (sscanf(l, "%s%n", word, &offset) != 1)
                    continue;
                l += offset;

                n = atoi(word);
                one_face_index.clear();
                for (size_t j = 0; j < n; j++) {
                    if (sscanf(l, "%s%n", word, &offset) != 1)
                        continue;
                    l += offset;
                    one_face_index.push_back(atoi(word));
                }
                if (one_face_index.size() >= 3)
                    F.push_back(one_face_index);
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
        if (VC.size() == V.size()) {
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

            if (VC.size() == V.size()) {
                uint8_t r, g, b;
                r = (uint8_t)(VC[i].r * 255.0);
                g = (uint8_t)(VC[i].g * 255.0);
                b = (uint8_t)(VC[i].b * 255.0);

                fwrite(&r, sizeof(char), 1, fp);
                fwrite(&g, sizeof(char), 1, fp);
                fwrite(&b, sizeof(char), 1, fp);
            }
        }

        // write Face
        for (Face face : F) {
            uint8_t n = (uint8_t)face.size();
            fwrite(&n, sizeof(uint8_t), 1, fp);
            for (size_t f : face) {
                uint32_t index = (uint32_t)f;
                fwrite(&index, sizeof(uint32_t), 1, fp);
            }
        }

        fclose(fp);
        return true;
    }

public:
    std::vector<Vertex> V; // v x y z -- Vertex Command
    std::vector<RGB> VC; // Vertex Color
    std::vector<Normal> VN; // vn x y z -- Vertex Normal Command
    std::vector<UV> VT; // vt u v [w] -- Vertex Texture Command

    std::vector<Face> F;    // Face Index List
    std::vector<size_t> FG; // Face Group
    std::vector<Face> FT;   // Face Texture Index List
    std::vector<Face> FN;   // Face Normal Index List
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

    void normalized() { n = n.normalized(); }

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
};

void test_plane()
{
    Points points;

    Point p1 = Point { 0.0, 0.0, 0.0 };
    Point p2 = Point { 1.0, 0.0, 0.0 };
    Point p3 = Point { 0.0, 1.0, 0.0 };
    Point p4 = Point { 0.0, 0.0, 1.0 };

    points.push_back(p1);
    points.push_back(p2);
    points.push_back(p3);
    // points.push_back(Eigen::Vector3f{0.01, 0.01, 0.01});

    Plane plane(points);
    std::cout << plane << std::endl;

    Point p = Point { 0.000001, 0.000001, 0.000001 };
    std::cout << p << " on plane (epision = default) ? " << plane.contains(p) << std::endl;
    std::cout << p << " on plane (epision = 0.1f)? " << plane.contains(p, 0.1f) << std::endl;

    BoundingBox aabb;
    aabb.update(p1);
    aabb.update(p2);
    aabb.update(p3);
    aabb.update(p4);
    std::cout << "aabb:" << aabb << std::endl;

    Plane B(plane.o - Point { EPISON / 2.0, EPISON / 2.0, EPISON / 2.0 },
        plane.n + Point { EPISON / 2.0, EPISON / 2.0, EPISON / 2.0 });

    std::cout << B << std::endl;
    std::cout << "Plane is same as B ?" << plane.same(B) << std::endl;

    Mesh mesh;
    mesh.V.push_back(p1);
    mesh.V.push_back(p2);
    mesh.V.push_back(p3);
    mesh.V.push_back(p4);

    Face face;
    face.push_back(0);
    face.push_back(1);
    face.push_back(2);
    mesh.F.push_back(face);

    face.clear();
    face.push_back(0);
    face.push_back(2);
    face.push_back(3);
    mesh.F.push_back(face);

    // mesh.loadOBJ((char *)"001.obj");

    mesh.savePLY("test.ply");
    mesh.saveOBJ("test.obj");

    Material material;
    material.save("test.mtl");
    // std::cout << material << std::endl;

}
