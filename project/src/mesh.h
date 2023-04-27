/************************************************************************************
***
***     Copyright 2023 Dell Du(18588220928@163.com), All Rights Reserved.
***
***     File Author: Dell, 2023年 03月 07日 星期二 18:29:34 CST
***
************************************************************************************/
#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
using namespace std;

#include <Eigen/Dense> // Version 3.4.9, eigen.tgz under dependencies
using namespace Eigen;

#include "tinylog.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tinyobj.h"

#define TINYPLY_IMPLEMENTATION
#include "tinyply.h"
// using namespace tinyply;

using Point = Eigen::Vector3f;
using Normal = Eigen::Vector3f;
// using UVMap = Eigen::Vector2f;
using Face = std::vector<size_t>;
// using Index = std::vector<size_t>;
using Points = std::vector<Point>;

std::ostream& operator<<(std::ostream& os, const Point& point)
{
    os << std::fixed << "(" << point.x() << ", " << point.y() << ", " << point.z() << ")";
    return os;
}

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

using GridColor = std::unordered_map<GridKey, Color>;
using GridDensity = std::unordered_map<GridKey, float>;
using GridIndex = std::unordered_map<GridKey, std::vector<uint32_t>, HashFunc, EqualKey>;

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
            for (size_t f : mesh.F[i])
                os << " " << f + 1; // face index list start from 1 in obj format
            os << std::endl;
        }

        return os;
    }

    bool loadOBJ(const char* filename);
    bool saveOBJ(const char* filename);
    bool loadPLY(const char* filename);

public:
    std::vector<Point> V;
    std::vector<Color> C;
    // std::vector<Normal> N;
    std::vector<Face> F;
};

bool Mesh::loadOBJ(const char* filename)
{
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;

    const char* basepath = NULL;
    bool triangulate = true;

    reset();
    tlog::info() << "Loading " << filename << " ... ";

    std::string warn, err;
    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filename, basepath, triangulate);
    if (!warn.empty()) {
        std::cout << "WARN: " << warn << std::endl;
    }

    if (!err.empty()) {
        tlog::error() << err;
        return false;
    }

    if (!ret) {
        tlog::error() << "Failed to load " << filename;
        return false;
    }

    for (size_t v = 0; v < attrib.vertices.size() / 3; v++) {
        V.push_back(Point { attrib.vertices[3 * v + 0], attrib.vertices[3 * v + 1], attrib.vertices[3 * v + 2] });
    }

    for (size_t v = 0; v < attrib.colors.size() / 3; v++) {
        C.push_back(Color { attrib.colors[3 * v + 0], attrib.colors[3 * v + 1], attrib.colors[3 * v + 2] });
    }

    // for (size_t v = 0; v < attrib.normals.size() / 3; v++) {
    //     N.push_back(Normal { attrib.normals[3 * v + 0], attrib.normals[3 * v + 1], attrib.normals[3 * v + 2] });
    // }

    // for (size_t v = 0; v < attrib.texcoords.size() / 2; v++) {
    //     UV.push_back(UVMap{attrib.texcoords[2 * v + 0], attrib.texcoords[2 * v + 1]});
    // }

    // For each shape
    Face face; // temp face
    for (size_t i = 0; i < shapes.size(); i++) {
        size_t index_offset = 0;

        // For each face
        for (size_t f = 0; f < shapes[i].mesh.num_face_vertices.size(); f++) {
            size_t fnum = shapes[i].mesh.num_face_vertices[f];
            face.clear();
            for (size_t v = 0; v < fnum; v++) {
                tinyobj::index_t idx = shapes[i].mesh.indices[index_offset + v];
                face.push_back(idx.vertex_index);
                // idx.vertex_index, idx.normal_index, idx.texcoord_index
            }
            F.push_back(face);
            index_offset += fnum;
        }
    }

    return true;
}

bool Mesh::saveOBJ(const char* filename)
{
    ofstream out(filename);
    if (!out.good()) {
        tlog::error() << "Create OBJ file '" << filename << "' for saving ...";
        return false;
    }
    out << *this;
    out.close();
    return true;
}


struct float3 { float x, y, z; };

bool Mesh::loadPLY(const char* filename)
{
    std::cout << "........................................................................\n";
    std::cout << "Now Reading: " << filename << std::endl;


    std::unique_ptr<std::istream> file_stream;

    reset();
    tlog::info() << "Loading " << filename << " ...";

    try
    {
        file_stream.reset(new std::ifstream(filename, std::ios::binary));

        if (!file_stream || file_stream->fail()) {
          tlog::error() << "Open file " << filename ;
          return false;
        }

        file_stream->seekg(0, std::ios::beg);

        tinyply::PlyFile file;
        file.parse_header(*file_stream);

        std::cout << "\t[ply_header] Type: " << (file.is_binary_file() ? "binary" : "ascii") << std::endl;
        for (const auto & c : file.get_comments()) std::cout << "\t[ply_header] Comment: " << c << std::endl;
        for (const auto & c : file.get_info()) std::cout << "\t[ply_header] Info: " << c << std::endl;

        for (const auto & e : file.get_elements())
        {
            std::cout << "\t[ply_header] element: " << e.name << " (" << e.size << ")" << std::endl;
            for (const auto & p : e.properties)
            {
                std::cout << "\t[ply_header] \tproperty: " << p.name << " (type=" << tinyply::PropertyTable[p.propertyType].str << ")";
                if (p.isList) std::cout << " (list_type=" << tinyply::PropertyTable[p.listType].str << ")";
                std::cout << std::endl;
            }
        }

        // Because most people have their own mesh types, tinyply treats parsed data as structured/typed byte buffers. 
        // See examples below on how to marry your own application-specific data structures with this one. 
        std::shared_ptr<tinyply::PlyData> vertices, normals, colors, texcoords, faces, tripstrip;

        // The header information can be used to programmatically extract properties on elements
        // known to exist in the header prior to reading the data. For brevity of this sample, properties 
        // like vertex position are hard-coded: 
        try { vertices = file.request_properties_from_element("vertex", { "x", "y", "z" }); }
        catch (const std::exception & e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

        try { normals = file.request_properties_from_element("vertex", { "nx", "ny", "nz" }); }
        catch (const std::exception & e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

        try { colors = file.request_properties_from_element("vertex", { "red", "green", "blue", "alpha" }); }
        catch (const std::exception & e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

        try { colors = file.request_properties_from_element("vertex", { "r", "g", "b", "a" }); }
        catch (const std::exception & e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

        try { texcoords = file.request_properties_from_element("vertex", { "u", "v" }); }
        catch (const std::exception & e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

        // Providing a list size hint (the last argument) is a 2x performance improvement. If you have 
        // arbitrary ply files, it is best to leave this 0. 
        try { faces = file.request_properties_from_element("face", { "vertex_indices" }, 3); }
        catch (const std::exception & e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

        // Tristrips must always be read with a 0 list size hint (unless you know exactly how many elements
        // are specifically in the file, which is unlikely); 
        try { tripstrip = file.request_properties_from_element("tristrips", { "vertex_indices" }, 0); }
        catch (const std::exception & e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

        file.read(*file_stream);

        if (vertices) {
           std::cout << "\tRead " << vertices->count  << " total vertices "<< std::endl;
           std::cout << "\ttype " << (int)vertices->t << std::endl; // type 7
        }
        if (normals) {
            std::cout << "\tRead " << normals->count   << " total vertex normals " << std::endl;
            std::cout << "\ttype " << (int)normals->t << std::endl; // type 7
        }

        if (colors) {
             std::cout << "\tRead " << colors->count << " total vertex colors " << std::endl;
             std::cout << "\ttype " << (int)colors->t << std::endl;
        }
        if (texcoords) {
            std::cout << "\tRead " << texcoords->count << " total vertex texcoords " << std::endl;
            std::cout << "\ttype " << (int)texcoords->t << std::endl;
        }
        if (faces) {
            std::cout << "\tRead " << faces->count     << " total faces (triangles) " << std::endl;
            std::cout << "\ttype " << (int)faces->t << std::endl; //type 5
            std::cout << "\tisList " << faces->isList << std::endl;

            // int *p = (int *)faces->buffer.get();
            // for (size_t i = 0; i < faces->count; i++) {
            //     // std::cout << "i: " << i << std::endl;
            //     for (int j = 0; j < 3; j++) {
            //       std::cout << p[j] << " ";
            //     }
            //     std::cout << std::endl;
            //     p += 3;
            // }
            // 3 291170 292064 292065
            // 3 291170 292065 291171
            // 3 291171 292065 292066
            // 3 291171 292066 290214
            // 3 293001 290216 292998
            // 3 292998 290216 290215
            // 3 293001 291610 290216
            // 3 293175 290322 291172
            // 3 290323 293175 291172
            // 3 292498 290324 291612

        }
        if (tripstrip) {
           std::cout << "\tRead " << (tripstrip->buffer.size_bytes() / tinyply::PropertyTable[tripstrip->t].stride) << " total indicies (tristrip) " << std::endl;
           std::cout << "\ttype " << (int)faces->t << std::endl;
        }

        // Example One: converting to your own application types
        if (vertices && vertices->t == tinyply::Type::FLOAT32) {
            // std::vector<float3> verts(vertices->count);
            // std::memcpy(verts.data(), vertices->buffer.get(), vertices->buffer.size_bytes());
            V.resize(vertices->count);
            std::memcpy(V.data()->data(), vertices->buffer.get(), vertices->buffer.size_bytes());
        }
        if (faces && (faces->t == tinyply::Type::INT32 || faces->t == tinyply::Type::INT32)) {
            Face one_face;
            int *p = (int *)faces->buffer.get();
            for (size_t i = 0; i < faces->count; i++) {
                one_face.clear();
                one_face.push_back(p[0]);
                one_face.push_back(p[1]);
                one_face.push_back(p[2]);
                F.push_back(one_face);
                p += 3;
            }
        }



        // // Example Two: converting to your own application type
        // {
        //     // std::vector<float3> verts_floats;
        //     // std::vector<double3> verts_doubles;
        //     // if (vertices->t == tinyply::Type::FLOAT32) { /* as floats ... */ }
        //     // if (vertices->t == tinyply::Type::FLOAT64) { /* as doubles ... */ }
        // }
    }
    catch (const std::exception & e)
    {
        std::cerr << "Caught tinyply exception: " << e.what() << std::endl;
    }
    return true;
}
