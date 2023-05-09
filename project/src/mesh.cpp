/************************************************************************************
***
***     Copyright 2023 Dell Du(18588220928@163.com), All Rights Reserved.
***
***     File Author: Dell, 2023年 03月 07日 星期二 18:29:34 CST
***
************************************************************************************/

#include "mesh.h"
#include "nanoflann.hpp"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tinyobj.h"

#define TINYPLY_IMPLEMENTATION
#include "tinyply.h"

#include "marching.h"

struct PlaneNormals {
    using coord_t = float; //!< The type of each coordinate

    // Must return the number of normals
    inline size_t kdtree_get_point_count() const { return data.size(); }

    // Returns the dim'th component of the idx'th point in the class:
    // Since this is inlined and the "dim" argument is typically an immediate
    // value, the
    //  "if/else's" are actually solved at compile time.
    inline float kdtree_get_pt(const size_t idx, const size_t dim) const
    {
        if (dim == 0)
            return data[idx].x();
        else if (dim == 1)
            return data[idx].y();
        else
            return data[idx].z();
    }

    // Optional bounding-box computation: return false to default to a standard
    // bbox computation loop.
    //   Return true if the BBOX was already computed by the class and returned
    //   in "bb" so it can be avoided to redo it again. Look at bb.size() to
    //   find out the expected dimensionality (e.g. 2 or 3 for point clouds)
    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /* bb */) const
    {
        return false;
    }

public:
    std::vector<Normal> data;
};

bool Mesh::loadOBJ(const char* filename)
{
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;

    const char* basepath = NULL;
    bool triangulate = true;
    bool default_vcols_fallback = false;

    reset();
    tlog::info() << "Loading " << filename << " ... ";

    std::string warn, err;
    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filename,
        basepath, triangulate, default_vcols_fallback);
    if (!warn.empty()) {
        tlog::warning() << "WARN: " << warn;
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
        V.push_back(Point { attrib.vertices[v * 3 + 0], attrib.vertices[v * 3 + 1], attrib.vertices[v * 3 + 2] });
    }

    for (size_t v = 0; v < attrib.colors.size() / 3; v++) {
        C.push_back(Color { attrib.colors[v * 3 + 0], attrib.colors[v * 3 + 1], attrib.colors[v * 3 + 2] });
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
            face.clear(); // new face
            for (size_t v = 0; v < fnum; v++) {
                tinyobj::index_t idx = shapes[i].mesh.indices[index_offset + v];
                face.push_back((uint32_t)idx.vertex_index);

                // idx.vertex_index, idx.normal_index, idx.texcoord_index
            }
            F.push_back(face);
            index_offset += fnum;
        }
    }

    tlog::info() << "Loading mesh vertext: " << V.size() << ", faces: " << F.size();

    return true;
}

bool Mesh::saveOBJ(const char* filename)
{
    tlog::info() << "Save mesh to " << filename << " ... ";

    ofstream out(filename);
    if (!out.good()) {
        tlog::error() << "Create OBJ file '" << filename << "' for saving ...";
        return false;
    }
    out << *this;
    out.close();
    return true;
}

bool Mesh::loadPLY(const char* filename)
{
    std::unique_ptr<std::istream> file_stream;

    reset();
    tlog::info() << "Load mesh from " << filename << " ...";

    try {
        file_stream.reset(new std::ifstream(filename, std::ios::binary));

        if (!file_stream || file_stream->fail()) {
            tlog::error() << "Open file " << filename;
            return false;
        }

        file_stream->seekg(0, std::ios::beg);

        tinyply::PlyFile file;
        file.parse_header(*file_stream);

        tlog::info() << "PLY file format is " << (file.is_binary_file() ? "binary" : "ascii");

        // Because most people have their own mesh types, tinyply treats parsed data as structured/typed byte buffers.
        // See examples below on how to marry your own application-specific data structures with this one.
        // std::shared_ptr<tinyply::PlyData> vertices, normals, colors, texcoords, faces, tripstrip;
        std::shared_ptr<tinyply::PlyData> vertices, colors, faces;

        // The header information can be used to programmatically extract properties on elements
        // known to exist in the header prior to reading the data. For brevity of this sample, properties
        // like vertex position are hard-coded:
        try {
            vertices = file.request_properties_from_element("vertex", { "x", "y", "z" });
        } catch (const std::exception& e) { /*std::cerr << "tinyply exception: " << e.what() << std::endl */
            ;
        }

        // try { normals = file.request_properties_from_element("vertex", { "nx", "ny", "nz" }); }
        // catch (const std::exception & e) { /* std::cerr << "tinyply exception: " << e.what() << std::endl */; }

        try {
            colors = file.request_properties_from_element("vertex", { "red", "green", "blue" });
        } catch (const std::exception& e) { /* std::cerr << "tinyply exception: " << e.what() << std::endl */
            ;
        }

        try {
            colors = file.request_properties_from_element("vertex", { "r", "g", "b" });
        } catch (const std::exception& e) { /* std::cerr << "tinyply exception: " << e.what() << std::endl */
            ;
        }

        // try { texcoords = file.request_properties_from_element("vertex", { "u", "v" }); }
        // catch (const std::exception & e) { /* std::cerr << "tinyply exception: " << e.what() << std::endl */; }

        // Providing a list size hint (the last argument) is a 2x performance improvement. If you have
        // arbitrary ply files, it is best to leave this 0.
        try {
            faces = file.request_properties_from_element("face", { "vertex_indices" }, 3);
        } catch (const std::exception& e) { /* std::cerr << "tinyply exception: " << e.what() << std::endl */
            ;
        }

        // Tristrips must always be read with a 0 list size hint (unless you know exactly how many elements
        // are specifically in the file, which is unlikely);
        // try { tripstrip = file.request_properties_from_element("tristrips", { "vertex_indices" }, 0); }
        // catch (const std::exception & e) { /*std::cerr << "tinyply exception: " << e.what() << std::endl */; }

        file.read(*file_stream);

        // Example One: converting to your own application types
        if (vertices && vertices->t == tinyply::Type::FLOAT32) {
            // std::vector<float3> verts(vertices->count);
            // std::memcpy(verts.data(), vertices->buffer.get(), vertices->buffer.size_bytes());
            V.resize(vertices->count);
            std::memcpy(V.data()->data(), vertices->buffer.get(), vertices->buffer.size_bytes());
        }
        if (colors) {
            if (colors->t == tinyply::Type::UINT8 || colors->t == tinyply::Type::INT8) {
                uint8_t* p = (uint8_t*)colors->buffer.get();
                for (size_t i = 0; i < colors->count; i++, p += 3)
                    C.push_back(Color { (float)p[0] / 255.0f, (float)p[1] / 255.0f, (float)p[2] / 255.0f });
            }
            if (colors->t == tinyply::Type::FLOAT32) {
                float* p = (float*)colors->buffer.get();
                for (size_t i = 0; i < colors->count; i++, p += 3)
                    C.push_back(Color { p[0], p[1], p[2] });
            }
        }

        if (faces && (faces->t == tinyply::Type::INT32 || faces->t == tinyply::Type::INT32)) {
            Face one_face;
            uint32_t* p = (uint32_t*)faces->buffer.get();
            for (size_t i = 0; i < faces->count; i++, p += 3) {
                one_face.clear();
                one_face.push_back(p[0]);
                one_face.push_back(p[1]);
                one_face.push_back(p[2]);
                F.push_back(one_face);
            }
        }
    } catch (const std::exception& e) {
        tlog::error() << "loadPLY exception: " << e.what();
        return false;
    }

    tlog::info() << "Loading mesh vertext: " << V.size() << ", faces: " << F.size();

    return true;
}

bool Mesh::load(const char* filename)
{
    char* p = strrchr((char*)filename, (int)'.');

    if (p == NULL || (strcasecmp(p, ".ply") != 0 && strcasecmp(p, ".obj") != 0)) {
        tlog::error() << "Mesh load only support .ply or .obj format";
        return false;
    }

    if (strcasecmp(p, ".ply") == 0)
        return loadPLY(filename);

    return loadOBJ(filename);
}

bool Mesh::save(const char* filename)
{
    char* p = strrchr((char*)filename, (int)'.');

    if (p == NULL || strcasecmp(p, ".obj") != 0) {
        tlog::error() << "Mesh save only support .obj format";
        return false;
    }

    return saveOBJ(filename);
}

void Mesh::clean(Mask mask)
{
    // 1) Make sure all mask is valid
    for (size_t i = mask.size(); i < V.size(); i++)
        mask.push_back(true);

    // 2) Remove bad faces
    for (Face f : F) {
        auto it = f.begin();
        while (it != f.end()) {
            // face index >= V.size() or index is not valid
            if (*it >= V.size() || !mask[*it]) {
                it = f.erase(it);
            } else {
                ++it;
            }
        }
    }
    auto it = F.begin();
    while (it != F.end()) {
        if (it->size() < 3) { // Face size must be >= 3
            it = F.erase(it);
        } else {
            ++it;
        }
    }

    // 3) Remap face index
    size_t offset = 0;
    std::unordered_map<size_t, size_t> new_face_index;
    for (size_t i = 0; i < V.size(); i++) {
        if (mask[i])
            new_face_index[i] = offset++;
    }
    for (Face f : F) {
        for (size_t i = 0; i < f.size(); i++)
            f[i] = new_face_index[f[i]];
    }

    // 4) Remove unused points
    std::vector<Point> vertex;
    for (size_t i = 0; i < V.size(); i++) {
        if (mask[i])
            vertex.push_back(V[i]);
    }
    V = vertex;
}

GridIndex Mesh::grid_index(const AABB& aabb)
{
    GridIndex gi;

    tlog::info() << "Point cloud index ...";
    for (size_t n = 0; n < V.size(); n++) {
        GridKey key = aabb.grid_key(V[n]);
        gi[key].push_back(n);
    }
    tlog::info() << "Point cloud index numbers: " << gi.size();

    return gi;
}

GridNormal Mesh::grid_normal(GridIndex gi)
{
    GridNormal gn;
    float mean_density;
    GridKey new_key;
    bool has_color = (V.size() == C.size());

    for (auto n : gi) {
        const GridKey& key = n.first;
        const IndexList& index_list = n.second;
        if (index_list.size() < 1)
            continue;

        Eigen::Vector3f mean_point = Vector3f::Zero();
        Eigen::Vector3f mean_color = Vector3f::Zero();

        for (size_t i = 0; i < index_list.size(); i++) {
            mean_point.x() += V[index_list[i]].x();
            mean_point.y() += V[index_list[i]].y();
            mean_point.z() += V[index_list[i]].z();
            if (has_color) {
                mean_color.x() += C[index_list[i]].r;
                mean_color.y() += C[index_list[i]].g;
                mean_color.z() += C[index_list[i]].b;
            }
        }

        mean_point.x() /= index_list.size();
        mean_point.y() /= index_list.size();
        mean_point.z() /= index_list.size();
        if (has_color) {
            mean_color.x() /= index_list.size();
            mean_color.y() /= index_list.size();
            mean_color.z() /= index_list.size();
        }

        mean_density = index_list.size();
        new_key.i = key.i - 1;
        new_key.j = key.j + 0;
        new_key.k = key.k + 0;
        if (gi.find(new_key) != gi.end())
            mean_density += gi[key].size();
        new_key.i = key.i + 1;
        new_key.j = key.j + 0;
        new_key.k = key.k + 0;
        if (gi.find(new_key) != gi.end())
            mean_density += (float)gi[new_key].size();
        new_key.i = key.i + 0;
        new_key.j = key.j - 1;
        new_key.k = key.k + 0;
        if (gi.find(new_key) != gi.end())
            mean_density += (float)gi[new_key].size();
        new_key.i = key.i + 0;
        new_key.j = key.j + 1;
        new_key.k = key.k + 0;
        if (gi.find(new_key) != gi.end())
            mean_density += (float)gi[new_key].size();
        new_key.i = key.i + 0;
        new_key.j = key.j + 0;
        new_key.k = key.k - 1;
        if (gi.find(new_key) != gi.end())
            mean_density += (float)gi[new_key].size();
        new_key.i = key.i + 0;
        new_key.j = key.j + 0;
        new_key.k = key.k + 1;
        if (gi.find(new_key) != gi.end())
            mean_density += (float)gi[new_key].size();
        mean_density /= 7.0;

        gn[key] = GridCell { mean_point, Color { mean_color.x(), mean_color.y(), mean_color.z() }, mean_density };
    }

    return gn;
}

Mesh Mesh::grid_sample(uint32_t N)
{
    Mesh outmesh;

    AABB aabb(V);
    aabb.voxel(N);
    GridIndex gi = grid_index(aabb);
    GridNormal gn = grid_normal(gi);
    bool has_color = (V.size() == C.size());

    for (auto n : gn) {
        const GridKey& key = n.first;
        const GridCell& cell = gn[key];
        outmesh.V.push_back(cell.point);
        if (has_color)
            outmesh.C.push_back(cell.color);
    }

    return outmesh;
}

Mesh Mesh::grid_mesh(uint32_t N)
{
    static uint32_t cube_offset[8 * 3] = {
        0, 0, 0,
        1, 0, 0,
        1, 1, 0,
        0, 1, 0,
        0, 0, 1,
        1, 0, 1,
        1, 1, 1,
        0, 1, 1
    };

    float cube_points[8 * 3];
    float cube_colors[8 * 3];
    float cube_density[8];
    int has_color = (V.size() == C.size()) ? 1 : 0;

    Mesh outmesh;
    // uint32_t n_triangles = 0;

    AABB aabb(V);
    aabb.voxel(N);
    GridIndex gi = grid_index(aabb);
    GridNormal gn = grid_normal(gi);

    // for (auto n : gn) {

    for (uint32_t i = 0; i < aabb.dim.x(); i++) {
        for (uint32_t j = 0; j < aabb.dim.y(); j++) {
            for (uint32_t k = 0; k < aabb.dim.z(); k++) {

                GridKey key { i, j, k };
                // if (gn.find(key) == gn.end())
                //     continue;

                // const GridKey& key = n.first;

                // std::cout << key << " -- " << gn[key].density << std::endl;
                // continue;

                bool skip = true;
                for (int ii = 0; ii < 8; ii++) {
                    GridKey new_key { i + cube_offset[ii * 3 + 0], j + cube_offset[ii * 3 + 1], k + cube_offset[ii * 3 + 2] };

                    if (gn.find(new_key) == gn.end()) {
                        Point local = aabb.key_point(new_key);

                        cube_points[ii * 3 + 0] = local.x(); // 0.0f;
                        cube_points[ii * 3 + 1] = local.y(); // 0.0f;
                        cube_points[ii * 3 + 2] = local.z(); // 0.0f;

                        cube_colors[ii * 3 + 0] = 0.0f;
                        cube_colors[ii * 3 + 1] = 0.0f;
                        cube_colors[ii * 3 + 2] = 0.0f;

                        cube_density[ii] = 0.0;
                    } else {
                        skip = false;
                        const GridCell& cell = gn[new_key];

                        cube_points[ii * 3 + 0] = cell.point.x();
                        cube_points[ii * 3 + 1] = cell.point.y();
                        cube_points[ii * 3 + 2] = cell.point.z();

                        cube_colors[ii * 3 + 0] = cell.color.r;
                        cube_colors[ii * 3 + 1] = cell.color.g;
                        cube_colors[ii * 3 + 2] = cell.color.b;

                        cube_density[ii] = cell.density;
                    }
                }

                // if (! skip)
                outmesh.cube_mc(has_color, cube_points, cube_colors, cube_density, 2.0 / 7.0 /*borderval*/);
            }
        }
    }
    // }

    return outmesh;
}

void Mesh::snap(float e, float t)
{
    std::vector<Plane> planes;

    auto snap_logger = tlog::Logger("Create planes from face ...");
    auto progress = snap_logger.progress(F.size() + 1);
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

    PlaneNormals normals;
    for (Plane plane : planes)
        normals.data.push_back(plane.n);

    // construct a kd-tree index:
    using normal_kd_tree_t = nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<float, PlaneNormals>,
        PlaneNormals, 3 /* dim */
        >;
    normal_kd_tree_t index(3 /*dim*/, normals, { 10 /* max leaf */ });
    float query_point[3];

    const size_t num_results = 256;
    size_t ret_index[num_results];
    float out_dist_sqr[num_results];
    nanoflann::KNNResultSet<float> resultSet(num_results);
    resultSet.init(ret_index, out_dist_sqr);

    snap_logger = tlog::Logger("Create support planes ...");
    progress = snap_logger.progress(planes.size() + 1);

    std::vector<bool> need_check_planes(planes.size(), true);
    for (size_t i = 0; i < planes.size(); i++) {
        progress.update(i);
        if (!need_check_planes[i])
            continue;

        // do a knn search
        query_point[0] = planes[i].n.x();
        query_point[1] = planes[i].n.y();
        query_point[2] = planes[i].n.z();
        resultSet.init(ret_index, out_dist_sqr); // important ?
        index.findNeighbors(resultSet, &query_point[0]);

        for (size_t k = 1; k < resultSet.size(); k++) { // skip query point self
            size_t j = ret_index[k];
            if (!need_check_planes[j] || j <= i)
                continue;

            if (planes[i].coincide(planes[j], e, t)) {
                planes[i].ref_points.insert(planes[i].ref_points.end(), planes[j].ref_points.begin(), planes[j].ref_points.end());
                planes[i].ref_indexs.insert(planes[i].ref_indexs.end(), planes[j].ref_indexs.begin(), planes[j].ref_indexs.end());

                planes[j].ref_points.clear();
                planes[j].ref_indexs.clear();
                need_check_planes[j] = 0;
            }
        }
    }
    snap_logger.success("OK !");

    snap_logger = tlog::Logger("Refine support planes ...");
    progress = snap_logger.progress(planes.size() + 1);
    std::vector<Plane> support_planes;
    for (size_t i = 0; i < planes.size(); i++) {
        progress.update(i);
        if (need_check_planes[i]) {
            planes[i].refine();
            support_planes.push_back(planes[i]);
        }
    }
    planes.clear();
    need_check_planes.clear();
    snap_logger.success("OK !");
    tlog::info() << "Support planes: " << support_planes.size();

    snap_logger = tlog::Logger("Snap points ...");
    progress = snap_logger.progress(support_planes.size() + 1);
    for (size_t i = 0; i < support_planes.size(); i++) {
        progress.update(i);
        Plane plane = support_planes[i];
        if (plane.ref_indexs.size() <= 3)
            continue;

        for (uint32_t j : plane.ref_indexs) {
            if (plane.contains(V[j], e)) {
                V[j] = plane.project(V[j]);
            }
        }
    }
    snap_logger.success("OK !");

    support_planes.clear();
}

extern void test_segment();

void test_mesh()
{
    Mesh mesh;
    mesh.load("/tmp/dragon.obj");

    Mesh outmesh;
    outmesh = mesh.grid_mesh(256);
    outmesh.save("/tmp/mesh.obj");
}

void test_plane()
{
    // Mesh mesh;
    // mesh.load("/tmp/dragon.obj");
    // mesh.dump();
    // // mesh.simplify(0.1);
    // // mesh.save("/tmp/test1.obj");

    // // Mesh outmesh = mesh.grid_sample(256);
    // Mesh outmesh = mesh.grid_mesh(512);

    // outmesh.save("/tmp/test1.obj");

    // test_mesh();
    test_segment();
}
