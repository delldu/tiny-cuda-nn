/************************************************************************************
***
***     Copyright 2023 Dell Du(18588220928@163.com), All Rights Reserved.
***
***     File Author: Dell, 2023年 03月 07日 星期二 18:29:34 CST
***
************************************************************************************/

#include "mesh.h"
#include "nanoflann.hpp"

struct PlaneNormals
{
    using coord_t = float;  //!< The type of each coordinate

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
    mesh.load("mesh.obj");
    // mesh.load("lego/point/pc.ply");
    mesh.dump();
    mesh.snap(10*D_EPISON, T_EPISON_15);
    mesh.save("test.obj");


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
          tlog::error() << "Open file " << filename ;
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
        try { vertices = file.request_properties_from_element("vertex", { "x", "y", "z" }); }
        catch (const std::exception & e) { /*std::cerr << "tinyply exception: " << e.what() << std::endl */; }

        // try { normals = file.request_properties_from_element("vertex", { "nx", "ny", "nz" }); }
        // catch (const std::exception & e) { /* std::cerr << "tinyply exception: " << e.what() << std::endl */; }

        try { colors = file.request_properties_from_element("vertex", { "red", "green", "blue" }); }
        catch (const std::exception & e) { /* std::cerr << "tinyply exception: " << e.what() << std::endl */; }

        try { colors = file.request_properties_from_element("vertex", { "r", "g", "b", "a" }); }
        catch (const std::exception & e) { /* std::cerr << "tinyply exception: " << e.what() << std::endl */; }

        // try { texcoords = file.request_properties_from_element("vertex", { "u", "v" }); }
        // catch (const std::exception & e) { /* std::cerr << "tinyply exception: " << e.what() << std::endl */; }

        // Providing a list size hint (the last argument) is a 2x performance improvement. If you have 
        // arbitrary ply files, it is best to leave this 0. 
        try { faces = file.request_properties_from_element("face", { "vertex_indices" }, 3); }
        catch (const std::exception & e) { /* std::cerr << "tinyply exception: " << e.what() << std::endl */ ; }

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
        if (colors && (colors->t == tinyply::Type::UINT8 || colors->t == tinyply::Type::INT8)) {
            uint8_t *p = (uint8_t *)colors->buffer.get();
            for (size_t i = 0; i < colors->count; i++, p += 3)
                C.push_back(Color{(float)p[0]/255.0f, (float)p[1]/255.0f, (float)p[2]/255.0f});
        }

        if (faces && (faces->t == tinyply::Type::INT32 || faces->t == tinyply::Type::INT32)) {
            Face one_face;
            int *p = (int *)faces->buffer.get();
            for (size_t i = 0; i < faces->count; i++, p += 3) {
                one_face.clear();
                one_face.push_back(p[0]);
                one_face.push_back(p[1]);
                one_face.push_back(p[2]);
                F.push_back(one_face);
            }
        }
    }
    catch (const std::exception & e) {
        tlog::error() << "loadPLY exception: " << e.what();
        return false;
    }

    return true;
}


bool Mesh::load(const char* filename)
{
    char *p = strrchr((char *)filename, (int)'.');

    if (p == NULL || (strcasecmp(p, ".ply") != 0  && strcasecmp(p, ".obj") != 0)) {
        tlog::error() << "Mesh load only support .ply or .obj format";
        return false;
    }

    if (strcasecmp(p, ".ply") == 0)
        return loadPLY(filename);

    return loadOBJ(filename);
}


bool Mesh::save(const char* filename)
{
    char *p = strrchr((char *)filename, (int)'.');

    if (p == NULL || strcasecmp(p, ".obj") != 0) {
        tlog::error() << "Mesh save only support .obj format";
        return false;
    }

    return saveOBJ(filename);
}

GridIndex Mesh::grid_index(size_t N)
{
    GridIndex gi;

    BoundingBox aabb(V);
    aabb.voxel(N);

    auto grid_logger = tlog::Logger("Grid index ...");
    auto progress = grid_logger.progress(V.size());
    for (size_t n = 0; n < V.size(); n++) {
        progress.update(n);
        GridKey key = aabb.grid_key(V[n]);
        gi[key].push_back(n);
    }
    grid_logger.success("OK !");

    return gi;
}


Mesh Mesh::grid_mesh(GridIndex gi)
{
    Mesh outmesh; // outmesh.N will be used for saving density ?

    float density_mean = 0.0f;
    float density_stdv = 0.0f;
    bool has_color = (V.size() == C.size());
    for (auto n:gi) {
        const IndexList &index_list = n.second;
        density_mean += index_list.size();
        density_stdv += index_list.size() * index_list.size();

        Eigen::Vector3f sum_point = Vector3f::Zero();
        Eigen::Vector3f sum_color = Vector3f::Zero();
        for (size_t i = 0; i < index_list.size(); i++) {
            sum_point.x() += V[index_list[i]].x();
            sum_point.y() += V[index_list[i]].y();
            sum_point.z() += V[index_list[i]].z();
            if (has_color) {
                sum_color.x() += C[index_list[i]].r;
                sum_color.y() += C[index_list[i]].g;
                sum_color.z() += C[index_list[i]].b;
            }
        }
        sum_point.x() /= index_list.size();
        sum_point.y() /= index_list.size();
        sum_point.z() /= index_list.size();

        if (has_color) {
            sum_color.x() /= index_list.size();
            sum_color.y() /= index_list.size();
            sum_color.z() /= index_list.size();
        }

        outmesh.V.push_back(Point{sum_point.x(), sum_point.y(), sum_point.z()});
        // outmesh.N.push_back(Normal{(float)index_list.size(), 0.0f, 0.0f});

        if (has_color) {
            outmesh.C.push_back(Color{sum_color.x(), sum_color.y(), sum_color.z()});
        }
    }
    if (gi.size() > 0) {
        density_mean /= gi.size();

        density_stdv /= gi.size();
        density_stdv -= density_mean;
        density_stdv = sqrtf(density_stdv);
    }

    // for (auto n: outmesh.N) {
    //     n.y() = density_mean;
    //     n.z() = density_stdv;
    // }

    return outmesh;
}

Mesh Mesh::remesh(size_t N)
{
    Mesh new_mesh;
    GridIndex gi = grid_index(N);
    new_mesh = grid_mesh(gi);

    // Call MC to create F for new_mesh
    new_mesh.F.clear();

    return new_mesh;
}

Mesh Mesh::simple(float ratio)
{
    Mesh new_mesh;

    if (ratio > 0.0f && ratio < 1.0f) {
        new_mesh.F.clear();
    }

    return new_mesh;
}

ClusterIndex Mesh::segment(float radius)
{
    ClusterIndex cluster;

    if (radius > 0.0f && radius < 1.0f) {
        IndexList c;
        c.push_back(100);

        cluster.push_back(c);
    }

    return cluster;
}


void Mesh::snap(float e, float t)
{
    std::vector<Plane> planes;

    auto snap_logger = tlog::Logger("Create planes from face ...");
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

    PlaneNormals normals;
    for (Plane plane:planes)
        normals.data.push_back(plane.n);

    // construct a kd-tree index:
    using normal_kd_tree_t = nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<float, PlaneNormals>,
        PlaneNormals, 3 /* dim */
    >;
    normal_kd_tree_t index(3 /*dim*/, normals, {10 /* max leaf */});
    float query_point[3];

    const size_t num_results = 256;
    size_t ret_index[num_results];
    float out_dist_sqr[num_results];
    nanoflann::KNNResultSet<float> resultSet(num_results);
    resultSet.init(ret_index, out_dist_sqr);

    snap_logger = tlog::Logger("Create support planes ...");
    progress = snap_logger.progress(planes.size());

    std::vector<int> need_check_planes(planes.size(), 1);
    for (size_t i = 0; i < planes.size(); i++) {
        progress.update(i);
        if (need_check_planes[i] == 0)
            continue;

        // do a knn search
        query_point[0] = planes[i].n.x();
        query_point[1] = planes[i].n.y();
        query_point[2] = planes[i].n.z();
        resultSet.init(ret_index, out_dist_sqr); // important ?
        index.findNeighbors(resultSet, &query_point[0]);

        for (size_t k = 1; k < resultSet.size(); k++) {
            size_t j = ret_index[k];
            if (need_check_planes[j] == 0 || j <= i)
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
    progress = snap_logger.progress(planes.size());
    std::vector<Plane> support_planes;
    for (size_t i = 0; i < planes.size(); i++) {
        progress.update(i);
        if (need_check_planes[i] == 1) {
            planes[i].refine();
            support_planes.push_back(planes[i]);
        }
    }
    planes.clear();
    need_check_planes.clear();
    snap_logger.success("OK !");
    tlog::info() << "Support planes: " << support_planes.size();

    snap_logger = tlog::Logger("Snap points ...");
    progress = snap_logger.progress(support_planes.size());
    for (size_t i = 0; i < support_planes.size(); i++) {
        progress.update(i);
        Plane plane = support_planes[i];
        if (plane.ref_indexs.size() <= 3)
            continue;

        for (size_t j : plane.ref_indexs) {
            if (plane.contains(V[j], e)) {
                V[j] = plane.project(V[j]);
            }
        }
    }
    snap_logger.success("OK !");

    support_planes.clear();
}
