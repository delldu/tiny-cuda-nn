/************************************************************************************
***
***     Copyright 2023 Dell Du(18588220928@163.com), All Rights Reserved.
***
***     File Author: Dell, 2023年 03月 07日 星期二 18:29:34 CST
***
************************************************************************************/

#include "mesh.h"

MeshList Mesh::segment(uint32_t N, size_t outliers_threshold)
{
    size_t pos;
    uint32_t x, y, z;
    MeshList cluster;

    tlog::info() << "Point cloud segment ...";

    AABB aabb(V);
    aabb.voxel(N);
    bool has_color = (V.size() == C.size());
    GridIndex gi = grid_index(aabb);
    BitCube bc { aabb.dim.x(), aabb.dim.y(), aabb.dim.z() };

    for (auto n : gi) {
        const GridKey& key = n.first;
        bc.xyz_pos(key.i, key.j, key.k, &pos);
        bc.set_pos(pos, true);
    }
    std::vector<std::set<size_t>> bc_cluster = bc.segment();
    
    for (size_t i = 0; i < bc_cluster.size(); i++) {
        if (bc_cluster[i].size() < outliers_threshold)
            continue;

        Mesh tmesh;
        for (auto it = bc_cluster[i].begin(); it != bc_cluster[i].end(); it++) {
            pos = *it;
            bc.pos_xyz(pos, &x, &y, &z);
            GridKey key = GridKey { x, y, z };
            if (gi.find(key) == gi.end())
                continue;

            const IndexList& index_list = gi[key];
            for (size_t j = 0; j < index_list.size(); j++) {
                tmesh.V.push_back(V[index_list[j]]);
                if (has_color)
                    tmesh.C.push_back(C[index_list[j]]);
            }
        }
        cluster.push_back(tmesh);
    }

    tlog::info() << "Point cloud segment numbers: " << cluster.size();
    return cluster;
}

void Mesh::merge(MeshList cluster)
{
    static Color fake_colors[8] = {
        Color { 120 / 255.0f, 120 / 255.0f, 120 / 255.0f },
        Color { 180 / 255.0f, 120 / 255.0f, 120 / 255.0f },
        Color { 6 / 255.0f, 230 / 255.0f, 230 / 255.0f },
        Color { 80 / 255.0f, 50, 50 / 255.0f },
        Color { 4 / 255.0f, 200 / 255.0f, 3 / 255.0f },
        Color { 120 / 255.0f, 120 / 255.0f, 80 / 255.0f },
        Color { 140 / 255.0f, 140 / 255.0f, 140 / 255.0f },
        Color { 204 / 255.0f, 5 / 255.0f, 255 / 255.0f },
    };

    int label_color = 0;
    size_t offset = V.size();

    tlog::info() << "Point cloud merge ...";

    for (Mesh m : cluster) {
        for (Point p : m.V)
            V.push_back(p);

        for (Face f : m.F) {
            Face new_face;
            for (size_t fi : f)
                new_face.push_back(fi + offset);
            F.push_back(new_face);
        }
        Color fake_color = fake_colors[label_color % 8];
        label_color++;

        for (size_t i = 0; i < m.V.size(); i++)
            C.push_back(fake_color);

        offset += m.V.size();
    }
}

void test_segment()
{
    Mesh mesh;
    mesh.load("lego/002.ply");
    mesh.dump();

    Mesh outmesh;
    MeshList meshlist = mesh.segment(1024, 100);
    outmesh.merge(meshlist);

    outmesh.save("/tmp/segment.obj");
}
