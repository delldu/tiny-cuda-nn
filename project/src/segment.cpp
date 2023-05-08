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
    MeshList cluster;

    AABB aabb(V);
    aabb.voxel(N);
    bool has_color = (V.size() == C.size());
    GridIndex gi = grid_index(aabb);

    tlog::info() << "Point cloud segment ...";

    auto scan_keyset = [&](const IndexList& index_list, int start) {
        std::set<GridKey> s;

        for (size_t i = start; i < index_list.size(); i++) {
            GridKey tkey = aabb.grid_key(V[index_list[i]]);
            // neighbors keys -- total 27 (including tkey self)
            for (int ii = -1; ii <= 1; ii++) {
                for (int jj = -1; jj <= 1; jj++) {
                    for (int kk = -1; kk <= 1; kk++) {
                        s.insert(GridKey{tkey.i + ii, tkey.j + jj, tkey.k + kk});
                    }
                }
            }
        }
        return s;
    };

    for (auto n : gi) {
        const GridKey& key = n.first;
        const IndexList &index_list = n.second;
        if (index_list.empty())
            continue;

        size_t start = 0;
        while (gi[key].size() != start) {
            std::set<GridKey> keyset = scan_keyset(gi[key], start);
            start = gi[key].size(); // set for next

            for (GridKey new_key : keyset) {
                if (new_key == key || gi.find(new_key) == gi.end() || gi[new_key].empty())
                    continue;

                // move gi[new_key] to gi[key]
                gi[key].insert(gi[key].end(), gi[new_key].begin(), gi[new_key].end());
                gi[new_key].clear();
            }
        }
    }

    for (auto n:gi) {
        const IndexList &index_list = n.second;
        if (index_list.size() < outliers_threshold + 1)
            continue;

        Mesh tmesh;
        for (size_t i = 0; i < index_list.size(); i++) {
            tmesh.V.push_back(V[index_list[i]]);
            if (has_color)
                tmesh.C.push_back(C[index_list[i]]);
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
    MeshList meshlist = mesh.segment(256, 100);
    outmesh.merge(meshlist);

    outmesh.save("/tmp/segment.obj");
}
