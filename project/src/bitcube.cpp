/************************************************************************************
***
***     Copyright 2023 Dell Du(18588220928@163.com), All Rights Reserved.
***
***     File Author: Dell, 2023年 03月 07日 星期二 18:29:34 CST
***
************************************************************************************/

#include "mesh.h"

std::vector<size_t> BitCube::find_valid_nb3x3x3(size_t pos)
{
    size_t nb_pos;
    uint32_t x, y, z;
    std::vector<size_t> s;

    pos_xyz(pos, &x, &y, &z);
    for (int ii = -1; ii <= 1; ii++) {
        for (int jj = -1; jj <= 1; jj++) {
            for (int kk = -1; kk <= 1; kk++) {
                xyz_pos(x + ii, y + jj, z + kk, &nb_pos);
                if (get_pos(nb_pos))
                    s.push_back(nb_pos);
            }
        }
    }

    return s;
}

std::vector<std::set<size_t>> BitCube::segment()
{
    std::set<size_t> res;
    std::queue<size_t> Q;
    std::vector<std::set<size_t>> cluster;

    for (size_t pos = 0; pos < size(); pos++) {
        if (!get_pos(pos))
            continue;

        // new cluster
        res.clear();
        Q.push(pos);
        while (!Q.empty()) {
            size_t tpos = Q.front();
            Q.pop();
            set_pos(tpos, false);
            res.insert(tpos); // now tpos has been moved from BitCude to res

            std::vector<size_t> nbs = find_valid_nb3x3x3(tpos);
            for (size_t i = 0; i < nbs.size(); i++) {
                Q.push(nbs[i]);
                set_pos(nbs[i], false);
            }
        }

        // save result
        cluster.push_back(res); // if res.size() < 5, means in 27 cells, probabilty <= 5/27 < 0.2
    }

    return cluster;
}
