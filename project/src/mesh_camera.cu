/************************************************************************************
***
***     Copyright 2023 Dell Du(18588220928@163.com), All Rights Reserved.
***
***     File Author: Dell, 2023年 03月 07日 星期二 18:29:34 CST
***
************************************************************************************/
#include "../include/meshbox.h"
#include "../include/mesh_common.h"

static void read_camera(const string filename, Camera& camera)
{
    int i, j;
    char line[512], *p;
    ifstream fp;

    fp.open(filename.c_str(), ifstream::in);

    for (i = 0; i < 3; i++) {
        if (fp.eof())
            break;

        fp.getline(line, 512);
        for (j = 0, p = strtok(line, " "); p; p = strtok(NULL, " "), j++) {
            if (j >= 0 && j < 3) {
                camera.K(i, j) = (float)atof(p);
            } else if (j >= 3 && j < 6) {
                camera.R(i, j - 3) = (float)atof(p);
            } else if (j >= 6 && j < 7) {
                camera.T(i, j - 6) = (float)atof(p);
            } else {
                ; // comments, skip ...
            }
        }
    }
    camera.update();

    fp.close();
}

void Camera::load(const string filename) { read_camera(filename, *this); }

void Camera::dump()
{
    std::cout << "Camera:" << std::endl;
    std::cout << "K:" << std::endl << this->K << std::endl;
    std::cout << "R:" << std::endl << this->R << std::endl;
    std::cout << "T:" << std::endl << this->T << std::endl;
    std::cout << "KR:" << std::endl << this->K * this->R << std::endl;
    std::cout << "KT:" << std::endl << this->K * this->T << std::endl;
    std::cout << "R_K_inv:" << std::endl << this->R_K_inv << std::endl;
}
