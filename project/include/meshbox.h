/************************************************************************************
***
***     Copyright 2023 Dell Du(18588220928@163.com), All Rights Reserved.
***
***     File Author: Dell, 2023年 03月 07日 星期二 18:29:34 CST
***
************************************************************************************/
#ifndef __MESHBOX__H
#define __MESHBOX__H

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
using namespace std;

#include <Eigen/Dense> // Version 3.4.9, eigen.tgz under dependencies
using namespace Eigen;

#define MIN_DEPTH 0.00001f
#define MAX_DEPTH 16384.0f
#define MAX_IMAGES 1024
#define MEGA_BYTES 1000000

struct Camera {
	Camera(): K(Matrix3f::Identity()), R(Matrix3f::Identity()), T(Vector3f::Zero()) {
		update();
	}

	void update() {
		KR = K * R;
		KT = K * T;
		O = K *R * T;
		R_K_inv = R.inverse() * K.inverse();
	}

	void load(const string filename);
	void dump();

	Eigen::Matrix3f K;
	Eigen::Matrix3f R;
	Eigen::Vector3f T;

	// Private
	Eigen::Matrix3f KR;
	Eigen::Vector3f KT;
	Eigen::Vector3f O;
	Eigen::Matrix3f R_K_inv; // R_inverse * K_inverse
};


struct Point {
	Point(): xyzw(Vector4f::Zero()), rgba(Vector4f::Zero()) {
	}

	Eigen::Vector4f xyzw; // w == valid ? 1.0f, 0.0f
	Eigen::Vector4f rgba;
};

bool has_cuda_device();
float get_gpu_memory();
void save_point_cloud(const string& filename, const vector<Point>& pc);
vector<string> load_files(const string dirname, const string extname);

#endif // __MESHBOX__H
