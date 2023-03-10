/************************************************************************************
***
***     Copyright 2023 Dell Du(18588220928@163.com), All Rights Reserved.
***
***     File Author: Dell, 2023年 03月 07日 星期二 18:29:34 CST
***
************************************************************************************/
#ifndef __MESHBOX__H
#define __MESHBOX__H

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

#ifdef __cplusplus
	extern "C" {
#endif
#define MIN_DEPTH 0.00001f
#define MAX_DEPTH 16384.0f
#define MAX_IMAGES 512
#define MEGA_BYTES 1000000

struct Camera {
	// transform matrix = K*[R | T]
	Camera(): K(Matrix3f::Identity()), R(Matrix3f::Identity()), T(Vector3f::Zero()) {
		update();
	}

	void update() {
		KR = K * R;
		O = K * T;
		FWD_norm = R.col(2).normalized();
		R_K_inv = R.inverse() * K.inverse();
	}

	void load(const string filename);
	void dump();

	Matrix3f K;
	Matrix3f R;
	Vector3f T;

	// private for past
	Matrix3f KR;
	Vector3f O;
	Vector3f FWD_norm; // Forward normal
	Matrix3f R_K_inv; // R_inverse * K_inverse
};

struct Ray {
	Ray(): o(Vector3f::Zero()), d(Vector3f::Zero()) {
	}

	Eigen::Vector3f o;
	Eigen::Vector3f d;
};

struct Point {
	Point(): xyzw(Vector4f::Zero()), rgba(Vector4f::Zero()) {
	}
	Vector4f xyzw; // w == valid ? 1.0f, 0.0f
	Vector4f rgba;
};

int eval_points(char *input_folder);

#ifdef __cplusplus
	}
#endif

#endif // __MESHBOX__H
