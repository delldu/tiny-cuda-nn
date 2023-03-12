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
		R_inv = R.inverse();

		// C3
		Matrix<float, 3, 4> P;
		for (int r = 0; r < 3; r++) {
			for (int c = 0; c < 3; c++)
				P(r, c) = KR(r, c);
			P(r, 3) = O(r, 0);
		}
		Matrix3f M1 = P(Eigen::placeholders::all, {1, 2, 3}); // x
		Matrix3f M2 = P(Eigen::placeholders::all, {0, 2, 3}); // y
		Matrix3f M3 = P(Eigen::placeholders::all, {0, 1, 3}); // z
		Matrix3f M4 = P(Eigen::placeholders::all, {0, 1, 2}); // t

		C3(0) = M1.determinant();
		C3(1) = -M2.determinant();
		C3(2) = M3.determinant();
		float t = -M4.determinant();
		C3(0) = C3(0)/t;
		C3(1) = C3(1)/t;
		C3(2) = C3(2)/t;
	}

	void load(const string filename);
	void dump();

	Matrix3f K;
	Matrix3f R;
	Vector3f T;

	// private for past
	Matrix3f KR;
	Vector3f O;
	Vector3f FWD_norm;// Forward normal
	Matrix3f R_inv;// R_inverse
	Matrix3f R_K_inv;// R_inverse * K_inverse
	Vector3f C3;
};

struct Ray {
	Ray(): o(Vector3f::Zero()), d(Vector3f::Zero()) {
	}

	Eigen::Vector3f o;
	Eigen::Vector3f d;
};

struct Point {
	Point(): xyz(Vector3f::Zero()), rgba(Vector4f::Zero()) {
	}
	Vector3f xyz;
	Vector4f rgba;  // w == valid ? 1.0f, 0.0f
};

int eval_points(char *input_folder);

#ifdef __cplusplus
	}
#endif

#endif // __MESHBOX__H
