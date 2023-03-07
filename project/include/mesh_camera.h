/************************************************************************************
***
***     Copyright 2023 Dell Du(18588220928@163.com), All Rights Reserved.
***
***     File Author: Dell, 2023年 03月 07日 星期二 18:29:34 CST
***
************************************************************************************/
#pragma once

#include <Eigen/Dense> // Version 3.4.9, eigen.tgz under dependencies
using namespace Eigen;

#include <string>
using namespace std;

struct Camera {
	Camera(): K(Matrix3f::Identity()), R(Matrix3f::Identity()), T(Vector3f::Zero()) {
		R_K_inv = R.inverse() * K.inverse();
	}

	void load(const string filename);
	void dump();

	Eigen::Matrix3f K;
	Eigen::Matrix3f R;
	Eigen::Vector3f T;
	Eigen::Matrix3f R_K_inv; // R_inverse * K_inverse
};

