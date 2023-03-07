/************************************************************************************
***
***     Copyright 2023 Dell Du(18588220928@163.com), All Rights Reserved.
***
***     File Author: Dell, 2023年 03月 07日 星期二 18:29:34 CST
***
************************************************************************************/
#include "../include/mesh_point.h"
#include "../include/mesh_common.h"

// static __device__ float l2_float4(float4 a)
// {
//  return sqrtf(pow2(a.x) + pow2(a.y) + pow2(a.z));

// }

// __device__ float depth_convert_cu(
//  const float &f, // focal length
//  const Camera_cu & cam_ref,
//  const Camera_cu & cam, const float &d)
// {
//  float baseline = l2_float4(cam_ref.C4 - cam.C4);
//  return f * baseline / d;
// }

__device__ void image_to_world(const float u, const float v, const float depth,
    const Matrix3f& RK_inv, const Vector3f& KT, Vector3f* __restrict__ X)
{
    Vector3f pt = Vector3f{ depth * u - KT.x(), depth * v - KT.y(), depth - KT.z() };
    *X = RK_inv * pt;
}

__device__ void world_to_image(const Vector3f& X, const Matrix3f& K, const Matrix3f& R,
    const Vector3f& T, float* u, float* v, float* depth)
{
    Vector3f temp = K * R * X + K * T;
    *depth = temp.z();
    *u = temp.x() / (temp.z() + 1e-10);
    *v = temp.y() / (temp.z() + 1e-10);
}
