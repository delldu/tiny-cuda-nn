/************************************************************************************
***
***     Copyright 2023 Dell Du(18588220928@163.com), All Rights Reserved.
***
***     File Author: Dell, 2023年 03月 07日 星期二 18:29:34 CST
***
************************************************************************************/
#pragma once

#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/config.h>
#include <stbi/stbi_wrapper.h>

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "tinylogger.h"

using namespace std;

using namespace tcnn;
using precision_t = network_precision_t;

#include <Eigen/Dense> // Version 3.4.9, eigen.tgz under dependencies
using namespace Eigen;
