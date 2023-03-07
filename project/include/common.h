/************************************************************************************
***
***     Copyright 2023 Dell Du(18588220928@163.com), All Rights Reserved.
***
***     File Author: Dell, 2022年 12月 29日 星期四 23:16:00 CST
***
************************************************************************************/
#pragma once

#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/config.h>

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

using namespace tcnn;
using precision_t = network_precision_t;

#include <Eigen/Dense> // Version 3.4.9, eigen.tgz under dependencies

