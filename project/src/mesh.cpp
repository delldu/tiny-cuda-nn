/************************************************************************************
***
***     Copyright 2023 Dell Du(18588220928@163.com), All Rights Reserved.
***
***     File Author: Dell, 2023年 03月 07日 星期二 18:29:34 CST
***
************************************************************************************/

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

#define EPISON 5.0e-4f

using Vertex = Eigen::Vector3f;
using Normal = Eigen::Vector3f;
using Face = std::vector<size_t>;
using Point = Eigen::Vector3f;
using Points = std::vector<Point>;

struct Color {
	float r, g, b, a;
};

struct Mesh {
	bool load(char *filename);
	bool save(char *filename);


public:
	std::vector<Vertex> V;
	std::vector<Face> F;

	void dump_vertex() {
		for (Vertex v:V) {
			std::cout << v << std::endl;
		}
	}

	void dump_face() {
		for (Face face:F) {
			for (size_t fi:face)
				std::cout << fi << std::endl;
		}
	}
};

struct Plane {
    Plane(Point o, Normal n): o(o), n(n) {
    	n = n.normalized();
    }

    // p on plane ?
	bool contains(const Point p, float e=EPISON) const {
		return fabs(n.dot(p - o)) < e; // (p - o) _|_ n
	}

	Point project(const Point p) {
		float t = n.dot(p - o);
		return o + t * n;
	}

	void normalized() {
		n = n.normalized();
	}

	// fitting ...
	Plane(const Points points) {
		size_t size = points.size();
		if (size >= 3) {
			Eigen::MatrixXf A = Eigen::MatrixXf::Ones(size, 3);
			Eigen::VectorXf b = Eigen::VectorXf::Zero(size);

			for (size_t i = 0; i < size; i++) {
				A(i, 0) = points[i].x();
				A(i, 1) = points[i].y();
				b(i) = points[i].z();
			}
			Eigen::VectorXf x = A.colPivHouseholderQr().solve(b);
			n.x() = x[0];
			n.y() = x[1];
			n.z() = -1.0f;

			x = A.colwise().mean();
			o.x() = x[0];
			o.y() = x[1];
			o.z() = b.colwise().mean().x();
		}
		n = n.normalized();
	}

	bool same(const Plane& other, float e=EPISON) {
		return this->contains(other.o, e) && other.contains(this->o, e) && (fabs(this->n.dot(other.n)) > 1.0f - e);
	}


public:
	Point o = Vector3f::Zero(); // orignal
	Normal n = Vector3f::Ones(); // normal, should be normalized !!!
};


std::ostream& operator<<(std::ostream& os, const Point& point) {
	os << "Point " << std::fixed << "(" << point.x() << ", " << point.y() << ", " << point.z() << ")";
	return os;
}

std::ostream& operator<<(std::ostream& os, const Plane& plane) {
	os << "Plane " << std::fixed;
	os << "[o: " << plane.o.x() << ", " << plane.o.y() << ", " << plane.o.z() << "; ";
	os << "n: " << plane.n.x() << ", " << plane.n.y() << ", " << plane.n.z() << "]";
	return os;
}


struct BoundingBox {
	BoundingBox() {}
	BoundingBox(const Point& a, const Point& b) : min{a}, max{b} {}

	void update(const Point& point) {
		min = min.cwiseMin(point);
		max = max.cwiseMax(point);
	}

	void inflate(float amount) {
		min -= Point::Constant(amount);
		max += Point::Constant(amount);
	}

	Point diag() {
		return max - min;
	}

	Point relative_pos(const Point& pos) {
		return (pos - min).cwiseQuotient(diag());
	}

	Point center() {
		return 0.5f * (max + min);
	}

	BoundingBox intersection(const BoundingBox& other) {
		BoundingBox result = *this;
		result.min = result.min.cwiseMax(other.min);
		result.max = result.max.cwiseMin(other.max);
		return result;
	}

	bool intersects(const BoundingBox& other) {
		return !intersection(other).is_empty();
	}

	bool is_empty() const {
		return (max.array() < min.array()).any();
	}

	bool contains(const Point& p) {
		return p.x() >= min.x() && p.x() <= max.x() &&
			p.y() >= min.y() && p.y() <= max.y() &&
			p.z() >= min.z() && p.z() <= max.z();
	}

	Point min = Point::Constant(std::numeric_limits<float>::infinity());
	Point max = Point::Constant(-std::numeric_limits<float>::infinity());
};

std::ostream& operator<<(std::ostream& os, const BoundingBox& bb) {
	os << "[";
	os << "min=[" << bb.min.x() << "," << bb.min.y() << "," << bb.min.z() << "], ";
	os << "max=[" << bb.max.x() << "," << bb.max.y() << "," << bb.max.z() << "]";
	os << "]";
	return os;
}


void test_plane()
{
	Points points;

	Point p1 = Point{0.0, 0.0, 0.0};
	Point p2 = Point{0.0, 1.0, 0.0};
	Point p3 = Point{.0, 0.0, 0.0};


	points.push_back(p1);
	points.push_back(p2);
	points.push_back(p3);
	// points.push_back(Eigen::Vector3f{0.01, 0.01, 0.01});


	Plane plane(points);
	std::cout << plane << std::endl;

	Point p = Point{0.000001, 0.000001, 0.000001};
	std::cout << p << " on plane (epision = default) ? " << plane.contains(p) << std::endl;
	std::cout << p << " on plane (epision = 0.1f)? " << plane.contains(p, 0.1f) << std::endl;

	BoundingBox aabb;
	aabb.update(Point{-9.0, -8.0, -7.0});
	aabb.update(Point{0.0, 1.0, 0.0});
	aabb.update(Point{1.0, 2.0, 3.0});
	std::cout << "aabb:" << aabb << std::endl;

	Plane B(plane.o - Point{EPISON/2.0, EPISON/2.0, EPISON/2.0}, plane.n + Point{EPISON/2.0, EPISON/2.0, EPISON/2.0});

	std::cout << B << std::endl;
	std::cout << "Plane is same as B ?" << plane.same(B) << std::endl;
}
