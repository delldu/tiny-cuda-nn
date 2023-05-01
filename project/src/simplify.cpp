#include "mesh.h"
#include<queue>

struct EdgeCost;
using MinHeap = std::priority_queue<EdgeCost>;


struct Edge {
    Edge(size_t p1, size_t p2): p1(p1), p2(p2) {

    }

    friend std::ostream& operator<<(std::ostream& os, const Edge& e)
    {
        os << "Edge: <" << e.p1 << ", " << e.p2 << ">";
        return os;
    }

	bool operator< (const Edge& e) const  {
		if (p1 != e.p1)
			return p1 < e.p1;
		// ==> p1 == e.p1
		return p2 < e.p2;
	}

    size_t p1, p2;
};

struct Triangle {
    Triangle(size_t i1, size_t i2, size_t i3) {
        // size_t v[3] = {i1, i2, i3};
        size_t v[3];
        v[0] = i1; v[1] = i2; v[2] = i3;
        std::sort(v, v + 3);
        p1 = v[0]; p2 = v[1]; p3 = v[2];
    }

    bool valid() {
        return p1 < p2 && p2 < p3;
    }

    friend std::ostream& operator<<(std::ostream& os, const Triangle& t)
    {
        os << "Triangle: " << t.p1 << "," << t.p2 << "," << t.p3;
        return os;
    }

    size_t p1 = 0;
    size_t p2 = 0;
    size_t p3 = 0;
};

struct EdgeCost {
	EdgeCost(Edge e, float c): edge(e), cost(c) {
	}

    bool operator <(const EdgeCost &ec) const {
    	return ec.cost < cost;
    }

    friend std::ostream& operator<<(std::ostream& os, const EdgeCost& ec)
    {
        os << std::fixed << ec.edge << ",  Cost: " << ec.cost;
        return os;
    }

Edge edge;
float cost;
};



void Mesh::simplify(float ratio)
{
	MinHeap ec_heap;

	std::set<Edge> edges; // for query edge
	std::unordered_map<size_t, std::set<size_t>> v_edges; // v, point ...
	std::unordered_map<size_t, std::set<Edge>> v_triangles; // v, (e.p1, e.p2) ...


	size_t target = (size_t)(F.size() * ratio);
	AABB aabb(V);
	float threshold = aabb.diag().maxCoeff();

	auto edgeLength = [&](const Edge &e) -> float {
		return sqrtf((V[e.p1] - V[e.p2]).dot(V[e.p1] - V[e.p2]));
	};

	auto dump_edges = [&]() {
    	for (Edge e : edges)
    		std::cout << e << std::endl;
	};

	auto buildHeap = [&]() {
		while(! ec_heap.empty())
			ec_heap.pop();

		for (Face f:F) {
			if (f.size() != 3) {
				tlog::error() << "Face size: " << f.size() << " is not 3 !!!";
				continue;
			}
			Triangle t{f[0], f[1], f[2]};
			if (! t.valid()) {
				tlog::error() << "Triangle is not valid";
				continue;
			}

			edges.insert(Edge{t.p2, t.p3});
			edges.insert(Edge{t.p1, t.p3});
			edges.insert(Edge{t.p1, t.p2});

			v_triangles[t.p1].insert(Edge{t.p2, t.p3});
			v_triangles[t.p2].insert(Edge{t.p1, t.p3});
			v_triangles[t.p3].insert(Edge{t.p1, t.p2});

			v_edges[t.p1].insert(t.p2);
			v_edges[t.p1].insert(t.p3);
			v_edges[t.p2].insert(t.p1);
			v_edges[t.p2].insert(t.p3);
			v_edges[t.p3].insert(t.p1);
			v_edges[t.p3].insert(t.p2);
		}

		// Dump Edge ...
		dump_edges();


		// if (edgeLength(e) < threshold) {

		// }

	    // for (const auto& e : edge) {
	    //     addToHeap(e, threshold);
	    // }
	};

	auto solveEquation = []() {
		; // find best v
	};

	auto getPosition = [&]() {
		solveEquation();
	};

	auto selectEdge = [&]() {
		while(true) {
			getPosition();
			break;
		}
	};

	auto faceReverse = []() {
		; // what's up ?
	};


	auto removeEdge = [&]() {
		faceReverse();
	};


	buildHeap();
	// while(F.size() > target) {
	// 	selectEdge();
	// 	removeEdge();
	// }
}



// std::pair<Vector, double> getPosition(const Edge& e)
// {
//     Matrix q(4, Vector(4, 0));
//     for (const auto& f : face[e.first]) {
//         auto n = crossProduct(vertex[f.first] - vertex[e.first],
//             vertex[f.second] - vertex[e.first]);
//         n = n / norm(n);
//         // std::cout << "----- n.size(): " << n.size() << std::endl; // ==> n.size(): 3
//         n.push_back(-innerProduct(vertex[e.first], n));
//         outerProductFast(n, n, q);
//     }
//     for (const auto& f : face[e.second]) {
//         auto n = crossProduct(vertex[f.first] - vertex[e.second],
//             vertex[f.second] - vertex[e.second]);
//         n = n / norm(n);
//         n.push_back(-innerProduct(vertex[e.second], n));
//         outerProductFast(n, n, q);
//     }

//     Vector v;
//     try {
//         v = solveEquation(q, 3);
//     } catch (...) {
//         v = (vertex[e.first] + vertex[e.second]) / 2;
//     }
//     if (norm(v - vertex[e.first]) + norm(v - vertex[e.second]) > TOLERATE * norm(vertex[e.first] - vertex[e.second])) {
//         v = (vertex[e.first] + vertex[e.second]) / 2;
//     }
//     v.push_back(1);
//     double cost = innerProduct(innerProduct(v, q), v);
//     assert(cost > -EPS);
//     v.pop_back();
//     return make_pair(v, cost);
// }

// std::pair<Edge, Vector> selectEdge(double threshold)
// {
//     Edge idx = make_pair(-1, -1);
//     Vector pos;
//     std::pair<double, Edge> tmp;
//     while (!ec_heap.empty()) {
//         tmp = ec_heap.top(); // cost, e
//         ec_heap.pop();
//         if (edge.find(tmp.second) == edge.end())
//             continue;
//         if (removed[tmp.second.first] || removed[tmp.second.second])
//             continue;
//         if (edgeLen(tmp.second) > threshold)
//             continue;
//         auto act = getPosition(tmp.second); // v, cost
//         if (fabs(act.second + tmp.first) > EPS)
//             continue;
//         idx = tmp.second;  // ==> e
//         pos = act.first; // v
//         break;
//     }
//     printf("%lf %d %d", -tmp.first, idx.first, idx.second);
//     return std::make_pair(idx, pos); // ==> e, v
// }

// bool faceReverse(const Edge& e, const Vector& v1, const Vector& v2)
// {
//     const auto& x = vertex[e.first];
//     const auto& y = vertex[e.second];
//     return innerProduct(crossProduct(x - v1, y - v1),
//                crossProduct(x - v2, y - v2))
//         < 0;
// }

// void addToHeap(const Edge& e, double threshold)
// {
//     if (edgeLen(e) > threshold)
//         return;
//     auto pos = getPosition(e); // v, cost
//     ec_heap.push(make_pair(-pos.second, e));
// }

// void updateNeighborEdge(int v, double threshold)
// {
//     std::set<int> neighbor;
//     for (const auto& f : face[v]) {
//         neighbor.insert(f.first);
//         neighbor.insert(f.second);
//     }
//     for (auto x : neighbor) {
//         addToHeap(make_pair(min(x, v), max(x, v)), threshold);
//     }
// }

// void removeEdge(const Edge& e, const Vector& v, double threshold)
// {
//     std::vector<Edge> toRev;
//     for (const auto& f : face[e.first]) {
//         if (f.first == e.second || f.second == e.second)
//             continue;
//         auto reverse = faceReverse(f, vertex[e.first], v);
//         if (!reverse)
//             continue;
        
//         toRev.push_back(f);
//         assert(face[f.second].find(make_pair(e.first, f.first)) != face[f.second].end());
//         face[f.second].erase(make_pair(e.first, f.first));
//         face[f.second].insert(make_pair(f.first, e.first));

//         assert(face[f.first].find(make_pair(f.second, e.first)) != face[f.first].end());
//         face[f.first].erase(make_pair(f.second, e.first));
//         face[f.first].insert(make_pair(e.first, f.second));
//     }

//     for (const auto& f : toRev) {
//         face[e.first].erase(f);
//         face[e.first].insert(make_pair(f.second, f.first));
//     }

//     for (const auto& f : face[e.second]) {
//         assert(face[f.second].find(make_pair(e.second, f.first)) != face[f.second].end());
//         face[f.second].erase(make_pair(e.second, f.first));
//         auto reverse = faceReverse(f, vertex[e.second], v);
//         if (f.first != e.first && f.second != e.first) {
//             if (reverse) {
//                 face[f.second].insert(make_pair(f.first, e.first));
//             } else {
//                 face[f.second].insert(make_pair(e.first, f.first));
//             }
//         }

//         assert(face[f.first].find(make_pair(f.second, e.second)) != face[f.first].end());
//         face[f.first].erase(make_pair(f.second, e.second));
//         if (f.first != e.first && f.second != e.first) {
//             if (reverse) {
//                 face[f.first].insert(make_pair(e.first, f.second));
//             } else {
//                 face[f.first].insert(make_pair(f.second, e.first));
//             }
//         }

//         if (f.first == e.first || f.second == e.first)
//             faceN--;
//         else {
//             if (reverse) {
//                 face[e.first].insert(make_pair(f.second, f.first));
//             } else {
//                 face[e.first].insert(f);
//             }
//         }

//         auto tmp = make_pair(min(e.second, f.first), max(e.second, f.first));
//         if (edge.find(tmp) != edge.end())
//             edge.erase(tmp);
//         tmp = make_pair(min(e.second, f.second), max(e.second, f.second));
//         if (edge.find(tmp) != edge.end())
//             edge.erase(tmp);
//         if (f.first != e.first && f.second != e.first) {
//             edge.insert(make_pair(min(e.first, f.first), max(e.first, f.first)));
//             edge.insert(make_pair(min(e.first, f.second), max(e.first, f.second)));
//         }
//     }

//     edge.erase(e);
//     vertex[e.first] = v;
//     vertex[e.second].clear();
//     removed[e.second] = true;
//     face[e.second].clear();

//     std::set<int> neighbor;
//     for (const auto& f : face[e.first]) {
//         neighbor.insert(f.first);
//         neighbor.insert(f.second);
//     }
//     for (auto nb : neighbor) {
//         updateNeighborEdge(nb, threshold);
//     }
// }

// void buildHeap(double threshold)
// {
//     while (!ec_heap.empty())
//         ec_heap.pop();

//     for (const auto& e : edge) {
//         addToHeap(e, threshold);
//     }
// }
