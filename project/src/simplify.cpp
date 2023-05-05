#include "mesh.h"
#include <queue>

#define TOLERATE 2.0

struct EdgeCost;
using MinHeap = std::priority_queue<EdgeCost>;

struct Edge {
    Edge(size_t p1, size_t p2)
        : p1(p1)
        , p2(p2)
    {
    }

    friend std::ostream& operator<<(std::ostream& os, const Edge& e)
    {
        os << "Edge: <" << std::fixed << e.p1 << ", " << e.p2 << ">";
        return os;
    }

    bool operator<(const Edge& e) const
    {
        if (p1 != e.p1)
            return p1 < e.p1;
        return p2 < e.p2; // p1 == e.p1
    }

    size_t p1, p2;
};

struct EdgeCost {
    EdgeCost(Edge e, float c, Point p)
        : edge(e)
        , cost(c)
        , pos(p)
    {
        ; // e.p1 --> best, e.p2 --> best
    }

    bool operator<(const EdgeCost& ec) const
    {
        return ec.cost < cost;
    }

    friend std::ostream& operator<<(std::ostream& os, const EdgeCost& ec)
    {
        os << ec.edge << ", Point: (" << ec.pos.x() << ", " << ec.pos.y() << ", " << ec.pos.z() << "),  Cost: " << ec.cost;
        return os;
    }

    Edge edge;
    float cost;
    Point pos; // example pos  = (v1 + v2)/2.0
};

void Mesh::simplify(float ratio)
{
    MinHeap ec_heap;
    std::set<Edge> g_edges; // global edges
    std::vector<std::set<Edge>> v_faces; // face edge of vertex
    std::vector<bool> v_removed;
    size_t n_face;
    float threshold;

    // Eigen::initParallel();
    // Eigen::setNbThreads(12);

    auto pre_process = [&]() {
        v_faces.resize(V.size());
        v_removed.resize(V.size(), false);
        n_face = F.size() - (size_t)(ratio * F.size()); // number of face which will be deleted

        AABB aabb(V);
        threshold = aabb.diag().maxCoeff();

        for (Face f : F) {
            if (f.size() != 3) {
                tlog::error() << "Face is not triangle";
                continue;
            }
            size_t v[3];
            v[0] = f[0];
            v[1] = f[1];
            v[2] = f[2];

            v_faces[v[0]].insert(Edge { v[1], v[2] });
            v_faces[v[1]].insert(Edge { v[2], v[0] });
            v_faces[v[2]].insert(Edge { v[0], v[1] });

            std::sort(v, v + 3);
            g_edges.insert(Edge { v[0], v[1] });
            g_edges.insert(Edge { v[1], v[2] });
            g_edges.insert(Edge { v[0], v[2] });
        }
    };

    auto post_process = [&]() {
        Face out_face;
        std::vector<Point> out_vertex;
        std::vector<Color> out_colors;
        std::vector<size_t> out_index(V.size(), 0);
        size_t old_size = V.size();
        size_t offset = 0;

        bool has_color = (V.size() == C.size());
        for (size_t i = 0; i < V.size(); i++) {
            if (v_removed[i])
                continue;

            out_index[i] = offset++;
            out_vertex.push_back(V[i]);

            if (has_color)
                out_colors.push_back(C[i]);
        }

        V = out_vertex;
        if (has_color)
            C = out_colors;

        F.clear();
        for (size_t i = 0; i < old_size; i++) {
            if (v_removed[i])
                continue;

            for (const Edge& fe : v_faces[i]) {
                // out_face -- i, fe.p1, fe.p2

                // duplicate face ?
                if (i >= fe.p1 || i >= fe.p2)
                    continue;

                out_face.clear();
                out_face.push_back(out_index[i]);
                out_face.push_back(out_index[fe.p1]);
                out_face.push_back(out_index[fe.p2]);
                F.push_back(out_face);
            }
        }
    };

    auto edge_length = [&](const Edge& e) {
        return (V[e.p1] - V[e.p2]).norm();
    };

    auto best_position = [&](const Eigen::Matrix4f& Q, const Point& init_v) -> Point {
        Eigen::Matrix3f A = Q.block(0, 0, 3, 3); // 3x3
        if (fabsf(A.determinant()) < 0.000001f)
            return init_v; // init_position

        Eigen::Vector3f b = Vector3f { -Q(0, 3), -Q(1, 3), -Q(2, 3) };
        return A.colPivHouseholderQr().solve(b);
    };

    // position -- 3f, cost -- 1f
    auto get_position = [&](const Edge& e) -> EdgeCost {
        Eigen::Matrix4f Q = Matrix4f::Zero();
        for (const auto& fe : v_faces[e.p1]) {
            /*
				fe.p2  
				  |
			      |   e.p1 ------  e.p2 
				  |
				  |     
				fe.p1
        	*/
            Plane plane { V[e.p1], V[fe.p2], V[fe.p1] };
            Eigen::Vector4f abcd { plane.n.x(), plane.n.y(), plane.n.z(), -plane.n.dot(plane.o) };
            Q += abcd * abcd.transpose();
        }
        for (const auto& fe : v_faces[e.p2]) {
            /*
									fe.p2  
				  					|
			   e.p1 ------  e.p2    |   
				  					|
				  					|     
									fe.p1
        	*/
            Plane plane { V[e.p2], V[fe.p1], V[fe.p2] };
            Eigen::Vector4f abcd { plane.n.x(), plane.n.y(), plane.n.z(), -plane.n.dot(plane.o) };
            Q += abcd * abcd.transpose();
        }

        Point v = (V[e.p1] + V[e.p2]) / 2.0;
        v = best_position(Q, v);

        // v far away from edge ?
        if ((v - V[e.p1]).norm() + (v - V[e.p2]).norm() > TOLERATE * (V[e.p1] - V[e.p2]).norm()) {
            v = (V[e.p1] + V[e.p2]) / 2.0;
        }

        Eigen::Vector4f x = Vector4f { v.x(), v.y(), v.z(), 1.0f };
        float cost = x.transpose() * Q * x;

        return EdgeCost { e, cost, v };
    };

    auto put_edge_to_heap = [&](const Edge& e) {
        if (edge_length(e) > threshold)
            return;

        EdgeCost ec = get_position(e);
        ec_heap.push(ec);
    };

    auto face_reverse = [&](const Edge& e, const Point& v1, const Point& v2) -> bool {
        const Point& e1 = V[e.p1];
        const Point& e2 = V[e.p2];
        /*
			e1 -------- e2
			   \     /
				\   /
				  v1 
    	*/
        Normal n1 = (e2 - v1).cross(e1 - v1);
        Normal n2 = (e2 - v2).cross(e1 - v2);

        return n1.dot(n2) < 0;
    };

    auto remove_edge = [&](const Edge& e, const Point& v) {
        std::vector<Edge> reverse_faces;

        // Process edge.p1
        for (const auto& fe : v_faces[e.p1]) {
            // reverse face exist ?
            /*
				fe.p1  
				  |
			e.p1  |   ------  e.p2 
				  |    v ?
				  |     
				fe.p2
        	*/
            if (fe.p1 == e.p2 || fe.p2 == e.p2 || !face_reverse(fe, V[e.p1], v))
                continue;

            reverse_faces.push_back(fe);
            v_faces[fe.p2].erase(Edge { e.p1, fe.p1 });
            v_faces[fe.p2].insert(Edge { fe.p1, e.p1 });

            v_faces[fe.p1].erase(Edge { fe.p2, e.p1 });
            v_faces[fe.p1].insert(Edge { e.p1, fe.p2 });
        }

        for (const auto& fe : reverse_faces) {
            v_faces[e.p1].erase(fe);
            v_faces[e.p1].insert(Edge { fe.p2, fe.p1 });
        }

        // Process edge.p2
        for (const auto& fe : v_faces[e.p2]) {
            /*
								fe.p2  
				  				|
			e.p1 ------  e.p2   |   
				  				|
				  |     v ?     |
								fe.p1
        	*/

            bool reverse = face_reverse(fe, V[e.p2], v);

            // face like e.p1 --<|
            v_faces[fe.p2].erase(Edge { e.p2, fe.p1 });
            if (fe.p1 != e.p1 && fe.p2 != e.p1) {
                v_faces[fe.p2].insert(reverse ? Edge { fe.p1, e.p1 } : Edge { e.p1, fe.p1 });
            }
            v_faces[fe.p1].erase(Edge { fe.p2, e.p2 });
            if (fe.p1 != e.p1 && fe.p2 != e.p1) {
                v_faces[fe.p1].insert(reverse ? Edge { e.p1, fe.p2 } : Edge { fe.p2, e.p1 });
                v_faces[e.p1].insert(reverse ? Edge { fe.p2, fe.p1 } : Edge { fe.p1, fe.p2 });
            } else {
                n_face--; // one face gone !!!
            }

            Edge te = Edge { std::min(e.p2, fe.p1), std::max(e.p2, fe.p1) };
            if (g_edges.find(te) != g_edges.end())
                g_edges.erase(te);
            te = Edge { std::min(e.p2, fe.p2), std::max(e.p2, fe.p2) };
            if (g_edges.find(te) != g_edges.end())
                g_edges.erase(te);
            if (fe.p1 != e.p1 && fe.p2 != e.p1) {
                g_edges.insert(Edge { std::min(e.p1, fe.p1), std::max(e.p1, fe.p1) });
                g_edges.insert(Edge { std::min(e.p1, fe.p2), std::max(e.p1, fe.p2) });
            }
        }

        g_edges.erase(e);
        V[e.p1] = v;
        v_removed[e.p2] = true;
        v_faces[e.p2].clear();

        // Update neighbors of e.p1
        /*		    ___
				
				|   e.p1  |
					____
        */
        std::set<size_t> neighbors;
        for (const auto& fe : v_faces[e.p1]) {
            neighbors.insert(fe.p1);
            neighbors.insert(fe.p2);
        }
        for (size_t v : neighbors) {
            std::set<size_t> s_neighbors;
            for (const auto& fe : v_faces[v]) {
                s_neighbors.insert(fe.p1);
                s_neighbors.insert(fe.p2);
            }
            for (size_t x : s_neighbors)
                put_edge_to_heap(Edge { std::min(x, v), std::max(x, v) });
        }
    };

    pre_process();
    // build heap
    for (const Edge& e : g_edges)
        put_edge_to_heap(e);

    while (n_face > 0 && !ec_heap.empty()) {
        EdgeCost ec = ec_heap.top();
        ec_heap.pop();

        if (g_edges.find(ec.edge) == g_edges.end())
            continue;

        if (v_removed[ec.edge.p1] || v_removed[ec.edge.p2])
            continue;

        auto nec = get_position(ec.edge); // ec cost changed more ?
        if (fabs(nec.cost - ec.cost) > 1.0e-6) {
            ec_heap.push(nec);
            continue;
        }

        remove_edge(ec.edge, ec.pos);
    }

    post_process();
}
