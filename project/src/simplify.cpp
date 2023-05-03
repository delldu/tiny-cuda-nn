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
        os << "Edge: <" << e.p1 << ", " << e.p2 << ">";
        return os;
    }

    bool operator<(const Edge& e) const
    {
        if (p1 != e.p1)
            return p1 < e.p1;
        // ==> p1 == e.p1
        return p2 < e.p2;
    }

    size_t p1, p2;
};

// struct Triangle {
//     Triangle(size_t i1, size_t i2, size_t i3)
//     {
//         // size_t v[3] = {i1, i2, i3};
//         size_t v[3];
//         v[0] = i1;
//         v[1] = i2;
//         v[2] = i3;
//         std::sort(v, v + 3);
//         p1 = v[0];
//         p2 = v[1];
//         p3 = v[2];
//     }

//     bool valid()
//     {
//         return p1 < p2 && p2 < p3;
//     }

//     friend std::ostream& operator<<(std::ostream& os, const Triangle& t)
//     {
//         os << "Triangle: " << t.p1 << "," << t.p2 << "," << t.p3;
//         return os;
//     }

//     size_t p1 = 0;
//     size_t p2 = 0;
//     size_t p3 = 0;
// };

struct EdgeCost {
    EdgeCost(Edge e, float c)
        : edge(e)
        , cost(c)
    {
    	; // e.p1 --> best, e.p2 --> best
    }

    bool operator<(const EdgeCost& ec) const
    {
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
    std::set<Edge> g_edges; // global edges
    std::vector<std::set<Edge>> v_faces; // face of vertex
    std::vector<bool> removed;
    size_t n_face;

    // cost = x.t * Q * x
    // std::unordered_map<size_t, Eigen::Matrix4f> v_qmatrixs; // QMatrix for point v which has adjacent face

    size_t target = (size_t)(F.size() * ratio);
    AABB aabb(V);
    float threshold = aabb.diag().maxCoeff();


    auto pre_process = [&]() {
    	n_face = F.size();
        v_faces.resize(V.size());
    	removed.resize(V.size(), false);

    	for (Face f:F) {
    		if (f.size() != 3) {
    			tlog::error() << "Face is not triangle";
    			continue;
    		}
    		size_t v[3];
    		v[0] = f[0]; v[1] = f[1]; v[2] = f[2];
    		v_faces[v[0]].insert(Edge{v[1], v[2]});
    		v_faces[v[1]].insert(Edge{v[2], v[0]});
    		v_faces[v[2]].insert(Edge{v[0], v[1]});

            std::sort(v, v + 3);
            g_edges.insert(Edge{v[0], v[1]});
            g_edges.insert(Edge{v[1], v[2]});
            g_edges.insert(Edge{v[0], v[2]});
    	}
    };

    auto post_process = [&]() {
        size_t offset = 0;
	    std::vector<Point> out_vertex;
	    std::vector<Color> out_colors;
        std::vector<size_t> out_index(V.size(), 0);

        bool has_color = (V.size() == C.size());
        for (size_t i = 0; i < V.size(); i++) {
            if (removed[i])
                continue;
            out_index[i] = ++offset;
            out_vertex.push_back(V[i]);

            if (has_color)
            	out_colors.push_back(C[i]);
        }
	    V = out_vertex;
	    if (has_color)
	    	C = out_colors;

	    F.clear();
	    Face new_face;
        for (size_t i = 0; i < V.size(); i++) {
            if (removed[i])
                continue;
            for (const auto& f : v_faces[i]) {
            	// i, f.p1, f.p2 is out_face
            	new_face.clear();
            	new_face.push_back(out_index[i]);
            	new_face.push_back(out_index[f.p1]);
            	new_face.push_back(out_index[f.p2]);
            	F.push_back(new_face);
            }
        }
    };


    auto edge_length = [&](const Edge& e) {
        return (V[e.p1] - V[e.p2]).norm();
    };


    auto solve_equation = [&](const Eigen::Matrix4f& Q, const Point &v) -> Point {
    	Eigen::Matrix3f A = Q.block(0, 0, 3, 3); // 3x3
    	if (fabsf(A.determinant()) < 0.000001f)
    		return v; // init_position

    	Eigen::Vector3f b = Vector3f{- Q(0, 3), -Q(1, 3), Q(2, 3)};
	    return A.colPivHouseholderQr().solve(b);
    };


    auto get_position = [&](const Edge& e) -> std::pair<float, Point> {
        Eigen::Matrix4f Q = Matrix4f::Zero();

        for (const auto& f : v_faces[e.p1]) {
        	/*
				f.p2  
				  |
			      |   e.p1 ------  e.p2 
				  |
				  |     
				f.p1
        	*/
        	Plane plane {V[e.p1], V[f.p2], V[f.p1]};

	    	Eigen::Vector4f abcd{plane.n.x(), plane.n.y(), plane.n.z(), -plane.n.dot(plane.o)};
	    	Q += abcd * abcd.transpose();
        }
        for (const auto& f : v_faces[e.p2]) {
        	/*
									f.p2  
				  					|
			   e.p1 ------  e.p2    |   
				  					|
				  					|     
									f.p1
        	*/
        	Plane plane {V[e.p2], V[f.p1], V[f.p2]};
	    	Eigen::Vector4f abcd{plane.n.x(), plane.n.y(), plane.n.z(), -plane.n.dot(plane.o)};
	    	Q += abcd * abcd.transpose();
        }

        Point v = (V[e.p1] + V[e.p2])/2.0;
        v = solve_equation(Q, v);

        if ((v - V[e.p1]).norm() + (v - V[e.p2]).norm() > TOLERATE * (V[e.p1] - V[e.p2]).norm()) {
            v = (V[e.p1] + V[e.p2]) / 2;
        }

    	Eigen::Vector4f x = Vector4f {v.x(), v.y(), v.z(), 1.0f};
    	float cost = x.transpose()*Q*x;

        return make_pair(cost, v);
    };


    auto put_edge_to_heap = [&](const Edge& e) {
        if (edge_length(e) > threshold)
            return;

        std::pair<float, Point> cost_pos = get_position(e); // cost, pos
        ec_heap.push(EdgeCost{e, cost_pos.first});
    };
 
    auto face_reverse = [&](const Edge& e, const Point& v1, const Point& v2) -> bool {
    	const Point &e1 = V[e.p1];
    	const Point &e2 = V[e.p2];
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


    auto remove_edge = [&](const Edge &e, const Point &v) {
        std::vector<Edge> reverse_faces;

        // Process edge.p1
        for (const auto& f : v_faces[e.p1]) {
            // reverse face exist ?
        	/*
				f.p1  
				  |
			e.p1  |   ------  e.p2 
				  |    v ?
				  |     
				f.p2
        	*/
            if (f.p1 == e.p2 || f.p2 == e.p2 || ! face_reverse(f, V[e.p1], v))
                continue;

            reverse_faces.push_back(f);
            v_faces[f.p2].erase(Edge{e.p1, f.p1});
            v_faces[f.p2].insert(Edge{f.p1, e.p1});

            v_faces[f.p1].erase(Edge{f.p2, e.p1});
            v_faces[f.p1].insert(Edge{e.p1, f.p2});
        }

        for (const auto& f : reverse_faces) {
            v_faces[e.p1].erase(f);
            v_faces[e.p1].insert(Edge{f.p2, f.p1});
        }

        // Process edge.p2
        for (const auto& f : v_faces[e.p2]) {
        	/*
								f.p2  
				  				|
			e.p1 ------  e.p2   |   
				  				|
				  |     v ?     |
								f.p1
        	*/

            bool reverse = face_reverse(f, V[e.p2], v);

 			// face like e.p1 --<|
            v_faces[f.p2].erase(Edge{e.p2, f.p1});
            if (f.p1 != e.p1 && f.p2 != e.p1) {
                v_faces[f.p2].insert(reverse? Edge{f.p1, e.p1} : Edge{e.p1, f.p1});
            }
            v_faces[f.p1].erase(Edge{f.p2, e.p2});
            if (f.p1 != e.p1 && f.p2 != e.p1) {
                v_faces[f.p1].insert(reverse?Edge{e.p1, f.p2} : Edge{f.p2, e.p1});
                v_faces[e.p1].insert(reverse?Edge{f.p2, f.p1} : Edge{f.p1, f.p2});
            } else {
                n_face--; // one face gone !!!
            }

            Edge tmp = Edge{std::min(e.p2, f.p1), std::max(e.p2, f.p1)};
            if (g_edges.find(tmp) != g_edges.end())
                g_edges.erase(tmp);
            tmp = Edge{std::min(e.p2, f.p2), std::max(e.p2, f.p2)};
            if (g_edges.find(tmp) != g_edges.end())
                g_edges.erase(tmp);
            if (f.p1 != e.p1 && f.p2 != e.p1) {
                g_edges.insert(Edge{std::min(e.p1, f.p1), std::max(e.p1, f.p1)});
                g_edges.insert(Edge{std::min(e.p1, f.p2), std::max(e.p1, f.p2)});
            }
        }

        g_edges.erase(e);
        V[e.p1] = v;
        // V[e.p2].clear();
        removed[e.p2] = true;
        v_faces[e.p2].clear();

        // Update neighbors of e.p1
        /*		    ___
				
				|   e.p1  |
					____
        */
        std::set<size_t> neighbors;
        for (const auto& f : v_faces[e.p1]) {
            neighbors.insert(f.p1);
            neighbors.insert(f.p2);
        }
        for (size_t v : neighbors) {
            std::set<size_t> s_neighbors;
            for (const auto& f : v_faces[v]) {
                s_neighbors.insert(f.p1);
                s_neighbors.insert(f.p2);
            }
            for (size_t x : s_neighbors)
                put_edge_to_heap(Edge{std::min(x, v), std::max(x, v)});
        }
    };


	pre_process();

    // std::cout << "Orignal Face Size: " << F.size() << " Target: " << target << std::endl;

    for (const Edge& e : g_edges)
        put_edge_to_heap(e);

    while(n_face > target && ! ec_heap.empty()) {
    	// select Edge
    	EdgeCost ec = ec_heap.top();
    	ec_heap.pop();

        if (g_edges.find(ec.edge) == g_edges.end() || removed[ec.edge.p1] || removed[ec.edge.p2])
            continue;

    	// std::cout << "EdgeCost ---- " << ec << std::endl;
        std::pair<float, Point> cost_pos = get_position(ec.edge); // cost, pos
        // std::cout << "GetPos ---- cost: " << cost_pos.first << std::endl;

        remove_edge(ec.edge, cost_pos.second /* v -- pos */);
        std::cout << "n_face: " << n_face << std::endl;
    }

	post_process();
}
