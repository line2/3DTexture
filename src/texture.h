#pragma once

#include "common.h"

#include "image.h"
#include <Eigen/Eigen>
#include <vector>
#include <set>
#include <unordered_map>
#include <map>

enum class ProjectionType {
    Perspective,
    Orthographic
};

typedef std::vector<std::vector<int>> Adj_face_list;

struct Model { //3D model
	std::vector<Eigen::Vector3f> vertices;
	std::vector<Eigen::Vector3f> normals;
	std::vector<Eigen::Vector2f> texcoords;
	std::vector<int> faces;

	void clear() {
		vertices.clear();
		normals.clear();
		texcoords.clear();
		faces.clear();
	}
};

struct View {
	ProjectionType projection_type = ProjectionType::Perspective;
	Image img;		// RGB image taken from different direction
	Eigen::Matrix3f R;	// Extrinsic: Rotation Matrix
	Eigen::Vector3f t;	// Extrinsic: Translation Vector
	Eigen::Matrix3f K;	// Perspective intrinsic matrix

	// Orthographic intrinsics, in pixel units.
	float orthographic_sx = 1.0f;
	float orthographic_sy = 1.0f;
	float orthographic_cx = 0.0f;
	float orthographic_cy = 0.0f;
};

inline Eigen::Vector2f project_vertex(const Eigen::Vector3f& vertex, const View& view)
{
	const Eigen::Vector3f p_cam = view.R * vertex + view.t;

	if (view.projection_type == ProjectionType::Perspective) {
		const Eigen::Vector3f p_img = view.K * p_cam;
		return Eigen::Vector2f(p_img(0) / p_img(2), p_img(1) / p_img(2));
	}

	return Eigen::Vector2f(
		view.orthographic_sx * p_cam(0) + view.orthographic_cx,
		view.orthographic_sy * p_cam(1) + view.orthographic_cy);
}

inline Eigen::Vector3f surface_to_camera_direction(const View& view)
{
	// R's third row is the camera z axis in world coordinates. A visible face
	// normal points from the surface towards the camera, i.e. along -z.
	return -Eigen::Vector3f(view.R(2, 0), view.R(2, 1), view.R(2, 2));
}

typedef struct Connected_area { // a connected area of the same label. correspond to patches(img).
	int label;
	std::vector<int> faces;
}CArea;

typedef struct SUB_Vertex_info { // sub-vertex color, index, uv coordinates, adjacent vertices
	int v_index;
	int label;
	float x, y;
	//int patch;
	Eigen::Vector3f color;
	Eigen::Vector3f color_ave;
	std::set<int> adj_vertices;

	SUB_Vertex_info() : v_index(-1), label(-1) {};
	SUB_Vertex_info(int _label, int _v_index) : v_index(_v_index), label(_label) {};
}VInfo;

struct BBox { // patch bounding box
	int patch_index;
	int bb[4];
};

typedef struct Texture_map { // patch table
	int texture_page;
	int x, y;
}TMap;

class Graph
{
public:
	int faces_num;
	int views_num;
	std::vector<std::vector<float>> data_costs;
	std::vector<bool> b_good_face;
	std::vector<std::vector<int>> valid_mask;

	Graph(int _faces_num, int _views_num);
	int calculate_data_cost(const Model& model, const std::vector<View>& views);
};


#define VEC_VERTICE_SIZE std::vector
#define VEC_SUB_VERTICE_SIZE std::vector
#define VEC_PATCHES_SIZE std::vector

typedef class Texture_generator{
private:
	int texture_width;
	int texture_height;
	float color_lambda = 0.1f;
	bool use_color_average = true;
	VEC_PATCHES_SIZE<CArea> connected_areas; // connected areas(faces) of the same label. correspond to patches(img).
	VEC_PATCHES_SIZE<BBox> patch_bboxes; // image patches' bounding boxes
	VEC_PATCHES_SIZE<TMap> texture_map; // patches location map to texture img.
	
	VEC_SUB_VERTICE_SIZE<VInfo> vertex_infos; // sub-vertex color, index, uv coordinates, adjacent vertices
	VEC_SUB_VERTICE_SIZE<Eigen::Vector3f> delta_color; // color adjustments for 3 channels.

	VEC_VERTICE_SIZE<std::map<int, int>> sub_vertex_table; //vector::index: vertex, map::key: label, map::value: sub_vertex
	
public:
	int patches_num;
	int sub_vertices_num;

	void configure_color_adjustment();

	Texture_generator() : texture_width(1000), texture_height(1000) {}
	Texture_generator(int _texture_height, int _texture_width) : texture_width(_texture_width), texture_height(_texture_height) {}

	int find_connected_area(const Adj_face_list& adj_face_list, const std::vector<int>& label);

	// compute the topology_info, color_info and patch info for each vertex (for color adjustment)
	void vertices_prepare(const std::vector<View>& views, const Model& model, const std::vector<int>& label);

	// calculate bounding boxes for image patches
	void calculate_bboxes(const Model& model);

	// arrange patch location on texture image.
	int patch_arrange();


	void color_adjustment_for_vertices(const Model& model, const std::vector<View>& views);

	// generate uv texcoord for model and renderer to show texture
	void generate_texcoord(const Model& src_model, Model& dst_model);


	void generate_texture(const Model& model, const std::vector<View> &views, int texture_num,
		std::vector<Image>& texture_imgs);
}TG;

int labelling(const Model &model, const Adj_face_list &adj_face_list,
	const std::vector<View>& views, std::vector<int>& labels);

int seamless_texturing(const Model& src_model, const std::vector<View> &views, const Adj_face_list& adj_face_list,
	const std::vector<int>& labels, Model& rst_model, std::vector<Image>& texture_imgs);