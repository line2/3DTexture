#include "texture.h"
#include <iostream>
#include "helper.h"
#include <chrono>
#include <filesystem>


int main(int argc, char** argv)
{
	if (argc < 4) {
		LOGE("usage: %s <mesh.obj> <view_dir> <output_dir> [label_file]", argv[0]);
		return -1;
	}

	std::string obj_file = argv[1];
	std::string view_dir = argv[2];
	std::string out_dir = argv[3];
	std::string label_file;
	if (argc > 4)
		label_file = argv[4];
	std::error_code ec;
	std::filesystem::create_directories(out_dir, ec);
	if (!std::filesystem::is_directory(out_dir, ec)) {
		LOGE("failed to create output directory %s", out_dir.c_str());
		return -1;
	}
	std::chrono::steady_clock::time_point t0, t1;

	std::vector<View> views;
	if (load_view(view_dir, views) != 0) {
		return -1;
	}

	Model model;
	if (read_obj_model(obj_file, model) != 0) {
		return -1;
	}
	
	Adj_face_list adj_face_list;
	if (build_adj_list(model, adj_face_list) != 0) {
		return -1;
	}

	std::vector<int> labels;
#ifdef TEX_DEBUG
	read_label_file(label_file, labels);
#else
	t0 = std::chrono::steady_clock::now();
	labelling(model, adj_face_list, views, labels);
	t1 = std::chrono::steady_clock::now();
	LOGI("Labelling time: %fs", std::chrono::duration<float>(t1 - t0).count());
	write_pile(out_dir + "/label.obj", out_dir + "/label.txt", model, labels);
#endif

	std::vector<cv::Mat> texture_imgs;
	t0 = std::chrono::steady_clock::now();
	int texture_num = seamless_texturing(model, views, adj_face_list, labels, model, texture_imgs);
	t1 = std::chrono::steady_clock::now();
	LOGI("Seamless texturing time: %fs", std::chrono::duration<float>(t1 - t0).count());
	
	write_obj(out_dir + "/result.obj", model, texture_num);

	write_mtl(out_dir + "/result.mtl", texture_num);

	write_texture(out_dir, texture_imgs);
	return 0;
}