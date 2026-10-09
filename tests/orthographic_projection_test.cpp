#include "texture.h"
#include "helper.h"

#include <Eigen/Core>
#include "image.h"

#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

void add_triangle(Model& model, const Eigen::Vector3f& a,
    const Eigen::Vector3f& b, const Eigen::Vector3f& c)
{
    const int base = static_cast<int>(model.vertices.size());
    const int normal_index = static_cast<int>(model.normals.size());

    model.vertices.push_back(a);
    model.vertices.push_back(b);
    model.vertices.push_back(c);
    model.normals.emplace_back(0.0f, 0.0f, -1.0f);

    model.faces.insert(model.faces.end(), {
        base, base + 1, base + 2,
        normal_index, normal_index, normal_index,
        0, 0, 0, 0
    });
}

bool test_projection()
{
    View view;
    view.projection_type = ProjectionType::Orthographic;
    view.R = Eigen::Matrix3f::Identity();
    view.t = Eigen::Vector3f::Zero();
    view.orthographic_sx = 100.0f;
    view.orthographic_sy = 100.0f;
    view.orthographic_cx = 320.0f;
    view.orthographic_cy = 240.0f;

    const Eigen::Vector2f uv = project_vertex(
        Eigen::Vector3f(1.0f, 2.0f, 3.0f), view);
    const bool passed = std::abs(uv.x() - 420.0f) < 1e-5f &&
        std::abs(uv.y() - 440.0f) < 1e-5f;

    if (!passed) {
        std::cerr << "unexpected orthographic projection: "
            << uv.x() << ", " << uv.y() << "\n";
    }
    return passed;
}

bool test_camera_file_parser()
{
    const std::filesystem::path directory =
        std::filesystem::temp_directory_path() / "3DTexture_orthographic_test";
    std::error_code ec;
    std::filesystem::remove_all(directory, ec);
    std::filesystem::create_directories(directory, ec);
    if (!std::filesystem::is_directory(directory, ec)) {
        std::cerr << "failed to create temporary test directory\n";
        return false;
    }

    const Image image(640, 480);
    if (!Image::save_png((directory / "view.png").string(), image)) {
        std::cerr << "failed to write temporary PNG\n";
        return false;
    }

    {
        std::ofstream out(directory / "view.cam");
        out << "ORTHO\n"
            << "0 0 0\n"
            << "1 0 0 0 1 0 0 0 1\n"
            << "100 100 320 240\n";
    }

    std::vector<View> views;
    if (load_view(directory.string(), views) != 0 || views.size() != 1) {
        std::cerr << "failed to load orthographic view\n";
        return false;
    }

    const View& view = views.front();
    const bool passed = view.projection_type == ProjectionType::Orthographic &&
        view.R.isIdentity(1e-6f) &&
        std::abs(view.orthographic_sx - 100.0f) < 1e-6f &&
        std::abs(view.orthographic_sy - 100.0f) < 1e-6f &&
        std::abs(view.orthographic_cx - 320.0f) < 1e-6f &&
        std::abs(view.orthographic_cy - 240.0f) < 1e-6f;

    std::filesystem::remove_all(directory, ec);
    return passed;
}

bool test_parallel_ray_occlusion()
{
    Model model;
    add_triangle(model,
        Eigen::Vector3f(-0.1f, -0.1f, 0.0f),
        Eigen::Vector3f( 0.1f, -0.1f, 0.0f),
        Eigen::Vector3f( 0.0f,  0.1f, 0.0f));
    add_triangle(model,
        Eigen::Vector3f(-0.1f, -0.1f, 1.0f),
        Eigen::Vector3f( 0.1f, -0.1f, 1.0f),
        Eigen::Vector3f( 0.0f,  0.1f, 1.0f));

    View view;
    view.projection_type = ProjectionType::Orthographic;
    view.R = Eigen::Matrix3f::Identity();
    view.t = Eigen::Vector3f::Zero();
    view.img = Image(640, 480);
    view.orthographic_sx = 100.0f;
    view.orthographic_sy = 100.0f;
    view.orthographic_cx = 320.0f;
    view.orthographic_cy = 240.0f;

    Graph graph(static_cast<int>(model.faces.size() / 10), 1);
    if (graph.calculate_data_cost(model, {view}) != 0) {
        return false;
    }

    // With an identity rotation, the camera is at negative infinity along z.
    // The triangle at z=0 is visible; the triangle at z=1 is behind it.
    const bool passed = graph.b_good_face[0] && !graph.b_good_face[1];
    if (!passed) {
        std::cerr << "unexpected orthographic visibility: "
            << graph.b_good_face[0] << ", " << graph.b_good_face[1] << "\n";
    }
    return passed;
}

} // namespace

int main()
{
    const bool passed = test_projection() &&
        test_camera_file_parser() &&
        test_parallel_ray_occlusion();

    if (!passed) {
        return 1;
    }
    std::cout << "orthographic projection tests passed\n";
    return 0;
}
