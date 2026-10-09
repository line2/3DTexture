#include "engine.h"
#include "glb_loader.h"
#include "helper.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

bool file_exists(const fs::path& path) {
    std::error_code ec;
    return fs::is_regular_file(path, ec);
}

InputFormat detect_format(const fs::path& path) {
    std::error_code ec;
    if (!fs::is_directory(path, ec)) {
        return InputFormat::Unknown;
    }
    if (file_exists(path / "eval_metadata.json")) {
        return InputFormat::Test04;
    }
    if (file_exists(path / "sample_manifest.json") &&
        file_exists(path / "rank_000" / "controls" / "render_manifest.json")) {
        return InputFormat::Eval6Cases;
    }
    if (file_exists(path / "mesh.obj") && fs::is_directory(path / "rgb_cam", ec)) {
        return InputFormat::Tum;
    }
    return InputFormat::Unknown;
}

fs::path find_mesh(const fs::path& case_dir, InputFormat format) {
    if (format == InputFormat::Eval6Cases) {
        std::vector<fs::path> candidates;
        for (const fs::directory_entry& entry : fs::directory_iterator(case_dir / "mesh")) {
            const fs::path& path = entry.path();
            if (path.extension() == ".glb") {
                candidates.push_back(path);
            }
        }
        std::sort(candidates.begin(), candidates.end());
        for (const fs::path& path : candidates) {
            if (path.stem().string().find("_textured") != std::string::npos) {
                return path;
            }
        }
        if (!candidates.empty()) {
            return candidates.front();
        }
    }

    std::vector<fs::path> candidates;
    if (fs::is_directory(case_dir / "mesh")) {
        for (const fs::directory_entry& entry : fs::directory_iterator(case_dir / "mesh")) {
            if (entry.path().extension() == ".obj") {
                candidates.push_back(entry.path());
            }
        }
    } else {
        for (const fs::directory_entry& entry : fs::directory_iterator(case_dir)) {
            if (entry.path().extension() == ".obj") {
                candidates.push_back(entry.path());
            }
        }
    }
    std::sort(candidates.begin(), candidates.end());
    if (candidates.size() == 1) {
        return candidates.front();
    }
    if (candidates.empty()) {
        throw std::runtime_error("no OBJ or GLB mesh found in " + case_dir.string());
    }
    std::string names;
    for (const fs::path& path : candidates) {
        names += path.filename().string() + ", ";
    }
    throw std::runtime_error("multiple mesh candidates in " + case_dir.string() + ": " + names);
}

Eigen::Vector3f source_to_render(const Eigen::Vector3f& source) {
    return Eigen::Vector3f(source.x(), -source.z(), source.y());
}

json read_json(const fs::path& path) {
    std::ifstream stream(path);
    if (!stream.good()) {
        throw std::runtime_error("failed to open JSON: " + path.string());
    }
    try {
        return json::parse(stream);
    } catch (const std::exception& exc) {
        throw std::runtime_error("invalid JSON in " + path.string() + ": " + exc.what());
    }
}

float fit_height(const json& metadata) {
    const json* selected_fit = nullptr;
    if (metadata.contains("shards")) {
        for (const json& shard : metadata.at("shards")) {
            if (shard.contains("global_control_render_manifest")) {
                selected_fit = &shard.at("global_control_render_manifest")
                    .at("camera_fit").at("selected_fit");
                break;
            }
        }
    }
    if (selected_fit == nullptr || !selected_fit->contains("per_view")) {
        return 1.0f;
    }
    for (const json& view : selected_fit->at("per_view")) {
        if (view.value("name", "") == "front" && view.contains("height")) {
            const float height = view.at("height").get<float>();
            if (height > 1e-6f) {
                return height;
            }
        }
    }
    return 1.0f;
}

void normalize_model(Model& model, float target_height) {
    if (model.vertices.empty()) {
        throw std::runtime_error("cannot normalize an empty mesh");
    }
    Eigen::Vector3f min = model.vertices.front();
    Eigen::Vector3f max = model.vertices.front();
    for (const Eigen::Vector3f& vertex : model.vertices) {
        min = min.cwiseMin(vertex);
        max = max.cwiseMax(vertex);
    }
    const Eigen::Vector3f center = (min + max) * 0.5f;
    const Eigen::Vector3f extent = max - min;
    const float vertical_extent = std::max(extent.z(), 1e-12f);
    const float scale = target_height / vertical_extent;
    for (Eigen::Vector3f& vertex : model.vertices) {
        vertex = (vertex - center) * scale;
    }
}

View make_orthographic_view(const json& camera, const fs::path& image_path) {
    View view;
    view.img = Image::load(image_path.string());
    if (view.img.empty()) {
        throw std::runtime_error("failed to load view image: " + image_path.string());
    }
    if (camera.value("projection_type", "") != "ORTHO") {
        throw std::runtime_error("only ORTHO metadata views are supported");
    }

    if (!camera.contains("camera_to_world") ||
        !camera.at("camera_to_world").is_array() ||
        camera.at("camera_to_world").size() != 4) {
        throw std::runtime_error("camera_to_world must be a 4x4 matrix");
    }
    Eigen::Matrix4f camera_to_world = Eigen::Matrix4f::Identity();
    for (int row = 0; row < 4; ++row) {
        const json& json_row = camera.at("camera_to_world").at(row);
        if (!json_row.is_array() || json_row.size() != 4) {
            throw std::runtime_error("camera_to_world must be a 4x4 matrix");
        }
        for (int column = 0; column < 4; ++column) {
            camera_to_world(row, column) =
                json_row.at(column).get<float>();
        }
    }

    const Eigen::Vector3f right = camera_to_world.block<3, 1>(0, 0);
    const Eigen::Vector3f up = camera_to_world.block<3, 1>(0, 1);
    const Eigen::Vector3f backward = camera_to_world.block<3, 1>(0, 2);
    const Eigen::Vector3f camera_center = camera_to_world.block<3, 1>(0, 3);

    view.projection_type = ProjectionType::Orthographic;
    view.R.row(0) = right;
    view.R.row(1) = -up;
    view.R.row(2) = -backward;
    view.t = -(view.R * camera_center);

    const float ortho_scale = camera.at("render_ortho_scale").get<float>();
    if (!(ortho_scale > 0.0f)) {
        throw std::runtime_error("render_ortho_scale must be positive");
    }
    const float image_size = static_cast<float>(std::max(view.img.width(), view.img.height()));
    view.orthographic_sx = image_size / ortho_scale;
    view.orthographic_sy = image_size / ortho_scale;
    view.orthographic_cx = view.img.width() * 0.5f;
    view.orthographic_cy = view.img.height() * 0.5f;
    return view;
}

std::vector<std::pair<int, std::string>> selected_test04_views(const json& metadata) {
    if (!metadata.contains("selected_views") ||
        !metadata.at("selected_views").is_array()) {
        throw std::runtime_error("metadata has no selected_views array");
    }
    std::vector<std::pair<int, std::string>> result;
    for (const json& view : metadata.at("selected_views")) {
        result.emplace_back(view.at("selected_index").get<int>(),
            view.at("name").get<std::string>());
    }
    std::sort(result.begin(), result.end());
    return result;
}

std::map<std::string, json> collect_manifest_views(const json& metadata) {
    std::map<std::string, json> views;
    if (!metadata.contains("shards") || !metadata.at("shards").is_array()) {
        throw std::runtime_error("metadata has no shards array");
    }
    for (const json& shard : metadata.at("shards")) {
        if (!shard.contains("global_control_render_manifest")) {
            continue;
        }
        const json& manifest = shard.at("global_control_render_manifest");
        if (!manifest.contains("views") || !manifest.at("views").is_array()) {
            continue;
        }
        for (const json& view : manifest.at("views")) {
            views[view.at("label").get<std::string>()] = view;
        }
    }
    return views;
}

void load_test04(const fs::path& case_dir, Model& model, std::vector<View>& views) {
    const json metadata = read_json(case_dir / "eval_metadata.json");
    const fs::path mesh_path = find_mesh(case_dir, InputFormat::Test04);
    if (mesh_path.extension() == ".obj") {
        if (read_obj_model(mesh_path.string(), model) != 0) {
            throw std::runtime_error("failed to read OBJ mesh: " + mesh_path.string());
        }
    } else {
        throw std::runtime_error("test04 currently requires an OBJ mesh");
    }
    for (Eigen::Vector3f& vertex : model.vertices) {
        vertex = source_to_render(vertex);
    }
    normalize_model(model, fit_height(metadata));

    const auto selected = selected_test04_views(metadata);
    const auto manifest_views = collect_manifest_views(metadata);
    views.clear();
    views.reserve(selected.size());
    for (const auto& item : selected) {
        const auto found = manifest_views.find(item.second);
        if (found == manifest_views.end()) {
            throw std::runtime_error("camera manifest is missing view: " + item.second);
        }
        fs::path image = case_dir / (item.second + ".png");
        if (!file_exists(image)) {
            image = case_dir / (item.second + ".jpg");
        }
        if (!file_exists(image)) {
            throw std::runtime_error("missing view image: " + image.string());
        }
        views.push_back(make_orthographic_view(found->second, image));
    }
    if (views.empty()) {
        throw std::runtime_error("test04 metadata selected no views");
    }
}

std::vector<std::pair<int, std::string>> selected_eval6_views(const fs::path& case_dir) {
    std::vector<std::pair<int, std::string>> result;
    for (int rank = 0; rank < 4; ++rank) {
        std::ostringstream rank_name;
        rank_name << "rank_" << std::setw(3) << std::setfill('0') << rank;
        const fs::path selected_path = case_dir / rank_name.str() /
            "controls" / "selected_views.json";
        if (!file_exists(selected_path)) {
            continue;
        }
        const json selected = read_json(selected_path);
        if (!selected.is_array()) {
            throw std::runtime_error("selected_views.json must be an array");
        }
        for (const json& view : selected) {
            result.emplace_back(view.at("selected_index").get<int>(),
                view.at("name").get<std::string>());
        }
    }
    std::sort(result.begin(), result.end());
    if (result.size() != 16) {
        throw std::runtime_error("eval6 case must select exactly 16 views");
    }
    return result;
}

std::map<std::string, json> collect_eval6_manifest_views(const fs::path& case_dir) {
    std::map<std::string, json> result;
    for (int rank = 0; rank < 4; ++rank) {
        std::ostringstream rank_name;
        rank_name << "rank_" << std::setw(3) << std::setfill('0') << rank;
        const fs::path manifest_path = case_dir / rank_name.str() /
            "controls" / "render_manifest.json";
        if (!file_exists(manifest_path)) {
            continue;
        }
        const json manifest = read_json(manifest_path);
        if (!manifest.contains("views") || !manifest.at("views").is_array()) {
            continue;
        }
        for (const json& view : manifest.at("views")) {
            result[view.at("label").get<std::string>()] = view;
        }
    }
    return result;
}

void load_eval6(const fs::path& case_dir, Model& model, std::vector<View>& views) {
    const fs::path mesh_path = find_mesh(case_dir, InputFormat::Eval6Cases);
    if (mesh_path.extension() == ".obj") {
        if (read_obj_model(mesh_path.string(), model) != 0) {
            throw std::runtime_error("failed to read OBJ mesh: " + mesh_path.string());
        }
    } else if (mesh_path.extension() == ".glb") {
        if (read_glb_model(mesh_path.string(), model) != 0) {
            throw std::runtime_error("failed to read GLB mesh: " + mesh_path.string());
        }
    } else {
        throw std::runtime_error("unsupported eval6 mesh: " + mesh_path.string());
    }

    for (Eigen::Vector3f& vertex : model.vertices) {
        vertex = source_to_render(vertex);
    }
    const json manifest = read_json(case_dir / "rank_000" / "controls" /
        "render_manifest.json");
    normalize_model(model, fit_height(manifest));

    const auto selected = selected_eval6_views(case_dir);
    const auto manifest_views = collect_eval6_manifest_views(case_dir);
    views.clear();
    views.reserve(selected.size());
    for (const auto& item : selected) {
        const auto found = manifest_views.find(item.second);
        if (found == manifest_views.end()) {
            throw std::runtime_error("camera manifest is missing view: " + item.second);
        }
        const fs::path image = case_dir / ("composite_" + item.second + ".png");
        if (!file_exists(image)) {
            throw std::runtime_error("missing view image: " + image.string());
        }
        views.push_back(make_orthographic_view(found->second, image));
    }
}

} // namespace

int discover_cases(const RunOptions& options, std::vector<DatasetCase>& cases) {
    cases.clear();
    const fs::path dataset(options.dataset);
    InputFormat direct_format = detect_format(dataset);
    if (options.format != InputFormat::Unknown &&
        direct_format != InputFormat::Unknown && direct_format != options.format) {
        return -1;
    }
    if (direct_format != InputFormat::Unknown) {
        std::string id = dataset.filename().string();
        if (id.empty()) {
            id = "case";
        }
        cases.push_back({id, dataset.string(), direct_format});
        return 0;
    }

    std::error_code ec;
    if (!fs::is_directory(dataset, ec)) {
        return -1;
    }
    for (const fs::directory_entry& entry : fs::directory_iterator(dataset, ec)) {
        if (!entry.is_directory()) {
            continue;
        }
        const InputFormat format = detect_format(entry.path());
        if (format == InputFormat::Unknown) {
            continue;
        }
        if (options.format != InputFormat::Unknown && format != options.format) {
            continue;
        }
        cases.push_back({entry.path().filename().string(), entry.path().string(), format});
    }
    std::sort(cases.begin(), cases.end(),
        [](const DatasetCase& a, const DatasetCase& b) { return a.id < b.id; });
    return cases.empty() ? -1 : 0;
}

int load_case(const DatasetCase& dataset_case, Model& model,
    std::vector<View>& views) {
    model.clear();
    views.clear();
    try {
        const fs::path path(dataset_case.path);
        switch (dataset_case.format) {
        case InputFormat::Tum:
            if (read_obj_model((path / "mesh.obj").string(), model) != 0 ||
                load_view((path / "rgb_cam").string(), views) != 0 ||
                views.empty()) {
                return -1;
            }
            return 0;
        case InputFormat::Test04:
            load_test04(path, model, views);
            return 0;
        case InputFormat::Eval6Cases:
            load_eval6(path, model, views);
            return 0;
        case InputFormat::Unknown:
        default:
            return -1;
        }
    } catch (const std::exception& exc) {
        LOGE("%s", exc.what());
        return -1;
    }
}
