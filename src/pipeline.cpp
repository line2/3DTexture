#include "engine.h"
#include "helper.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

void ensure_directory(const fs::path& path) {
    std::error_code ec;
    fs::create_directories(path, ec);
    if (!fs::is_directory(path, ec)) {
        throw std::runtime_error("failed to create output directory: " + path.string());
    }
}

json weld_stats_json(const WeldStats& stats) {
    json result;
    result["enabled"] = stats.enabled;
    result["input_vertices"] = stats.input_vertices;
    result["output_vertices"] = stats.output_vertices;
    result["removed_duplicate_vertices"] = stats.removed_duplicate_vertices;
    result["input_faces"] = stats.input_faces;
    result["output_faces"] = stats.output_faces;
    result["removed_degenerate_faces"] = stats.removed_degenerate_faces;
    return result;
}

void write_json_file(const fs::path& path, const json& value) {
    std::ofstream stream(path);
    if (!stream.good()) {
        throw std::runtime_error("failed to open JSON output: " + path.string());
    }
    stream << value.dump(2) << '\n';
}

} // namespace

int run_case(const DatasetCase& dataset_case, const RunOptions& options,
    const std::string& output_dir, CaseResult& result) {
    result = {};
    result.id = dataset_case.id;
    const auto start = std::chrono::steady_clock::now();
    try {
        ensure_directory(output_dir);
        Model model;
        std::vector<View> views;
        if (load_case(dataset_case, model, views) != 0) {
            throw std::runtime_error("failed to load dataset case: " + dataset_case.path);
        }
        result.input_vertices = model.vertices.size();

        WeldStats stats;
        if (weld_model(model, options.weld, stats) != 0) {
            throw std::runtime_error("vertex welding removed all faces");
        }
        result.output_vertices = model.vertices.size();
        result.faces = model.faces.size() / 10;

        Adj_face_list adjacency;
        if (build_adj_list(model, adjacency) != 0) {
            throw std::runtime_error("failed to build face adjacency");
        }

        std::vector<int> labels;
        const auto labelling_start = std::chrono::steady_clock::now();
        if (labelling(model, adjacency, views, labels) != 0) {
            throw std::runtime_error("view labelling failed");
        }
        LOGI("Labelling time: %fs",
            std::chrono::duration<float>(std::chrono::steady_clock::now() - labelling_start).count());
        write_pile(std::string(output_dir) + "/label.obj",
            std::string(output_dir) + "/label.txt", model, labels);

        std::vector<Image> texture_images;
        int patch_count = 0;
        const auto texturing_start = std::chrono::steady_clock::now();
        const int texture_pages = seamless_texturing(model, views, adjacency,
            labels, model, texture_images, &patch_count);
        if (texture_pages < 1) {
            throw std::runtime_error("texture patch packing failed");
        }
        LOGI("Seamless texturing time: %fs",
            std::chrono::duration<float>(std::chrono::steady_clock::now() - texturing_start).count());
        result.patches = static_cast<std::size_t>(patch_count);
        result.texture_pages = texture_pages;

        if (write_obj(std::string(output_dir) + "/result.obj", model, texture_pages) != 0) {
            throw std::runtime_error("failed to write result.obj");
        }
        write_mtl(std::string(output_dir) + "/result.mtl", texture_pages);
        write_texture(output_dir, texture_images);

        const auto end = std::chrono::steady_clock::now();
        result.seconds = std::chrono::duration<double>(end - start).count();
        result.success = true;

        json manifest;
        manifest["case_id"] = dataset_case.id;
        manifest["dataset"] = dataset_case.path;
        manifest["format"] = dataset_case.format == InputFormat::Tum ? "tum" :
            dataset_case.format == InputFormat::Test04 ? "test04" : "eval6cases";
        manifest["view_count"] = views.size();
        manifest["vertices"] = model.vertices.size();
        manifest["faces"] = model.faces.size() / 10;
        manifest["patches"] = patch_count;
        manifest["texture_pages"] = texture_pages;
        manifest["welding"] = weld_stats_json(stats);
        manifest["seconds"] = result.seconds;
        manifest["outputs"] = {
            "result.obj", "result.mtl", "label.obj", "label.txt"
        };
        for (int i = 0; i < texture_pages; ++i) {
            std::ostringstream name;
            name << std::setfill('0') << std::setw(4) << i << ".png";
            manifest["outputs"].push_back(name.str());
        }
        write_json_file(fs::path(output_dir) / "run_manifest.json", manifest);
        return 0;
    } catch (const std::exception& exc) {
        result.error = exc.what();
        LOGE("case %s failed: %s", dataset_case.id.c_str(), result.error.c_str());
        return -1;
    }
}

int run_all_cases(const RunOptions& options) {
    try {
        std::vector<DatasetCase> cases;
        if (discover_cases(options, cases) != 0) {
            throw std::runtime_error("no supported dataset case found in: " + options.dataset);
        }
        const fs::path output_root(options.output);
        ensure_directory(output_root);

        std::vector<CaseResult> results;
        results.reserve(cases.size());
        for (const DatasetCase& dataset_case : cases) {
            const fs::path output_dir = output_root / dataset_case.id;
            CaseResult result;
            const int status = run_case(dataset_case, options, output_dir.string(), result);
            results.push_back(result);
            if (status != 0) {
                // Continue so summary.json records all failures.
                continue;
            }
            LOGI("case %s complete: vertices=%zu faces=%zu patches=%zu pages=%d",
                dataset_case.id.c_str(), result.output_vertices, result.faces,
                result.patches, result.texture_pages);
        }

        json summary;
        summary["dataset"] = options.dataset;
        summary["case_count"] = cases.size();
        summary["success_count"] = std::count_if(results.begin(), results.end(),
            [](const CaseResult& result) { return result.success; });
        summary["failure_count"] = results.size() - summary["success_count"].get<int>();
        summary["cases"] = json::array();
        for (const CaseResult& result : results) {
            json item;
            item["id"] = result.id;
            item["success"] = result.success;
            if (!result.error.empty()) {
                item["error"] = result.error;
            }
            item["input_vertices"] = result.input_vertices;
            item["output_vertices"] = result.output_vertices;
            item["faces"] = result.faces;
            item["patches"] = result.patches;
            item["texture_pages"] = result.texture_pages;
            item["seconds"] = result.seconds;
            summary["cases"].push_back(item);
        }
        write_json_file(output_root / "summary.json", summary);
        return summary["success_count"].get<int>() == static_cast<int>(cases.size()) ? 0 : -1;
    } catch (const std::exception& exc) {
        LOGE("%s", exc.what());
        return -1;
    }
}

int run_legacy(const std::string& mesh, const std::string& view_dir,
    const std::string& output_dir) {
    try {
        ensure_directory(output_dir);
        Model model;
        std::vector<View> views;
        if (read_obj_model(mesh, model) != 0 || load_view(view_dir, views) != 0 ||
            views.empty()) {
            return -1;
        }
        if (model.normals.empty()) {
            WeldStats stats;
            if (weld_model(model, false, stats) != 0) {
                return -1;
            }
        }
        Adj_face_list adjacency;
        if (build_adj_list(model, adjacency) != 0) {
            return -1;
        }
        std::vector<int> labels;
        if (labelling(model, adjacency, views, labels) != 0) {
            return -1;
        }
        write_pile(output_dir + "/label.obj", output_dir + "/label.txt", model, labels);
        std::vector<Image> texture_images;
        const int texture_pages = seamless_texturing(model, views, adjacency,
            labels, model, texture_images, nullptr);
        if (texture_pages < 1) {
            return -1;
        }
        if (write_obj(output_dir + "/result.obj", model, texture_pages) != 0) {
            return -1;
        }
        write_mtl(output_dir + "/result.mtl", texture_pages);
        write_texture(output_dir, texture_images);
        return 0;
    } catch (const std::exception& exc) {
        LOGE("%s", exc.what());
        return -1;
    }
}
