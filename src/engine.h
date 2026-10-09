#pragma once

#include "texture.h"

#include <string>
#include <vector>

enum class InputFormat {
    Unknown,
    Tum,
    Test04,
    Eval6Cases
};

struct RunOptions {
    std::string dataset;
    std::string output;
    InputFormat format = InputFormat::Unknown;
    int threads = 0;
    int texture_size = 0;
    float color_lambda = -1.0f;
    int color_average = -1;
    float pairwise_weight = -1.0f;
    bool weld = true;
};

struct WeldStats {
    bool enabled = false;
    std::size_t input_vertices = 0;
    std::size_t output_vertices = 0;
    std::size_t removed_duplicate_vertices = 0;
    std::size_t input_faces = 0;
    std::size_t output_faces = 0;
    std::size_t removed_degenerate_faces = 0;
};

struct DatasetCase {
    std::string id;
    std::string path;
    InputFormat format = InputFormat::Unknown;
};

struct CaseResult {
    std::string id;
    bool success = false;
    std::string error;
    std::size_t input_vertices = 0;
    std::size_t output_vertices = 0;
    std::size_t faces = 0;
    std::size_t patches = 0;
    int texture_pages = 0;
    double seconds = 0.0;
};

int parse_cli(int argc, char** argv, RunOptions& options);
int discover_cases(const RunOptions& options, std::vector<DatasetCase>& cases);
int load_case(const DatasetCase& dataset_case, Model& model,
    std::vector<View>& views);
int weld_model(Model& model, bool enabled, WeldStats& stats);
int run_case(const DatasetCase& dataset_case, const RunOptions& options,
    const std::string& output_dir, CaseResult& result);
int run_all_cases(const RunOptions& options);
int run_legacy(const std::string& mesh, const std::string& view_dir,
    const std::string& output_dir);
