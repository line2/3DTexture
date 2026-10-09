#include "engine.h"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

namespace {

void usage(std::ostream& stream) {
    stream
        << "usage:\n"
        << "  3DTexture run --dataset DIR --output DIR [options]\n"
        << "  3DTexture MESH.obj RGB_CAM_DIR OUTPUT_DIR\n\n"
        << "options:\n"
        << "  --format auto|tum|test04|eval6cases\n"
        << "  --threads N\n"
        << "  --texture-size N\n"
        << "  --color-lambda FLOAT\n"
        << "  --color-average 0|1\n"
        << "  --pairwise-weight FLOAT\n"
        << "  --no-weld\n";
}

bool parse_int(const char* text, int& value) {
    char* end = nullptr;
    value = std::strtol(text, &end, 10);
    return end != text && *end == '\0';
}

bool parse_float(const char* text, float& value) {
    char* end = nullptr;
    value = std::strtof(text, &end);
    return end != text && *end == '\0';
}

bool parse_bool(const char* text, int& value) {
    if (std::strcmp(text, "0") == 0 || std::strcmp(text, "false") == 0) {
        value = 0;
        return true;
    }
    if (std::strcmp(text, "1") == 0 || std::strcmp(text, "true") == 0) {
        value = 1;
        return true;
    }
    return false;
}

InputFormat parse_format(const std::string& text) {
    if (text == "auto") {
        return InputFormat::Unknown;
    }
    if (text == "tum") {
        return InputFormat::Tum;
    }
    if (text == "test04") {
        return InputFormat::Test04;
    }
    if (text == "eval6cases" || text == "eval_6cases") {
        return InputFormat::Eval6Cases;
    }
    return InputFormat::Unknown;
}

void set_environment_value(const char* name, const std::string& value) {
#ifdef _WIN32
    _putenv_s(name, value.c_str());
#else
    setenv(name, value.c_str(), 1);
#endif
}

void apply_options_to_environment(const RunOptions& options) {
    if (options.threads > 0) {
        set_environment_value("3DTEXTURE_THREADS", std::to_string(options.threads));
    }
    if (options.texture_size > 0) {
        set_environment_value("3DTEXTURE_TEXTURE_SIZE", std::to_string(options.texture_size));
    }
    if (options.color_lambda >= 0.0f) {
        set_environment_value("3DTEXTURE_COLOR_LAMBDA", std::to_string(options.color_lambda));
    }
    if (options.color_average >= 0) {
        set_environment_value("3DTEXTURE_COLOR_AVERAGE",
            options.color_average == 0 ? "0" : "1");
    }
    if (options.pairwise_weight >= 0.0f) {
        set_environment_value("3DTEXTURE_PAIRWISE_WEIGHT",
            std::to_string(options.pairwise_weight));
    }
}

} // namespace

int parse_cli(int argc, char** argv, RunOptions& options) {
    if (argc < 2) {
        usage(std::cerr);
        return -1;
    }
    if (std::strcmp(argv[1], "--help") == 0 || std::strcmp(argv[1], "-h") == 0) {
        usage(std::cout);
        return 1;
    }
    if (std::strcmp(argv[1], "run") != 0) {
        if (argc >= 4) {
            return 2;
        }
        usage(std::cerr);
        return -1;
    }

    options = {};
    for (int i = 2; i < argc; ++i) {
        const std::string argument = argv[i];
        auto require_value = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::cerr << "missing value for " << name << "\n";
                return nullptr;
            }
            return argv[++i];
        };

        if (argument == "--dataset") {
            const char* value = require_value("--dataset");
            if (value == nullptr) return -1;
            options.dataset = value;
        } else if (argument == "--output") {
            const char* value = require_value("--output");
            if (value == nullptr) return -1;
            options.output = value;
        } else if (argument == "--format") {
            const char* value = require_value("--format");
            if (value == nullptr) return -1;
            options.format = parse_format(value);
            if (options.format == InputFormat::Unknown && std::strcmp(value, "auto") != 0) {
                std::cerr << "unknown format: " << value << "\n";
                return -1;
            }
        } else if (argument == "--threads") {
            const char* value = require_value("--threads");
            if (value == nullptr || !parse_int(value, options.threads) ||
                options.threads < 1 || options.threads > 1024) {
                std::cerr << "invalid --threads value\n";
                return -1;
            }
        } else if (argument == "--texture-size") {
            const char* value = require_value("--texture-size");
            if (value == nullptr || !parse_int(value, options.texture_size) ||
                options.texture_size < 1 || options.texture_size > 16384) {
                std::cerr << "invalid --texture-size value\n";
                return -1;
            }
        } else if (argument == "--color-lambda") {
            const char* value = require_value("--color-lambda");
            if (value == nullptr || !parse_float(value, options.color_lambda) ||
                options.color_lambda < 0.0f) {
                std::cerr << "invalid --color-lambda value\n";
                return -1;
            }
        } else if (argument == "--color-average") {
            const char* value = require_value("--color-average");
            if (value == nullptr || !parse_bool(value, options.color_average)) {
                std::cerr << "invalid --color-average value\n";
                return -1;
            }
        } else if (argument == "--pairwise-weight") {
            const char* value = require_value("--pairwise-weight");
            if (value == nullptr || !parse_float(value, options.pairwise_weight) ||
                options.pairwise_weight < 0.0f) {
                std::cerr << "invalid --pairwise-weight value\n";
                return -1;
            }
        } else if (argument == "--no-weld") {
            options.weld = false;
        } else {
            std::cerr << "unknown option: " << argument << "\n";
            usage(std::cerr);
            return -1;
        }
    }

    if (options.dataset.empty() || options.output.empty()) {
        std::cerr << "--dataset and --output are required\n";
        usage(std::cerr);
        return -1;
    }
    apply_options_to_environment(options);
    return 0;
}
