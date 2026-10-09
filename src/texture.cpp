#include "engine.h"

#include <cstring>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    RunOptions options;
    const int parse_result = parse_cli(argc, argv, options);
    if (parse_result == 1) {
        return 0;
    }
    if (parse_result == 0) {
        return run_all_cases(options);
    }
    if (parse_result == 2) {
        if (argc < 4) {
            std::cerr << "legacy mode requires MESH.obj RGB_CAM_DIR OUTPUT_DIR\n";
            return -1;
        }
        return run_legacy(argv[1], argv[2], argv[3]);
    }
    return -1;
}
