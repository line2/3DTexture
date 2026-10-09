#include "engine.h"
#include "glb_loader.h"

#include <array>
#include <cassert>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

void add_face(Model& model, int a, int b, int c) {
    const int normal = static_cast<int>(model.normals.size());
    model.normals.emplace_back(0.0f, 0.0f, -1.0f);
    model.normals.emplace_back(0.0f, 0.0f, -1.0f);
    model.normals.emplace_back(0.0f, 0.0f, -1.0f);
    const int face[10] = {a, b, c, normal, normal + 1, normal + 2, 0, 0, 0, 0};
    model.faces.insert(model.faces.end(), face, face + 10);
}

void test_welding() {
    Model model;
    model.vertices = {
        {0.0f, 0.0f, 0.0f},
        {1.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f},
        {1.0f, 0.0f, 0.0f},
        {1.0f, 1.0f, 0.0f},
        {2.0f, 0.0f, 0.0f}
    };
    add_face(model, 0, 1, 2);
    add_face(model, 2, 3, 4);
    add_face(model, 0, 1, 5);

    WeldStats stats;
    assert(weld_model(model, true, stats) == 0);
    assert(model.vertices.size() == 5);
    assert(model.faces.size() == 20);
    assert(stats.removed_duplicate_vertices == 1);
    assert(stats.removed_degenerate_faces == 1);
    assert(!model.normals.empty());
}

void test_discovery() {
    const std::filesystem::path root =
        std::filesystem::temp_directory_path() / "3DTexture_engineering_test";
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
    std::filesystem::create_directories(root / "case" / "rank_000" / "controls", ec);
    {
        std::ofstream stream(root / "case" / "sample_manifest.json");
        stream << "{}";
    }
    {
        std::ofstream stream(root / "case" / "rank_000" / "controls" /
            "render_manifest.json");
        stream << "{}";
    }

    RunOptions options;
    options.dataset = root.string();
    options.format = InputFormat::Eval6Cases;
    std::vector<DatasetCase> cases;
    assert(discover_cases(options, cases) == 0);
    assert(cases.size() == 1);
    assert(cases[0].id == "case");
    assert(cases[0].format == InputFormat::Eval6Cases);
    std::filesystem::remove_all(root, ec);
}

std::vector<std::uint8_t> make_glb() {
    const std::string json = R"JSON({
      "scene": 0,
      "scenes": [{"nodes": [0]}],
      "nodes": [{"mesh": 0}],
      "meshes": [{"primitives": [{
        "attributes": {"POSITION": 1},
        "indices": 0,
        "mode": 4
      }]}],
      "accessors": [
        {"bufferView": 0, "componentType": 5125, "count": 3,
         "type": "SCALAR", "min": [0], "max": [2]},
        {"bufferView": 1, "componentType": 5126, "count": 3,
         "type": "VEC3", "min": [0, 0, 0], "max": [1, 1, 0]}
      ],
      "bufferViews": [
        {"buffer": 0, "byteOffset": 0, "byteLength": 12, "target": 34963},
        {"buffer": 0, "byteOffset": 12, "byteLength": 36, "target": 34962}
      ],
      "buffers": [{"byteLength": 48}]
    })JSON";
    std::vector<std::uint8_t> binary(48, 0);
    const std::uint32_t indices[3] = {0, 1, 2};
    std::memcpy(binary.data(), indices, sizeof(indices));
    const float positions[9] = {
        0.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f
    };
    std::memcpy(binary.data() + 12, positions, sizeof(positions));

    auto pad4 = [](std::size_t size) { return (size + 3) / 4 * 4; };
    const std::size_t json_size = json.size();
    const std::size_t json_padded = pad4(json_size);
    const std::size_t binary_size = binary.size();
    const std::size_t total = 12 + 8 + json_padded + 8 + binary_size;

    std::vector<std::uint8_t> result;
    const std::array<std::uint8_t, 4> magic = {'g', 'l', 'T', 'F'};
    result.insert(result.end(), magic.begin(), magic.end());
    auto push_u32 = [&](std::uint32_t value) {
        for (int i = 0; i < 4; ++i) {
            result.push_back(static_cast<std::uint8_t>(value >> (i * 8)));
        }
    };
    push_u32(2);
    push_u32(static_cast<std::uint32_t>(total));
    push_u32(static_cast<std::uint32_t>(json_size));
    result.insert(result.end(), {'J', 'S', 'O', 'N'});
    result.insert(result.end(), json.begin(), json.end());
    result.resize(result.size() + (json_padded - json_size), ' ');
    push_u32(static_cast<std::uint32_t>(binary_size));
    result.insert(result.end(), {'B', 'I', 'N', '0'});
    result.insert(result.end(), binary.begin(), binary.end());
    return result;
}

void test_glb_loader() {
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "3DTexture_engineering_test.glb";
    const std::vector<std::uint8_t> data = make_glb();
    {
        std::ofstream stream(path, std::ios::binary);
        stream.write(reinterpret_cast<const char*>(data.data()),
            static_cast<std::streamsize>(data.size()));
    }
    Model model;
    assert(read_glb_model(path.string(), model) == 0);
    assert(model.vertices.size() == 3);
    assert(model.faces.size() == 10);
    assert(!model.normals.empty());
    std::error_code ec;
    std::filesystem::remove(path, ec);
}

} // namespace

int main() {
    test_welding();
    test_discovery();
    test_glb_loader();
    std::cout << "engineering tests passed\n";
    return 0;
}
