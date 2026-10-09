#include "engine.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <map>

namespace {

void ensure_normals(Model& model) {
    if (!model.normals.empty()) {
        return;
    }
    const std::size_t face_count = model.faces.size() / 10;
    model.normals.clear();
    model.normals.reserve(face_count * 3);
    for (std::size_t face = 0; face < face_count; ++face) {
        const int base = static_cast<int>(face * 10);
        const Eigen::Vector3f& a = model.vertices[model.faces[base]];
        const Eigen::Vector3f& b = model.vertices[model.faces[base + 1]];
        const Eigen::Vector3f& c = model.vertices[model.faces[base + 2]];
        Eigen::Vector3f normal = (b - a).cross(c - a);
        const float length = normal.norm();
        normal = length > 1e-12f ? normal / length : Eigen::Vector3f(0.0f, 0.0f, -1.0f);
        const int normal_index = static_cast<int>(model.normals.size());
        model.normals.push_back(normal);
        model.normals.push_back(normal);
        model.normals.push_back(normal);
        model.faces[base + 3] = normal_index;
        model.faces[base + 4] = normal_index + 1;
        model.faces[base + 5] = normal_index + 2;
    }
}

} // namespace

int weld_model(Model& model, bool enabled, WeldStats& stats) {
    stats = {};
    stats.enabled = enabled;
    stats.input_vertices = model.vertices.size();
    stats.input_faces = model.faces.size() / 10;

    if (!enabled) {
        ensure_normals(model);
        stats.output_vertices = model.vertices.size();
        stats.output_faces = model.faces.size() / 10;
        return 0;
    }

    std::map<std::array<float, 3>, int> canonical_vertices;
    std::vector<int> vertex_remap(model.vertices.size(), -1);
    std::vector<Eigen::Vector3f> welded_vertices;
    welded_vertices.reserve(model.vertices.size());

    for (std::size_t i = 0; i < model.vertices.size(); ++i) {
        const Eigen::Vector3f& vertex = model.vertices[i];
        std::array<float, 3> key = {vertex.x(), vertex.y(), vertex.z()};
        auto found = canonical_vertices.find(key);
        if (found == canonical_vertices.end()) {
            const int new_index = static_cast<int>(welded_vertices.size());
            canonical_vertices.emplace(key, new_index);
            welded_vertices.push_back(vertex);
            vertex_remap[i] = new_index;
        } else {
            vertex_remap[i] = found->second;
        }
    }

    std::vector<int> kept_faces;
    kept_faces.reserve(model.faces.size());
    for (std::size_t face = 0; face + 9 < model.faces.size(); face += 10) {
        const int v0 = vertex_remap[model.faces[face]];
        const int v1 = vertex_remap[model.faces[face + 1]];
        const int v2 = vertex_remap[model.faces[face + 2]];
        if (v0 == v1 || v1 == v2 || v2 == v0) {
            ++stats.removed_degenerate_faces;
            continue;
        }
        int current[10] = {
            v0, v1, v2,
            model.faces[face + 3], model.faces[face + 4], model.faces[face + 5],
            0, 0, 0, 0
        };
        kept_faces.insert(kept_faces.end(), current, current + 10);
    }

    if (kept_faces.empty()) {
        return -1;
    }

    std::vector<int> vertex_use(welded_vertices.size(), 0);
    for (std::size_t i = 0; i + 2 < kept_faces.size(); i += 10) {
        ++vertex_use[kept_faces[i]];
        ++vertex_use[kept_faces[i + 1]];
        ++vertex_use[kept_faces[i + 2]];
    }
    std::vector<int> compact_remap(welded_vertices.size(), -1);
    std::vector<Eigen::Vector3f> compact_vertices;
    compact_vertices.reserve(welded_vertices.size());
    for (std::size_t i = 0; i < welded_vertices.size(); ++i) {
        if (vertex_use[i] == 0) {
            continue;
        }
        compact_remap[i] = static_cast<int>(compact_vertices.size());
        compact_vertices.push_back(welded_vertices[i]);
    }
    for (std::size_t i = 0; i + 2 < kept_faces.size(); i += 10) {
        kept_faces[i] = compact_remap[kept_faces[i]];
        kept_faces[i + 1] = compact_remap[kept_faces[i + 1]];
        kept_faces[i + 2] = compact_remap[kept_faces[i + 2]];
    }

    model.vertices = std::move(compact_vertices);
    model.faces = std::move(kept_faces);
    // Input texcoords are not used by the atlas generator. Clear them to avoid
    // emitting stale indices if an input format provided UVs.
    model.texcoords.clear();
    ensure_normals(model);

    stats.output_vertices = model.vertices.size();
    stats.output_faces = model.faces.size() / 10;
    stats.removed_duplicate_vertices =
        stats.input_vertices > stats.output_vertices
        ? stats.input_vertices - stats.output_vertices
        : 0;
    return 0;
}
