#include "glb_loader.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#define TINYGLTF_NO_STB_IMAGE
#define TINYGLTF_NO_STB_IMAGE_WRITE
#define TINYGLTF_NO_EXTERNAL_IMAGE
#define TINYGLTF_IMPLEMENTATION
#include <tiny_gltf.h>

namespace {

bool ignore_image_data(tinygltf::Image* image, int, std::string*, std::string*,
    int, int, const unsigned char*, int, void*) {
    if (image != nullptr) {
        image->image.clear();
        image->width = 0;
        image->height = 0;
        image->component = 0;
        image->bits = 0;
        image->pixel_type = TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE;
    }
    return true;
}

Eigen::Matrix4f node_matrix(const tinygltf::Node& node) {
    Eigen::Matrix4f matrix = Eigen::Matrix4f::Identity();
    if (node.matrix.size() == 16) {
        for (int c = 0; c < 4; ++c) {
            for (int r = 0; r < 4; ++r) {
                matrix(r, c) = static_cast<float>(node.matrix[c * 4 + r]);
            }
        }
        return matrix;
    }

    Eigen::Matrix3f rotation = Eigen::Matrix3f::Identity();
    if (node.rotation.size() == 4) {
        const float x = static_cast<float>(node.rotation[0]);
        const float y = static_cast<float>(node.rotation[1]);
        const float z = static_cast<float>(node.rotation[2]);
        const float w = static_cast<float>(node.rotation[3]);
        rotation = Eigen::Quaternionf(w, x, y, z).normalized().toRotationMatrix();
    }
    Eigen::Matrix4f scale = Eigen::Matrix4f::Identity();
    if (node.scale.size() == 3) {
        scale(0, 0) = static_cast<float>(node.scale[0]);
        scale(1, 1) = static_cast<float>(node.scale[1]);
        scale(2, 2) = static_cast<float>(node.scale[2]);
    }
    Eigen::Vector3f translation = Eigen::Vector3f::Zero();
    if (node.translation.size() == 3) {
        translation = Eigen::Vector3f(
            static_cast<float>(node.translation[0]),
            static_cast<float>(node.translation[1]),
            static_cast<float>(node.translation[2]));
    }
    matrix.block<3, 3>(0, 0) = rotation;
    matrix.block<3, 1>(0, 3) = translation;
    return matrix * scale;
}

const unsigned char* accessor_data(const tinygltf::Model& gltf,
    const tinygltf::Accessor& accessor, int& stride) {
    if (accessor.bufferView < 0) {
        throw std::runtime_error("GLB accessor has no buffer view");
    }
    const tinygltf::BufferView& view = gltf.bufferViews[accessor.bufferView];
    if (view.buffer < 0 || view.buffer >= static_cast<int>(gltf.buffers.size())) {
        throw std::runtime_error("GLB accessor has an invalid buffer");
    }
    const tinygltf::Buffer& buffer = gltf.buffers[view.buffer];
    const std::size_t offset = view.byteOffset + accessor.byteOffset;
    if (offset + accessor.count * 12 > buffer.data.size()) {
        throw std::runtime_error("GLB position accessor is out of bounds");
    }
    stride = accessor.ByteStride(view);
    if (stride < 12) {
        throw std::runtime_error("GLB position accessor has an invalid stride");
    }
    return buffer.data.data() + offset;
}

void append_primitive(const tinygltf::Model& gltf,
    const tinygltf::Primitive& primitive, const Eigen::Matrix4f& transform,
    Model& model, bool& has_normals) {
    const auto position_it = primitive.attributes.find("POSITION");
    if (position_it == primitive.attributes.end()) {
        throw std::runtime_error("GLB primitive has no POSITION");
    }
    if (primitive.mode != TINYGLTF_MODE_TRIANGLES &&
        primitive.mode != TINYGLTF_MODE_TRIANGLE_STRIP &&
        primitive.mode != TINYGLTF_MODE_TRIANGLE_FAN) {
        throw std::runtime_error("GLB primitive is not triangle based");
    }

    const tinygltf::Accessor& position_accessor = gltf.accessors[position_it->second];
    if (position_accessor.componentType != TINYGLTF_COMPONENT_TYPE_FLOAT ||
        position_accessor.type != TINYGLTF_TYPE_VEC3) {
        throw std::runtime_error("GLB POSITION must be float VEC3");
    }
    int position_stride = 0;
    const unsigned char* position_data = accessor_data(gltf, position_accessor, position_stride);

    std::vector<int> indices;
    if (primitive.indices >= 0) {
        const tinygltf::Accessor& index_accessor = gltf.accessors[primitive.indices];
        const tinygltf::BufferView& view = gltf.bufferViews[index_accessor.bufferView];
        const tinygltf::Buffer& buffer = gltf.buffers[view.buffer];
        const std::size_t offset = view.byteOffset + index_accessor.byteOffset;
        const int index_stride = index_accessor.ByteStride(view);
        indices.reserve(index_accessor.count);
        for (std::size_t i = 0; i < index_accessor.count; ++i) {
            const unsigned char* data = buffer.data.data() + offset + i * index_stride;
            switch (index_accessor.componentType) {
            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
                indices.push_back(data[0]);
                break;
            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
                indices.push_back(*reinterpret_cast<const std::uint16_t*>(data));
                break;
            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
                indices.push_back(*reinterpret_cast<const std::uint32_t*>(data));
                break;
            default:
                throw std::runtime_error("unsupported GLB index type");
            }
        }
    } else {
        indices.resize(position_accessor.count);
        for (std::size_t i = 0; i < indices.size(); ++i) {
            indices[i] = static_cast<int>(i);
        }
    }

    std::vector<Eigen::Vector3f> positions;
    positions.reserve(position_accessor.count);
    for (std::size_t i = 0; i < position_accessor.count; ++i) {
        const float* p = reinterpret_cast<const float*>(
            position_data + i * position_stride);
        Eigen::Vector4f transformed = transform * Eigen::Vector4f(p[0], p[1], p[2], 1.0f);
        positions.emplace_back(transformed[0], transformed[1], transformed[2]);
    }

    std::vector<Eigen::Vector3f> normals;
    const auto normal_it = primitive.attributes.find("NORMAL");
    if (normal_it != primitive.attributes.end()) {
        has_normals = true;
        const tinygltf::Accessor& normal_accessor = gltf.accessors[normal_it->second];
        if (normal_accessor.componentType != TINYGLTF_COMPONENT_TYPE_FLOAT ||
            normal_accessor.type != TINYGLTF_TYPE_VEC3) {
            throw std::runtime_error("GLB NORMAL must be float VEC3");
        }
        int normal_stride = 0;
        const unsigned char* normal_data = accessor_data(gltf, normal_accessor, normal_stride);
        normals.reserve(normal_accessor.count);
        Eigen::Matrix3f normal_matrix = transform.block<3, 3>(0, 0).inverse().transpose();
        for (std::size_t i = 0; i < normal_accessor.count; ++i) {
            const float* n = reinterpret_cast<const float*>(normal_data + i * normal_stride);
            Eigen::Vector3f transformed = normal_matrix * Eigen::Vector3f(n[0], n[1], n[2]);
            const float length = transformed.norm();
            if (length > 1e-12f) {
                transformed /= length;
            } else {
                transformed = Eigen::Vector3f(0.0f, 0.0f, 1.0f);
            }
            normals.push_back(transformed);
        }
    }

    const int vertex_base = static_cast<int>(model.vertices.size());
    for (const Eigen::Vector3f& position : positions) {
        model.vertices.push_back(position);
    }
    int normal_base = 0;
    if (!normals.empty()) {
        normal_base = static_cast<int>(model.normals.size());
        for (const Eigen::Vector3f& normal : normals) {
            model.normals.push_back(normal);
        }
    }

    const int normal_index = std::max(0, normal_base);
    auto add_face = [&](int a, int b, int c) {
        if (a == b || b == c || c == a) {
            return;
        }
        int face[10] = {
            vertex_base + a, vertex_base + b, vertex_base + c,
            normal_index + a, normal_index + b, normal_index + c,
            0, 0, 0, 0
        };
        model.faces.insert(model.faces.end(), face, face + 10);
    };

    if (primitive.mode == TINYGLTF_MODE_TRIANGLES) {
        for (std::size_t i = 0; i + 2 < indices.size(); i += 3) {
            add_face(indices[i], indices[i + 1], indices[i + 2]);
        }
    } else if (primitive.mode == TINYGLTF_MODE_TRIANGLE_STRIP) {
        for (std::size_t i = 0; i + 2 < indices.size(); ++i) {
            if (i % 2 == 0) {
                add_face(indices[i], indices[i + 1], indices[i + 2]);
            } else {
                add_face(indices[i + 1], indices[i], indices[i + 2]);
            }
        }
    } else {
        for (std::size_t i = 1; i + 1 < indices.size(); ++i) {
            add_face(indices[0], indices[i], indices[i + 1]);
        }
    }
    (void)has_normals;
}

void append_node(const tinygltf::Model& gltf, int node_index,
    Eigen::Matrix4f parent, Model& model, bool& has_normals) {
    if (node_index < 0 || node_index >= static_cast<int>(gltf.nodes.size())) {
        throw std::runtime_error("GLB node index is out of range");
    }
    const tinygltf::Node& node = gltf.nodes[node_index];
    const Eigen::Matrix4f transform = parent * node_matrix(node);
    if (node.mesh >= 0) {
        const tinygltf::Mesh& mesh = gltf.meshes[node.mesh];
        for (const tinygltf::Primitive& primitive : mesh.primitives) {
            append_primitive(gltf, primitive, transform, model, has_normals);
        }
    }
    for (int child : node.children) {
        append_node(gltf, child, transform, model, has_normals);
    }
}

} // namespace

int read_glb_model(const std::string& filename, Model& model) {
    model.clear();
    tinygltf::Model gltf;
    tinygltf::TinyGLTF loader;
    loader.SetImageLoader(&ignore_image_data, nullptr);
    std::string error;
    std::string warning;
    if (!loader.LoadBinaryFromFile(&gltf, &error, &warning, filename)) {
        if (!error.empty()) {
            LOGE("failed to read GLB %s: %s", filename.c_str(), error.c_str());
        }
        return -1;
    }
    if (!warning.empty()) {
        LOGW("GLB warning in %s: %s", filename.c_str(), warning.c_str());
    }

    bool has_normals = false;
    for (int root : gltf.scenes[gltf.defaultScene].nodes) {
        append_node(gltf, root, Eigen::Matrix4f::Identity(), model, has_normals);
    }
    if (model.vertices.empty() || model.faces.empty()) {
        LOGE("GLB %s contains no triangle mesh", filename.c_str());
        return -1;
    }
    return 0;
}
