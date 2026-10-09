#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// A compact, row-major, 8-bit RGB image backed by stb for image I/O.
class Image {
public:
    Image() = default;
    Image(int width, int height)
        : width_(width), height_(height), data_(static_cast<std::size_t>(width) * height * 3, 0)
    {
    }

    int width() const { return width_; }
    int height() const { return height_; }
    bool empty() const { return width_ <= 0 || height_ <= 0 || data_.empty(); }

    const std::uint8_t* data() const { return data_.data(); }
    std::uint8_t* data() { return data_.data(); }
    std::size_t size_bytes() const { return data_.size(); }

    const std::uint8_t* row(int y) const {
        return data_.data() + static_cast<std::size_t>(y) * width_ * 3;
    }
    std::uint8_t* row(int y) {
        return data_.data() + static_cast<std::size_t>(y) * width_ * 3;
    }

    const std::uint8_t* pixel(int x, int y) const {
        return row(y) + static_cast<std::size_t>(x) * 3;
    }
    std::uint8_t* pixel(int x, int y) {
        return row(y) + static_cast<std::size_t>(x) * 3;
    }

    Image crop(int x, int y, int crop_width, int crop_height) const;

    static Image load(const std::string& filename);
    static bool save_png(const std::string& filename, const Image& image);

private:
    int width_ = 0;
    int height_ = 0;
    std::vector<std::uint8_t> data_;
};
