#include "image.h"

#include <algorithm>
#include <stdexcept>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

Image Image::crop(int x, int y, int crop_width, int crop_height) const {
    if (x < 0 || y < 0 || crop_width < 0 || crop_height < 0 ||
        x + crop_width > width_ || y + crop_height > height_) {
        throw std::out_of_range("image crop is outside the source image");
    }

    Image result(crop_width, crop_height);
    for (int row = 0; row < crop_height; ++row) {
        const std::uint8_t* src = pixel(x, y + row);
        std::uint8_t* dst = result.row(row);
        std::copy(src, src + static_cast<std::size_t>(crop_width) * 3, dst);
    }
    return result;
}

Image Image::load(const std::string& filename) {
    int width = 0;
    int height = 0;
    int source_channels = 0;
    std::uint8_t* decoded = stbi_load(filename.c_str(), &width, &height,
        &source_channels, 3);
    if (decoded == nullptr || width <= 0 || height <= 0) {
        return {};
    }

    Image image(width, height);
    const std::size_t byte_count =
        static_cast<std::size_t>(width) * height * 3;
    std::copy(decoded, decoded + byte_count, image.data());
    stbi_image_free(decoded);
    return image;
}

bool Image::save_png(const std::string& filename, const Image& image) {
    if (image.empty()) {
        return false;
    }
    return stbi_write_png(filename.c_str(), image.width(), image.height(), 3,
        image.data(), image.width() * 3) != 0;
}
