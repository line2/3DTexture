cmake_minimum_required(VERSION 3.16)

project(3DTexture
    VERSION 1.0.0
    DESCRIPTION "Large-scale texturing of 3D reconstructions"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(3DTEXTURE_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)

find_package(PkgConfig REQUIRED)
pkg_check_modules(TBB REQUIRED IMPORTED_TARGET tbb)

add_executable(3DTexture
    src/texture.cpp
    src/image.cpp
    src/helper_impl.cpp
    src/graph.cpp
    src/labelling.cpp
    src/seamless_texturing.cpp
    src/texture_generator.cpp)

target_include_directories(3DTexture PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_include_directories(3DTexture SYSTEM PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/eigen335
    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/stb
    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/mapMAP
    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/libacc)
target_link_libraries(3DTexture PRIVATE PkgConfig::TBB)

if(MSVC)
    target_compile_options(3DTexture PRIVATE /W4 /permissive-)
else()
    target_compile_options(3DTexture PRIVATE -Wall -Wextra -Wpedantic)
    if(3DTEXTURE_WARNINGS_AS_ERRORS)
        target_compile_options(3DTexture PRIVATE -Werror)
    endif()
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(orthographic_projection_test
        tests/orthographic_projection_test.cpp
        src/graph.cpp
        src/helper_impl.cpp
        src/image.cpp)

    target_include_directories(orthographic_projection_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_include_directories(orthographic_projection_test SYSTEM PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/eigen335
    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/stb
        ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/libacc)
    if(MSVC)
        target_compile_options(orthographic_projection_test PRIVATE /W4 /permissive-)
    else()
        target_compile_options(orthographic_projection_test PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    add_test(NAME OrthographicProjection COMMAND orthographic_projection_test)
endif()

include(GNUInstallDirs)
install(TARGETS 3DTexture RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
