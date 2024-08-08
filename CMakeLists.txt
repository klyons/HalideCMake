cmake_minimum_required(VERSION 3.15)
project(HalideCMake)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)


get_filename_component(ROOT_DIR "${CMAKE_SOURCE_DIR}/../vcpkg" ABSOLUTE)

# Include vcpkg toolchain using a relative path
set(CMAKE_TOOLCHAIN_FILE "${ROOT_DIR}/scripts/buildsystems/vcpkg.cmake")


# Get Halide tools and include directories
set(HALIDE_TOOLS_DIR "${ROOT_DIR}/packages/halide_x64-windows/share/halide/tools")
set(HALIDE_INCLUDE_DIR "${ROOT_DIR}/packages/halide_x64-windows/include")
set(HALIDE_LIB_DIR "${ROOT_DIR}/packages/halide_x64-windows/lib")

#get ZLIB, JPEG, and TIFF directories
set(ZLIB_INCLUDE "${ROOT_DIR}/packages/zlib_x64-windows/include")  
set(ZLIB_LIB "${ROOT_DIR}/packages/zlib_x64-windows/lib/zlib.lib")  
set(JPEG_INCLUDE "${ROOT_DIR}/packages/libjpeg-turbo_x64-windows/include")
set(JPEG_LIB "${ROOT_DIR}/packages/libjpeg-turbo_x64-windows/lib/jpeg.lib")
set(TURBOJPEG_LIB "${ROOT_DIR}/packages/libjpeg-turbo_x64-windows/lib/turbojpeg.lib")
set(TIFF_INCLUDE "${ROOT_DIR}/packages/tiff_x64-windows/include")
set(TIFF_LIB "${ROOT_DIR}/packages/tiff_x64-windows/lib/tiff.lib")

# Print the HALIDE_TOOLS_DIR path
message(STATUS "HALIDE_TOOLS_DIR : ${HALIDE_TOOLS_DIR}")
message(STATUS "ROOT_DIR : ${ROOT_DIR}")

# added the libs and include for JPEG and image file format libraries
include_directories(${JPEG_INCLUDE_DIR} ${ZLIB_INCLUDE_DIR} ${PNG_INCLUDE_DIR} ${TIFF_INCLUDE_DIR})
link_directories(${JPEG_LIBRARY} ${ZLIB_LIBRARY} ${PNG_LIBRARY} ${TIFF_LIBRARY})

# Manually set Halide directory
set(CMAKE_PREFIX_PATH "${ROOT_DIR}/installed/x64-windows/share/halide")
set(CMAKE_PREFIX_PATH "${ROOT_DIR}/packages/libjpeg-turbo_x64-windows/share/libjpeg-turbo")
set(CMAKE_PREFIX_PATH "${ROOT_DIR}/packages/libjpeg-turbo_x64-windows/share/jpeg")
set(CMAKE_PREFIX_PATH "${ROOT_DIR}/packages/tiff_x64-windows/share/tiff")


# Find Halide and HalideHelpers packages
find_package(Halide CONFIG REQUIRED)
find_package(HalideHelpers REQUIRED)
find_package(libjpeg-turbo CONFIG REQUIRED)
find_package(TIFF)
find_package(JPEG)
find_package(ZLIB)


# Find TIFF package
#find_package(TIFF REQUIRED)

# Print found Halide targets
get_target_property(Halide_TARGETS Halide::Halide INTERFACE_LINK_LIBRARIES)
message(STATUS "Halide targets: ${Halide_TARGETS}")

# Add executable
add_executable(HalideCMake HalideCMake.cpp)

# Include directories
target_include_directories(HalideCMake PRIVATE ${HALIDE_TOOLS_DIR})
target_include_directories(HalideCMake PRIVATE ${HALIDE_INCLUDE_DIR})
target_include_directories(HalideCMake PRIVATE ${TIFF_INCLUDE})
#target_include_directories(HalideCMake PRIVATE JPEG_INCLUDE_DIRS)
#target_include_directories(HalideCMake PRIVATE ${PNG_INCLUDE})

#broader scope 
include_directories(${JPEG_INCLUDE})
include_directories(${ZLIB_INCLUDE})


# Link libraries

#target_link_libraries(${TIFF_LIB})
#target_link_libraries(${JPEG_LIB})
#target_link_libraries(${TURBOJPEG_LIB})
#target_link_libraries(${ZLIB_LIB})

#target_link_libraries(HalideCMake PRIVATE ${JPEG_LIBRARY} ${TURBOJPEG_LIBRARY})
target_link_libraries(HalideCMake PRIVATE Halide::Halide Halide::Tools JPEG::JPEG) #TIFF::TIFF ZLIB::ZLIB PNG::PNG


