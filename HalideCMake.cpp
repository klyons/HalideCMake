#define _CRT_SECURE_NO_WARNINGS
#include "Halide.h"
#include "halide_image_io.h"
#include "tiffio.h"

#include <vector>
#include<iostream>
#include <chrono>
#include<cmath>
#include<cstdint>

using namespace Halide;
using namespace Halide::Tools;
using namespace std;
//using namespace Halide::Runtime;


template <typename Func, typename... Args>
double timeFunction(Func func, Args&&... args) {
    auto start = std::chrono::high_resolution_clock::now();
    func(std::forward<Args>(args)...);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    return duration.count();
}

void brightManual(const std::string& filename, int factor) {
    //Halide::Buffer<uint8_t> input = load_image("Madonna.jpg");
    Halide::Buffer<uint8_t> input;
    try {
        input = Halide::Tools::load_image(filename);
    }
    catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
    }

    int h = input.height();
    int w = input.width();
    int c = input.channels();
    Halide::Buffer<uint8_t> output(w, h, c);
    uint8_t* inbuf = input.get()->data();
    uint8_t* outbuf = output.get()->data();


    for (int j = 0; j < h; j++) {
        for (int k = 0; k < w; k++) {

            *(outbuf + 3 * j * w + 3 * k) = std::clamp(*(inbuf + 3 * j * w + 3 * k) + factor, 0, 255);
            *(outbuf + 3 * j * w + 3 * k + 1) = std::clamp(*(inbuf + 3 * j * w + 3 * k + 1) + factor, 0, 255);
            *(outbuf + 3 * j * w + 3 * k + 2) = std::clamp(*(inbuf + 3 * j * w + 3 * k + 2) + factor, 0, 255);
            // *(outbuf + j*w + k) = std::clamp(*(inbuf + j*w + k)+factor,0,255);
            // *(outbuf + (h*w)+ j*w + k ) =  std::clamp(*(inbuf + (h*w)+ j*w + k)+factor,0,255);
            // *(outbuf + (2*h*w) + j*w + k) =  std::clamp(*(inbuf + (2*h*w) + j*w + k)+factor,0,255);

        }
    }
    //std::count << "saving image";
    save_image(output, "manual.jpg");
};

void brightHalide(const std::string& filename, int factor) {
    //Halide::Buffer<uint8_t> input = load_image("Madonna.jpg");
    Halide::Buffer<uint8_t> input;
    try {
        input = Halide::Tools::load_image(filename);
    }
    catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
    }


    Halide::Func brighter;
    Halide::Var x, y, c;

    Halide::Expr value = input(x, y, c);

    value = value + factor;
    value = Halide::min(value, 255.0f);

    value = Halide::cast<uint8_t>(value);

    brighter(x, y, c) = value;
    Halide::Buffer<uint8_t> output =
        brighter.realize({ input.width(), input.height(), input.channels() });

    save_image(output, "brighterHalide.png");
};

void brightHalide2(const std::string& filename, int factor) {
    Halide::Buffer<uint8_t> input;
    try {
        input = Halide::Tools::load_image(filename);
    }
    catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return; // Exit the function if image loading fails
    }

    Halide::Func brighten;
    Halide::Var x, y, c;

    // Ensure the input buffer is not empty
    if (input.width() == 0 || input.height() == 0 || input.channels() == 0) {
        std::cerr << "Input image is empty or not loaded correctly." << std::endl;
        return;
    }

    Halide::Expr value = input(x, y, c);
    Halide::Expr value_brightened = Halide::cast<int>(value) + factor;

    // Handle overflow for each color channel separately
    value_brightened = Halide::clamp(value_brightened, 0, 255);

    brighten(x, y, c) = Halide::cast<uint8_t>(value_brightened);

    // Ensure the dimensions match the input buffer
    Halide::Buffer<uint8_t> output = brighten.realize({ input.width(), input.height(), input.channels() });

    // Save the output image
    Halide::Tools::save_image(output, "brighterHalide2.png");
}


void SobelDetectors(const std::string& filename) {
    // Load the input image
    Buffer<uint8_t> input_uint8 = Halide::Tools::load_image(filename);

    // Cast the input image to int16_t
    Buffer<int16_t> input(input_uint8.width(), input_uint8.height(), input_uint8.channels());
    for (int y = 0; y < input.height(); y++) {
        for (int x = 0; x < input.width(); x++) {
            for (int c = 0; c < input.channels(); c++) {
                input(x, y, c) = static_cast<int16_t>(input_uint8(x, y, c));
            }
        }
    }

    // Define the Sobel kernels
    int16_t sobel_x[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };

    int16_t sobel_y[3][3] = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };

    // Define the algorithm
    Var x, y, c;
    Func clamped = BoundaryConditions::repeat_edge(input);
    Func sobel_x_func, sobel_y_func, sobel;

    RDom r(-1, 3, -1, 3);

    sobel_x_func(x, y, c) = 0;
    sobel_y_func(x, y, c) = 0;

    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            sobel_x_func(x, y, c) += sobel_x[i + 1][j + 1] * clamped(x + i, y + j, c);
            sobel_y_func(x, y, c) += sobel_y[i + 1][j + 1] * clamped(x + i, y + j, c);
        }
    }

    sobel(x, y, c) = cast<int16_t>(clamp(sqrt(sobel_x_func(x, y, c) * sobel_x_func(x, y, c) +
        sobel_y_func(x, y, c) * sobel_y_func(x, y, c)), 0, 255));

    // Schedule the algorithm
    sobel.vectorize(x, 16).parallel(y);

    // Realize the algorithm
    Buffer<int16_t> output(input.width(), input.height(), input.channels());

    // Realize the algorithm into the output buffer
    sobel.realize(output);

    // Save the output
    save_image(output, "SobelOutput.png");
}

float calculateKernelVariance(Buffer<uint8_t> input, int x, int y, int kernel_size) {
    int half_kernel = kernel_size / 2;
    RDom r(-half_kernel, kernel_size, -half_kernel, kernel_size);

    // Define a Func to compute the mean
    Func mean_func;
    mean_func() = sum(cast<float>(input(x + r.x, y + r.y))) / (kernel_size * kernel_size);

    // Realize the mean
    Buffer<float> mean_buf = mean_func.realize();
    float mean = mean_buf();

    // Define a Func to compute the variance
    Func variance_func;
    variance_func() = sum(pow(cast<float>(input(x + r.x, y + r.y)) - mean, 2)) / (kernel_size * kernel_size);

    // Realize the variance
    Buffer<float> variance_buf = variance_func.realize();
    return variance_buf();
}

void medianFilter(const std::string& filename, int kernel_size, float variance_threshold) {
    try {
        // Load the input image
        Buffer<uint8_t> input = Tools::load_image(filename);

        // Define the algorithm
        Var x, y, c;
        Func clamped = BoundaryConditions::repeat_edge(input);
        Func median;

        // Define a reduction domain for the specified kernel size
        int half_kernel = kernel_size / 2;
        RDom r(-half_kernel, kernel_size, -half_kernel, kernel_size);

        // Calculate the mean and variance using reduction domains
        Func mean_func, variance_func;
        mean_func(x, y, c) = sum(cast<float>(clamped(x + r.x, y + r.y, c))) / (kernel_size * kernel_size);
        Expr mean = mean_func(x, y, c);

        variance_func(x, y, c) = sum(pow(cast<float>(clamped(x + r.x, y + r.y, c)) - mean, 2)) / (kernel_size * kernel_size);
        Expr variance = variance_func(x, y, c);

        // Collect the values in the neighborhood
        std::vector<Expr> values;
        for (int i = 0; i < kernel_size * kernel_size; i++) {
            values.push_back(clamped(x + r.x, y + r.y, c));
        }

        // Apply the median filter if the variance is below the threshold
        median(x, y, c) = select(variance < variance_threshold, cast<uint8_t>(values[values.size() / 2]), clamped(x, y, c));

        // Schedule the algorithm
        median.vectorize(x, 16).parallel(y);

        // Create an output buffer
        Buffer<uint8_t> output(input.width(), input.height(), input.channels());

        // Realize the algorithm into the output buffer
        median.realize(output);

        // Save the output
        Tools::save_image(output, "bay_clean2.png");
    }
    catch (const Halide::CompileError& e) {
        std::cerr << "Halide Compile Error: " << e.what() << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Standard Exception: " << e.what() << std::endl;
    }
    catch (...) {
        std::cerr << "Unknown Exception occurred." << std::endl;
    }
}

void removeSaltAndPepperNoise(const std::string& filename, int kernel_size) {
    try {
        // Load the input image
        Buffer<uint8_t> input = Tools::load_image(filename);

        // Define the algorithm
        Var x, y, c;
        Func clamped = BoundaryConditions::repeat_edge(input);
        Func median;

        // Define a reduction domain for the specified kernel size
        int half_kernel = kernel_size / 2;
        RDom r(-half_kernel, kernel_size, -half_kernel, kernel_size);

        // Collect the values in the neighborhood
        std::vector<Expr> values;
        for (int i = 0; i < kernel_size * kernel_size; i++) {
            values.push_back(clamped(x + r.x, y + r.y, c));
        }

        // Sort the values to find the median
        for (int i = 0; i < values.size(); i++) {
            for (int j = i + 1; j < values.size(); j++) {
                values[i] = select(values[i] < values[j], values[i], values[j]);
            }
        }

        // Define the pure function with the correct type
        median(x, y, c) = cast<uint8_t>(0);

        // Define the update function
        median(x, y, c) = cast<uint8_t>(values[values.size() / 2]);

        // Schedule the algorithm
        median.vectorize(x, 16).parallel(y);

        // Create an output buffer
        Buffer<uint8_t> output(input.width(), input.height(), input.channels());

        // Realize the algorithm into the output buffer
        median.realize(output);

        // Save the output
        Tools::save_image(output, "bay_clean2.png");
    }
    catch (const Halide::CompileError& e) {
        std::cerr << "Halide Compile Error: " << e.what() << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Standard Exception: " << e.what() << std::endl;
    }
    catch (...) {
        std::cerr << "Unknown Exception occurred." << std::endl;
    }
}

void sharpenImage(const std::string& filename, float alpha) {
    try {
        // Load the input image
        Buffer<uint8_t> input = Tools::load_image(filename);

        // Define the algorithm
        Var x, y, c;
        Func clamped = BoundaryConditions::repeat_edge(input);
        Func blur_x, blur_y, sharpened;

        // Apply a simple Gaussian blur
        blur_x(x, y, c) = (clamped(x - 1, y, c) + 2 * clamped(x, y, c) + clamped(x + 1, y, c)) / 4;
        blur_y(x, y, c) = (blur_x(x, y - 1, c) + 2 * blur_x(x, y, c) + blur_x(x, y + 1, c)) / 4;

        // Calculate the sharpened image with a smaller alpha value
        sharpened(x, y, c) = cast<uint8_t>(clamped(x, y, c) + alpha * (clamped(x, y, c) - blur_y(x, y, c)));

        // Schedule the algorithm
        sharpened.vectorize(x, 16).parallel(y);

        // Create an output buffer
        Buffer<uint8_t> output(input.width(), input.height(), input.channels());

        // Realize the algorithm into the output buffer
        sharpened.realize(output);

        // Save the output
        Tools::save_image(output, "sharpened_image.png");
    }
    catch (const Halide::CompileError& e) {
        std::cerr << "Halide Compile Error: " << e.what() << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Standard Exception: " << e.what() << std::endl;
    }
    catch (...) {
        std::cerr << "Unknown Exception occurred." << std::endl;
    }
}

void sharpenImage2(const std::string& filename, float alpha) {
    try {
        // Load the input image
        Buffer<uint8_t> input = Tools::load_image(filename);

        // Define the algorithm
        Var x, y, c;
        Func clamped = BoundaryConditions::repeat_edge(input);
        Func blur_x, blur_y, sharpened;

        // Apply a simple Gaussian blur
        blur_x(x, y, c) = (clamped(x - 1, y, c) + 2 * clamped(x, y, c) + clamped(x + 1, y, c)) / 4;
        blur_y(x, y, c) = (blur_x(x, y - 1, c) + 2 * blur_x(x, y, c) + blur_x(x, y + 1, c)) / 4;

        // Calculate the sharpened image
        sharpened(x, y, c) = cast<uint8_t>(clamped(x, y, c) + alpha * (clamped(x, y, c) - blur_y(x, y, c)));

        // Schedule the algorithm
        sharpened.vectorize(x, 16).parallel(y);

        // Create an output buffer
        Buffer<uint8_t> output(input.width(), input.height(), input.channels());

        // Realize the algorithm into the output buffer
        sharpened.realize(output);

        // Save the output
        Tools::save_image(output, "sharpened_image.png");
    }
    catch (const Halide::CompileError& e) {
        std::cerr << "Halide Compile Error: " << e.what() << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Standard Exception: " << e.what() << std::endl;
    }
    catch (...) {
        std::cerr << "Unknown Exception occurred." << std::endl;
    }
}

void sharpenImageWithSobel(const std::string& filename, float alpha) {
    try {
        // Load the input image
        Buffer<uint8_t> input = Tools::load_image(filename);

        // Define the algorithm
        Var x, y, c;
        Func clamped = BoundaryConditions::repeat_edge(input);
        Func sobel_x, sobel_y, gradient_magnitude, sharpened;

        // Define Sobel kernels
        Expr sobel_x_kernel[3][3] = { {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1} };
        Expr sobel_y_kernel[3][3] = { {-1, -2, -1}, {0, 0, 0}, {1, 2, 1} };

        // Apply Sobel operator
        sobel_x(x, y, c) = (sobel_x_kernel[0][0] * clamped(x - 1, y - 1, c) + sobel_x_kernel[0][1] * clamped(x, y - 1, c) + sobel_x_kernel[0][2] * clamped(x + 1, y - 1, c) +
            sobel_x_kernel[1][0] * clamped(x - 1, y, c) + sobel_x_kernel[1][1] * clamped(x, y, c) + sobel_x_kernel[1][2] * clamped(x + 1, y, c) +
            sobel_x_kernel[2][0] * clamped(x - 1, y + 1, c) + sobel_x_kernel[2][1] * clamped(x, y + 1, c) + sobel_x_kernel[2][2] * clamped(x + 1, y + 1, c));

        sobel_y(x, y, c) = (sobel_y_kernel[0][0] * clamped(x - 1, y - 1, c) + sobel_y_kernel[0][1] * clamped(x, y - 1, c) + sobel_y_kernel[0][2] * clamped(x + 1, y - 1, c) +
            sobel_y_kernel[1][0] * clamped(x - 1, y, c) + sobel_y_kernel[1][1] * clamped(x, y, c) + sobel_y_kernel[1][2] * clamped(x + 1, y, c) +
            sobel_y_kernel[2][0] * clamped(x - 1, y + 1, c) + sobel_y_kernel[2][1] * clamped(x, y + 1, c) + sobel_y_kernel[2][2] * clamped(x + 1, y + 1, c));

        // Compute gradient magnitude
        gradient_magnitude(x, y, c) = sqrt(sobel_x(x, y, c) * sobel_x(x, y, c) + sobel_y(x, y, c) * sobel_y(x, y, c));

        // Enhance the edges
        sharpened(x, y, c) = cast<uint8_t>(clamped(x, y, c) + alpha * gradient_magnitude(x, y, c));

        // Schedule the algorithm
        sharpened.vectorize(x, 16).parallel(y);

        // Create an output buffer
        Buffer<uint8_t> output(input.width(), input.height(), input.channels());

        // Realize the algorithm into the output buffer
        sharpened.realize(output);

        // Save the output
        Tools::save_image(output, "sharpened_image_sobel.png");
    }
    catch (const Halide::CompileError& e) {
        std::cerr << "Halide Compile Error: " << e.what() << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Standard Exception: " << e.what() << std::endl;
    }
    catch (...) {
        std::cerr << "Unknown Exception occurred." << std::endl;
    }
}



void BayerDemosaicHalide(const std::string& inputFilename, const std::string& outputFilename) {
    try {
        // Load the TIFF image using the TIFF library
        TIFF* tiff = TIFFOpen(inputFilename.c_str(), "r");
        if (!tiff) {
            std::cerr << "Failed to open TIFF file: " << inputFilename << std::endl;
            return;
        }

        uint32_t width, height;
        uint16_t samplesPerPixel, bitsPerSample;
        TIFFGetField(tiff, TIFFTAG_IMAGEWIDTH, &width);
        TIFFGetField(tiff, TIFFTAG_IMAGELENGTH, &height);
        TIFFGetField(tiff, TIFFTAG_SAMPLESPERPIXEL, &samplesPerPixel);
        TIFFGetField(tiff, TIFFTAG_BITSPERSAMPLE, &bitsPerSample);

        uint64_t npixels = width * height;
        std::vector<uint8_t> raster(npixels * samplesPerPixel);
        if (!TIFFReadRGBAImage(tiff, width, height, (uint32*)raster.data(), 0)) {
            std::cerr << "Failed to read TIFF image: " << inputFilename << std::endl;
            TIFFClose(tiff);
            return;
        }
        TIFFClose(tiff);

        // Convert the loaded image data to a Halide Buffer
        Buffer<uint8_t> input(width, height, samplesPerPixel);
        for (uint32 y = 0; y < height; y++) {
            for (uint32 x = 0; x < width; x++) {
                for (uint16 c = 0; c < samplesPerPixel; c++) {
                    uint64_t index = static_cast<uint64_t>(y) * width + x;
                    if (index >= raster.size() / samplesPerPixel) {
                        std::cerr << "Index out of bounds: " << index << std::endl;
                        return;
                    }
                    input(x, y, c) = raster[index * samplesPerPixel + c];
                }
            }
        }

        // Define the Halide variables
        Var x("x"), y("y"), c("c");

        // Define the Halide function
        Func demosaic("demosaic");

        // Define the Bayer pattern
        Expr R = input(x, y);
        Expr G = input(x, y);
        Expr B = input(x, y);

        // Apply the Bayer pattern
        R = select((x % 2 == 0) && (y % 2 == 0), input(x, y),
            (x % 2 == 1) && (y % 2 == 0), (input(x - 1, y) + input(x + 1, y)) / 2,
            (x % 2 == 0) && (y % 2 == 1), (input(x, y - 1) + input(x, y + 1)) / 2,
            (input(x - 1, y - 1) + input(x + 1, y - 1) + input(x - 1, y + 1) + input(x + 1, y + 1)) / 4);

        G = select((x % 2 == 1) && (y % 2 == 1), input(x, y),
            (x % 2 == 0) && (y % 2 == 1), (input(x - 1, y) + input(x + 1, y)) / 2,
            (x % 2 == 1) && (y % 2 == 0), (input(x, y - 1) + input(x, y + 1)) / 2,
            (input(x - 1, y - 1) + input(x + 1, y - 1) + input(x - 1, y + 1) + input(x + 1, y + 1)) / 4);

        B = select((x % 2 == 1) && (y % 2 == 1), input(x, y),
            (x % 2 == 0) && (y % 2 == 1), (input(x - 1, y) + input(x + 1, y)) / 2,
            (x % 2 == 1) && (y % 2 == 0), (input(x, y - 1) + input(x, y + 1)) / 2,
            (input(x - 1, y - 1) + input(x + 1, y - 1) + input(x - 1, y + 1) + input(x + 1, y + 1)) / 4);

        // Combine the channels
        demosaic(x, y, c) = select(c == 0, R,
            c == 1, G,
            B);

        // Realize the function
        Buffer<uint8_t> output = demosaic.realize({ input.width(), input.height(), 3 });

        // Save the output image
        save_image(output, outputFilename);

        std::cout << "Demosaicing completed successfully!" << std::endl;
    }
    catch (const Halide::Error& e) {
        std::cerr << "Halide error: " << e.what() << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Standard exception: " << e.what() << std::endl;
    }
    catch (...) {
        std::cerr << "Unknown error occurred!" << std::endl;
    }
}



//this trakes an image... finds the sobel edges and then smoothes out the the image.
//then we combine the sobel edges with the smoothed image.  
void processImage(const std::string& filename, int kernel_size, float alpha) {
    try {
        // Load the input image
        Buffer<uint8_t> input = Tools::load_image(filename);

        // Define the algorithm
        Var x, y, c;
        Func clamped = BoundaryConditions::repeat_edge(input);
        Func sobel_x, sobel_y, gradient_magnitude, noise_reduced, combined;

        // Define Sobel kernels
        Expr sobel_x_kernel[3][3] = { {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1} };
        Expr sobel_y_kernel[3][3] = { {-1, -2, -1}, {0, 0, 0}, {1, 2, 1} };

        // Apply Sobel operator
        sobel_x(x, y, c) = (sobel_x_kernel[0][0] * clamped(x - 1, y - 1, c) + sobel_x_kernel[0][1] * clamped(x, y - 1, c) + sobel_x_kernel[0][2] * clamped(x + 1, y - 1, c) +
            sobel_x_kernel[1][0] * clamped(x - 1, y, c) + sobel_x_kernel[1][1] * clamped(x, y, c) + sobel_x_kernel[1][2] * clamped(x + 1, y, c) +
            sobel_x_kernel[2][0] * clamped(x - 1, y + 1, c) + sobel_x_kernel[2][1] * clamped(x, y + 1, c) + sobel_x_kernel[2][2] * clamped(x + 1, y + 1, c));

        sobel_y(x, y, c) = (sobel_y_kernel[0][0] * clamped(x - 1, y - 1, c) + sobel_y_kernel[0][1] * clamped(x, y - 1, c) + sobel_y_kernel[0][2] * clamped(x + 1, y - 1, c) +
            sobel_y_kernel[1][0] * clamped(x - 1, y, c) + sobel_y_kernel[1][1] * clamped(x, y, c) + sobel_y_kernel[1][2] * clamped(x + 1, y, c) +
            sobel_y_kernel[2][0] * clamped(x - 1, y + 1, c) + sobel_y_kernel[2][1] * clamped(x, y + 1, c) + sobel_y_kernel[2][2] * clamped(x + 1, y + 1, c));

        // Compute gradient magnitude
        gradient_magnitude(x, y, c) = sqrt(sobel_x(x, y, c) * sobel_x(x, y, c) + sobel_y(x, y, c) * sobel_y(x, y, c));

        // Apply salt and pepper noise reduction (median filter)
        Func clamped_noise_reduced = BoundaryConditions::repeat_edge(input);
        RDom r(-kernel_size / 2, kernel_size, -kernel_size / 2, kernel_size);
        Func median;
        median(x, y, c) = cast<uint8_t>(0);

        // Collect the values in the neighborhood
        std::vector<Expr> values;
        for (int i = 0; i < kernel_size * kernel_size; i++) {
            values.push_back(clamped_noise_reduced(x + r.x, y + r.y, c));
        }

        // Sort the values to find the median
        for (int i = 0; i < values.size(); i++) {
            for (int j = i + 1; j < values.size(); j++) {
                values[i] = select(values[i] < values[j], values[i], values[j]);
            }
        }

        // Select the median value
        Expr median_value = values[values.size() / 2];
        median(x, y, c) = median_value;

        // Combine the noise-reduced image with the original Sobel edges
        combined(x, y, c) = cast<uint8_t>(median(x, y, c) + alpha * gradient_magnitude(x, y, c));

        // Schedule the algorithm
        combined.vectorize(x, 16).parallel(y);

        // Create an output buffer
        Buffer<uint8_t> output(input.width(), input.height(), input.channels());

        // Realize the algorithm into the output buffer
        combined.realize(output);

        // Save the output
        Tools::save_image(output, "combined_image_sobel.png");
    }
    catch (const Halide::CompileError& e) {
        std::cerr << "Halide Compile Error: " << e.what() << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Standard Exception: " << e.what() << std::endl;
    }
    catch (...) {
        std::cerr << "Unknown Exception occurred." << std::endl;
    }
}
int main() {
    int brightness_factor = 40; // Change this value as needed
    //brightManual(brightness_factor);
    //brightHalide(brightness_factor);
    //brightHalide2(brightness_factor);
    //SobelDetectors();
    //removeSaltAndPepperNoise("bay_dust.jpg");
    //removeSaltAndPepperNoise2("bay_dust.jpg", 3);
    //sharpenImage("bay_clean.png", .02);
    //brightHalide2("sharpened_image.png", brightness_factor);
    //sharpenImageWithSobel("brighterHalide2.png", .4);
    //BayerDemosaicHallide("bay_dust.jpg", "color_imageH.png");
    //double sharpenImageTime = timeFunction(sharpenImage, "bay_dust.jpg", .2);

    double medianFilterTime = timeFunction(medianFilter, "bay_dust.jpg", 3, 80); 

    double halideMosaicTime = timeFunction(BayerDemosaicHalide, "court0_LowerLeftQuadrant.tiff", "test1.png");
    std::cout << "Halide execution time: " << halideMosaicTime << " seconds" << std::endl;

    //processImage("bay_dust.jpg", 3, .01);
    return 0;
}

/*
Halide::Target target = Halide::get_host_target();
target.set_feature(Halide::Target::Vulkan);

Halide::Buffer<uint8_t> output =
brighter.realize({ input.width(), input.height(), input.channels() }, target);
*/