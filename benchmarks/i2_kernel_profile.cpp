// Isolate I2 activation quantization and matrix multiplication costs.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <string>
#include <vector>

extern "C" {
void quantize_row_i8_s(const float * x, void * y, int64_t n, float * scale, int32_t * sum);
void ggml_gemm_i2_i8_s(
    int n,
    float * output,
    size_t output_stride,
    const void * weights,
    const void * activations,
    int activation_rows,
    int output_rows);
}

namespace {

using Clock = std::chrono::steady_clock;

struct Shape {
    int input;
    int output;
    int tokens;
};

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const size_t middle = values.size() / 2;
    if (values.size() % 2 == 0) {
        return (values[middle - 1] + values[middle]) / 2.0;
    }
    return values[middle];
}

template <typename Function>
double benchmark_us(Function && function, int warmups, int iterations) {
    for (int i = 0; i < warmups; ++i) {
        function();
    }

    std::vector<double> samples;
    samples.reserve(iterations);
    for (int i = 0; i < iterations; ++i) {
        const auto start = Clock::now();
        function();
        const auto stop = Clock::now();
        samples.push_back(
            std::chrono::duration<double, std::micro>(stop - start).count());
    }
    return median(std::move(samples));
}

std::vector<uint8_t> make_packed_weights(int input, int output, std::mt19937 & generator) {
    std::uniform_int_distribution<int> code(0, 2);
    std::vector<uint8_t> weights(static_cast<size_t>(input) * output / 4, 0);
    for (size_t i = 0; i < weights.size(); ++i) {
        uint8_t packed = 0;
        for (int lane = 0; lane < 4; ++lane) {
            packed |= static_cast<uint8_t>(code(generator) << (6 - 2 * lane));
        }
        weights[i] = packed;
    }
    return weights;
}

void profile_shape(const Shape & shape, int warmups, int iterations) {
    if (shape.input % 128 != 0 || shape.output <= 0 || shape.tokens <= 0) {
        std::fprintf(stderr, "invalid shape: input must be divisible by 128\n");
        std::exit(EXIT_FAILURE);
    }

    std::mt19937 generator(0xB17);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> activations(static_cast<size_t>(shape.input) * shape.tokens);
    for (float & value : activations) {
        value = normal(generator);
    }
    std::vector<int8_t> quantized(activations.size());
    std::vector<float> scales(shape.tokens);
    std::vector<int32_t> sums(shape.tokens);
    std::vector<uint8_t> weights = make_packed_weights(shape.input, shape.output, generator);
    std::vector<float> output(static_cast<size_t>(shape.output) * shape.tokens);

    const auto quantize = [&]() {
        for (int row = 0; row < shape.tokens; ++row) {
            quantize_row_i8_s(
                activations.data() + static_cast<size_t>(row) * shape.input,
                quantized.data() + static_cast<size_t>(row) * shape.input,
                shape.input,
                scales.data() + row,
                sums.data() + row);
        }
    };
    quantize();

    const auto multiply = [&]() {
        ggml_gemm_i2_i8_s(
            shape.input,
            output.data(),
            shape.output,
            weights.data(),
            quantized.data(),
            shape.tokens,
            shape.output);
    };

    multiply();
    double max_abs_error = 0.0;
    for (int token = 0; token < shape.tokens; ++token) {
        for (int row = 0; row < shape.output; ++row) {
            int32_t expected = 0;
            const uint8_t * packed_row =
                weights.data() + static_cast<size_t>(row) * shape.input / 4;
            const int8_t * activation_row =
                quantized.data() + static_cast<size_t>(token) * shape.input;
            for (int block = 0; block < shape.input / 128; ++block) {
                for (int lane = 0; lane < 32; ++lane) {
                    const uint8_t packed = packed_row[block * 32 + lane];
                    for (int group = 0; group < 4; ++group) {
                        const int code = (packed >> (6 - 2 * group)) & 0x03;
                        expected += code * activation_row[block * 128 + group * 32 + lane];
                    }
                }
            }
            const float actual = output[static_cast<size_t>(token) * shape.output + row];
            max_abs_error = std::max(
                max_abs_error, std::abs(static_cast<double>(actual) - expected));
        }
    }

    const double quantize_us = benchmark_us(quantize, warmups, iterations);
    const double multiply_us = benchmark_us(multiply, warmups, iterations);
    const double combined_us = quantize_us + multiply_us;
    const double quantize_fraction = quantize_us / combined_us;

    const double checksum = std::accumulate(output.begin(), output.end(), 0.0);
    std::printf(
        "{\"input\":%d,\"output\":%d,\"tokens\":%d,"
        "\"quantize_us\":%.3f,\"multiply_us\":%.3f,"
        "\"quantize_fraction\":%.8f,\"checksum\":%.9g,"
        "\"max_abs_error\":%.9g}\n",
        shape.input,
        shape.output,
        shape.tokens,
        quantize_us,
        multiply_us,
        quantize_fraction,
        checksum,
        max_abs_error);
}

}  // namespace

int main(int argc, char ** argv) {
    int tokens = 32;
    int warmups = 5;
    int iterations = 15;
    if (argc > 1) {
        tokens = std::stoi(argv[1]);
    }
    if (argc > 2) {
        iterations = std::stoi(argv[2]);
    }

    const std::vector<Shape> shapes = {
        {896, 896, tokens},
        {896, 128, tokens},
        {896, 4864, tokens},
        {4864, 896, tokens},
    };
    for (const Shape & shape : shapes) {
        profile_shape(shape, warmups, iterations);
    }
    return EXIT_SUCCESS;
}
