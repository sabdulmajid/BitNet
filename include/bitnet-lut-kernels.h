#pragma once

// Placeholder for non-LUT builds.
//
// setup_env.py or the TL1/TL2 codegen path overwrites this file with generated
// lookup-table kernels when GGML_BITNET_ARM_TL1 or GGML_BITNET_X86_TL2 is
// enabled. I2_S-only builds still include this path through CMake, but do not
// reference any symbols from it.
