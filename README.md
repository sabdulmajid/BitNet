# bitnet.cpp
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
![version](https://img.shields.io/badge/version-1.0-blue)

[<img src="./assets/header_model_release.png" alt="BitNet Model on Hugging Face" width="800"/>](https://huggingface.co/microsoft/BitNet-b1.58-2B-4T)

Try it out via this [demo](https://demo-bitnet-h0h8hcfqeqhrf5gf.canadacentral-01.azurewebsites.net/), or build and run it on your own [CPU](https://github.com/microsoft/BitNet?tab=readme-ov-file#build-from-source) or [GPU](https://github.com/microsoft/BitNet/blob/main/gpu/README.md).

bitnet.cpp is the official inference framework for 1-bit LLMs (e.g., BitNet b1.58). It offers a suite of optimized kernels, that support **fast** and **lossless** inference of 1.58-bit models on CPU and GPU (NPU support will coming next).

The first release of bitnet.cpp is to support inference on CPUs. bitnet.cpp achieves speedups of **1.37x** to **5.07x** on ARM CPUs, with larger models experiencing greater performance gains. Additionally, it reduces energy consumption by **55.4%** to **70.0%**, further boosting overall efficiency. On x86 CPUs, speedups range from **2.37x** to **6.17x** with energy reductions between **71.9%** to **82.2%**. Furthermore, bitnet.cpp can run a 100B BitNet b1.58 model on a single CPU, achieving speeds comparable to human reading (5-7 tokens per second), significantly enhancing the potential for running LLMs on local devices. Please refer to the [technical report](https://arxiv.org/abs/2410.16144) for more details.

**Latest optimization** introduces parallel kernel implementations with configurable tiling and embedding quantization support, achieving **1.15x to 2.1x** additional speedup over the original implementation across different hardware platforms and workloads. For detailed technical information, see the [optimization guide](src/README.md).

## Experimental Qwen Retrofit Work in This Fork

This section describes experiments in this fork, not an upstream `bitnet.cpp`
release claim. It tests whether pretrained dense Hugging Face models can be
retrofitted into BitNet-style W1.58A8 models. The current answer is deliberately
conservative:

- **Blind post-training ternarization is not viable** for the tested Qwen
  checkpoints. It causes catastrophic perplexity collapse.
- **QAT/distillation is materially better than naive PTQ**, which shows that
  training under ternary forward constraints recovers real signal.
- **The current QAT checkpoints are not deployment-quality yet.** They remain
  significantly worse than the FP references. A Qwen2.5-0.5B QAT checkpoint can
  be converted to GGUF and packed as I2_S, but the resulting text is still
  degenerate on a simple smoke prompt.
- **PyTorch ternary simulation is not the speed path.** On the Xeon Silver 4116
  host, the exported ternary checkpoints are smaller in memory but slower than
  FP under PyTorch because the probe dequantizes into dense matmuls. Real
  product claims require GGUF/I2_S or TL2 execution through `bitnet.cpp`.
- **Packed I2_S runtime is fast, but speed does not rescue blind
  ternarization.** On the same Xeon, Qwen2.5-0.5B I2_S GGUF runs much faster
  than F16/Q8 decode, but blind I2_S conversion collapses a basic generation
  prompt to punctuation.

Current evidence is tracked in
[benchmarks/results/qwen_retrofit_2026-05-03.md](benchmarks/results/qwen_retrofit_2026-05-03.md).
The benchmark harnesses are in [benchmarks/](benchmarks/).

### Current Perplexity Snapshot

| model | method | WikiText PPL | FineWeb-heldout PPL |
| --- | --- | ---: | ---: |
| Qwen2.5-0.5B | FP reference | 20.461 | 14.124 |
| Qwen2.5-0.5B | naive PTQ ternary | 169,414.428 | 608,726.749 |
| Qwen2.5-0.5B | QAT/distilled ternary | 1,079.167 | 373.775 |
| Qwen2.5-1.5B | FP reference | 13.901 | 10.269 |
| Qwen2.5-1.5B | naive PTQ ternary | 3,813,121.803 | 9,582,923.269 |
| Qwen2.5-1.5B | QAT/distilled ternary | 86.414 | 40.398 |

These numbers are BF16/CUDA quality measurements using PyTorch simulation of
W1.58A8 math. They are **not** `bitnet.cpp` CPU throughput claims.

### Current Official lm-eval Snapshot

EleutherAI `lm-eval` 0.4.11 was run on 100-example slices for ten tasks using
Qwen2.5-1.5B FP, naive PTQ ternary, and QAT/distilled ternary. Where a task
reports `acc_norm`, that metric is shown; otherwise raw `acc` is shown.

| task | metric | FP | naive PTQ | QAT ternary |
| --- | --- | ---: | ---: | ---: |
| ARC-Challenge | acc_norm | 0.410 | 0.300 | 0.300 |
| ARC-Easy | acc_norm | 0.760 | 0.220 | 0.510 |
| BoolQ | acc | 0.690 | 0.400 | 0.700 |
| COPA | acc | 0.830 | 0.510 | 0.640 |
| HellaSwag | acc_norm | 0.660 | 0.290 | 0.460 |
| OpenBookQA | acc_norm | 0.350 | 0.290 | 0.280 |
| PIQA | acc_norm | 0.800 | 0.580 | 0.590 |
| SciQ | acc_norm | 0.960 | 0.210 | 0.640 |
| TruthfulQA MC1 | acc | 0.280 | 0.220 | 0.200 |
| WinoGrande | acc | 0.720 | 0.490 | 0.580 |

Mean over these displayed metrics: FP 0.646, naive PTQ 0.351, QAT ternary
0.490. This supports the narrow claim that QAT/distillation recovers substantial
signal over blind ternarization. It does **not** support a claim that the
current ternary model preserves FP quality.

### Packed GGUF CPU Runtime Snapshot

CPU runtime: Intel Xeon Silver 4116, 12 threads, `llama-bench -p 512 -n 128
-ngl 0 -r 3`, no BLAS, AVX-512 available, llama.cpp submodule commit
`1f86f058`. The I2_S path is a real packed GGUF CPU measurement. It is not yet
an exact import of `ternary_state_dict.pt`; the tested GGUF artifacts were
created by converting dense HF checkpoints to F16 GGUF and then running
`llama-quantize`.

| source | GGUF type | file size | prefill tok/s | decode tok/s | smoke prompt result |
| --- | --- | ---: | ---: | ---: | --- |
| Qwen2.5-0.5B FP | F16 | 948 MiB | 331.82 | 16.39 | sensible |
| Qwen2.5-0.5B FP | Q8_0 | 507 MiB | 391.40 | 28.84 | sensible |
| Qwen2.5-0.5B FP | Q4_K_M | 379 MiB | 213.67 | 35.70 | sensible |
| Qwen2.5-0.5B FP | I2_S | 230 MiB | 532.24 | 53.11 | degenerate punctuation |
| Qwen2.5-0.5B QAT step-1000 | F16 | 1,208 MiB | 332.13 | 16.26 | degenerate text |
| Qwen2.5-0.5B QAT step-1000 | I2_S | 490 MiB | 525.52 | 49.97 | degenerate punctuation |
| Qwen2.5-1.5B FP | F16 | 2,950 MiB | 105.30 | 5.52 | sensible |
| Qwen2.5-1.5B FP | Q8_0 | 1,570 MiB | 135.45 | 10.07 | sensible |
| Qwen2.5-1.5B FP | Q4_K_M | 940 MiB | 95.17 | 15.72 | sensible |
| Qwen2.5-1.5B FP | I2_S | 766 MiB | 205.66 | 18.41 | repeated-token collapse |
| Qwen2.5-1.5B QAT step-5000 | F16 | 3,396 MiB | 105.21 | 5.52 | degenerate text |
| Qwen2.5-1.5B QAT step-5000 | I2_S | 1,211 MiB | 203.59 | 17.97 | repeated-token collapse |
| Qwen2.5-1.5B static ternary | F16 materialized | 3,396 MiB | 105.28 | 5.51 | sensible |
| Qwen2.5-1.5B static ternary | TQ2_0 | 1,219 MiB | 158.52 | 18.38 | sensible |
| Qwen2.5-1.5B static ternary | I2_S | 1,211 MiB | 190.79 | 18.61 | degenerate punctuation |

Interpretation: the CPU backend can execute packed I2_S quickly on this 2017
Xeon. The blocking problem is quality, not kernel availability. Standard Q8_0
and Q4_K_M preserve the simple prompt, while I2_S does not. Q4_K_M should be
read with care for Qwen2.5-0.5B because many tensors require fallback
quantization due column divisibility constraints; Qwen2.5-1.5B did not report
that fallback warning.

### Packed GGUF Perplexity Snapshot

`llama-perplexity` was run on a fixed 16-chunk WikiText-2 test excerpt
(8,192 tokens, `-c 512`, 12 threads) for Qwen2.5-1.5B GGUF artifacts.

| source | GGUF type | WikiText excerpt PPL | prompt-eval tok/s |
| --- | --- | ---: | ---: |
| Qwen2.5-1.5B FP | F16 | 12.2806 | 84.11 |
| Qwen2.5-1.5B FP | Q8_0 | 12.3207 | 104.28 |
| Qwen2.5-1.5B FP | Q4_K_M | 12.8452 | 75.53 |
| Qwen2.5-1.5B FP | I2_S | 1.206e51 | 140.03 |
| Qwen2.5-1.5B QAT step-5000 dense GGUF | F16 | 2728.9322 | 83.79 |
| Qwen2.5-1.5B QAT step-5000 dense GGUF | I2_S | 7.619e59 | 137.73 |
| Qwen2.5-1.5B static ternary | F16 materialized | 83.8300 | 83.27 |
| Qwen2.5-1.5B static ternary | TQ2_0 | 84.0553 | 113.75 |
| Qwen2.5-1.5B static ternary | I2_S | NaN | 132.97 |

This is the cleanest packed-runtime evidence so far: conventional Q8_0 and
Q4_K_M retain the FP language-modeling likelihood, while blind I2_S
ternarization destroys it. Materializing `ternary_state_dict.pt` as dense F16
recovers the PyTorch static-ternary quality, and llama.cpp `TQ2_0` preserves
that quality while giving a 2.06 bpw ternary GGUF artifact. The current I2_S
quantization path is faster but numerically invalid for this trained sparse
ternary artifact, so a native I2_S writer/kernel audit remains required before
claiming BitNet I2_S deployment quality.

### Xeon PyTorch Runtime Probe

CPU probe: Intel Xeon Silver 4116, 12 physical cores / 24 threads, AVX-512,
PyTorch FP32, 12 Torch threads, 512-token prompt, 32 generated tokens, median of
three measured repeats. These are PyTorch loader numbers, not packed
`bitnet.cpp` kernel numbers.

| model | method | prefill tok/s | gen tok/s | RSS GiB | model GiB | ternary GiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-0.5B | FP reference | 330.69 | 5.20 | 2.716 | 1.840 | - |
| Qwen2.5-0.5B | naive PTQ ternary | 244.82 | 2.03 | 3.370 | 0.841 | 0.587 |
| Qwen2.5-0.5B | QAT/distilled ternary | 219.71 | 1.41 | 2.173 | 0.967 | 0.968 |
| Qwen2.5-1.5B | FP reference | 118.74 | 1.95 | 6.631 | 5.751 | - |
| Qwen2.5-1.5B | naive PTQ ternary | 79.93 | 0.48 | 4.748 | 2.090 | 1.655 |
| Qwen2.5-1.5B | QAT/distilled ternary | 74.34 | 0.41 | 4.405 | 2.307 | 2.308 |

### Current Task-Accuracy Snapshot

The in-repo multiple-choice evaluator covers 100-example validation slices
for PIQA, ARC-Easy, ARC-Challenge, and HellaSwag. This is a fast regression
tool; the official `lm-eval` snapshot above is the stronger current evidence.
The strongest current ternary result in this fast local harness is
Qwen2.5-1.5B QAT/distilled:

| task | FP Qwen2.5-1.5B acc | naive PTQ acc | QAT ternary acc |
| --- | ---: | ---: | ---: |
| PIQA | 0.760 | 0.550 | 0.650 |
| ARC-Easy | 0.760 | 0.300 | 0.550 |
| ARC-Challenge | 0.440 | 0.190 | 0.320 |
| HellaSwag | 0.470 | 0.230 | 0.360 |

<img src="./assets/performance.png" alt="performance_comparison" width="800"/>


## Demo

A demo of bitnet.cpp running a BitNet b1.58 3B model on Apple M2:

https://github.com/user-attachments/assets/7f46b736-edec-4828-b809-4be780a3e5b1

## What's New:
- 01/15/2026 [BitNet CPU Inference Optimization](https://github.com/microsoft/BitNet/blob/main/src/README.md) ![NEW](https://img.shields.io/badge/NEW-red)
- 05/20/2025 [BitNet Official GPU inference kernel](https://github.com/microsoft/BitNet/blob/main/gpu/README.md)
- 04/14/2025 [BitNet Official 2B Parameter Model on Hugging Face](https://huggingface.co/microsoft/BitNet-b1.58-2B-4T)
- 02/18/2025 [Bitnet.cpp: Efficient Edge Inference for Ternary LLMs](https://arxiv.org/abs/2502.11880)
- 11/08/2024 [BitNet a4.8: 4-bit Activations for 1-bit LLMs](https://arxiv.org/abs/2411.04965)
- 10/21/2024 [1-bit AI Infra: Part 1.1, Fast and Lossless BitNet b1.58 Inference on CPUs](https://arxiv.org/abs/2410.16144)
- 10/17/2024 bitnet.cpp 1.0 released.
- 03/21/2024 [The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ](https://github.com/microsoft/unilm/blob/master/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf)
- 02/27/2024 [The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/abs/2402.17764)
- 10/17/2023 [BitNet: Scaling 1-bit Transformers for Large Language Models](https://arxiv.org/abs/2310.11453)

## Acknowledgements

This project is based on the [llama.cpp](https://github.com/ggerganov/llama.cpp) framework. We would like to thank all the authors for their contributions to the open-source community. Also, bitnet.cpp's kernels are built on top of the Lookup Table methodologies pioneered in [T-MAC](https://github.com/microsoft/T-MAC/). For inference of general low-bit LLMs beyond ternary models, we recommend using T-MAC.
## Official Models
<table>
    </tr>
    <tr>
        <th rowspan="2">Model</th>
        <th rowspan="2">Parameters</th>
        <th rowspan="2">CPU</th>
        <th colspan="3">Kernel</th>
    </tr>
    <tr>
        <th>I2_S</th>
        <th>TL1</th>
        <th>TL2</th>
    </tr>
    <tr>
        <td rowspan="2"><a href="https://huggingface.co/microsoft/BitNet-b1.58-2B-4T">BitNet-b1.58-2B-4T</a></td>
        <td rowspan="2">2.4B</td>
        <td>x86</td>
        <td>&#9989;</td>
        <td>&#10060;</td>
        <td>&#9989;</td>
    </tr>
    <tr>
        <td>ARM</td>
        <td>&#9989;</td>
        <td>&#9989;</td>
        <td>&#10060;</td>
    </tr>
</table>

## Supported Models
❗️**We use existing 1-bit LLMs available on [Hugging Face](https://huggingface.co/) to demonstrate the inference capabilities of bitnet.cpp. We hope the release of bitnet.cpp will inspire the development of 1-bit LLMs in large-scale settings in terms of model size and training tokens.**

<table>
    </tr>
    <tr>
        <th rowspan="2">Model</th>
        <th rowspan="2">Parameters</th>
        <th rowspan="2">CPU</th>
        <th colspan="3">Kernel</th>
    </tr>
    <tr>
        <th>I2_S</th>
        <th>TL1</th>
        <th>TL2</th>
    </tr>
    <tr>
        <td rowspan="2"><a href="https://huggingface.co/1bitLLM/bitnet_b1_58-large">bitnet_b1_58-large</a></td>
        <td rowspan="2">0.7B</td>
        <td>x86</td>
        <td>&#9989;</td>
        <td>&#10060;</td>
        <td>&#9989;</td>
    </tr>
    <tr>
        <td>ARM</td>
        <td>&#9989;</td>
        <td>&#9989;</td>
        <td>&#10060;</td>
    </tr>
    <tr>
        <td rowspan="2"><a href="https://huggingface.co/1bitLLM/bitnet_b1_58-3B">bitnet_b1_58-3B</a></td>
        <td rowspan="2">3.3B</td>
        <td>x86</td>
        <td>&#10060;</td>
        <td>&#10060;</td>
        <td>&#9989;</td>
    </tr>
    <tr>
        <td>ARM</td>
        <td>&#10060;</td>
        <td>&#9989;</td>
        <td>&#10060;</td>
    </tr>
    <tr>
        <td rowspan="2"><a href="https://huggingface.co/HF1BitLLM/Llama3-8B-1.58-100B-tokens">Llama3-8B-1.58-100B-tokens</a></td>
        <td rowspan="2">8.0B</td>
        <td>x86</td>
        <td>&#9989;</td>
        <td>&#10060;</td>
        <td>&#9989;</td>
    </tr>
    <tr>
        <td>ARM</td>
        <td>&#9989;</td>
        <td>&#9989;</td>
        <td>&#10060;</td>
    </tr>
    <tr>
        <td rowspan="2"><a href="https://huggingface.co/collections/tiiuae/falcon3-67605ae03578be86e4e87026">Falcon3 Family</a></td>
        <td rowspan="2">1B-10B</td>
        <td>x86</td>
        <td>&#9989;</td>
        <td>&#10060;</td>
        <td>&#9989;</td>
    </tr>
    <tr>
        <td>ARM</td>
        <td>&#9989;</td>
        <td>&#9989;</td>
        <td>&#10060;</td>
    </tr>
    <tr>
        <td rowspan="2"><a href="https://huggingface.co/collections/tiiuae/falcon-edge-series-6804fd13344d6d8a8fa71130">Falcon-E Family</a></td>
        <td rowspan="2">1B-3B</td>
        <td>x86</td>
        <td>&#9989;</td>
        <td>&#10060;</td>
        <td>&#9989;</td>
    </tr>
    <tr>
        <td>ARM</td>
        <td>&#9989;</td>
        <td>&#9989;</td>
        <td>&#10060;</td>
    </tr>
</table>



## Installation

### Requirements
- python>=3.9
- cmake>=3.22
- clang>=18
    - For Windows users, install [Visual Studio 2022](https://visualstudio.microsoft.com/downloads/). In the installer, toggle on at least the following options(this also automatically installs the required additional tools like CMake):
        -  Desktop-development with C++
        -  C++-CMake Tools for Windows
        -  Git for Windows
        -  C++-Clang Compiler for Windows
        -  MS-Build Support for LLVM-Toolset (clang)
    - For Debian/Ubuntu users, you can download with [Automatic installation script](https://apt.llvm.org/)

        `bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)"`
- conda (highly recommend)

### Build from source

> [!IMPORTANT]
> If you are using Windows, please remember to always use a Developer Command Prompt / PowerShell for VS2022 for the following commands. Please refer to the FAQs below if you see any issues.

1. Clone the repo
```bash
git clone --recursive https://github.com/microsoft/BitNet.git
cd BitNet
```
2. Install the dependencies
```bash
# (Recommended) Create a new conda environment
conda create -n bitnet-cpp python=3.9
conda activate bitnet-cpp

pip install -r requirements.txt
```
3. Build the project
```bash
# Manually download the model and run with local path
huggingface-cli download microsoft/BitNet-b1.58-2B-4T-gguf --local-dir models/BitNet-b1.58-2B-4T
python setup_env.py -md models/BitNet-b1.58-2B-4T -q i2_s

```
<pre>
usage: setup_env.py [-h] [--hf-repo {1bitLLM/bitnet_b1_58-large,1bitLLM/bitnet_b1_58-3B,HF1BitLLM/Llama3-8B-1.58-100B-tokens,tiiuae/Falcon3-1B-Instruct-1.58bit,tiiuae/Falcon3-3B-Instruct-1.58bit,tiiuae/Falcon3-7B-Instruct-1.58bit,tiiuae/Falcon3-10B-Instruct-1.58bit}] [--model-dir MODEL_DIR] [--log-dir LOG_DIR] [--quant-type {i2_s,tl1}] [--quant-embd]
                    [--use-pretuned]

Setup the environment for running inference

optional arguments:
  -h, --help            show this help message and exit
  --hf-repo {1bitLLM/bitnet_b1_58-large,1bitLLM/bitnet_b1_58-3B,HF1BitLLM/Llama3-8B-1.58-100B-tokens,tiiuae/Falcon3-1B-Instruct-1.58bit,tiiuae/Falcon3-3B-Instruct-1.58bit,tiiuae/Falcon3-7B-Instruct-1.58bit,tiiuae/Falcon3-10B-Instruct-1.58bit}, -hr {1bitLLM/bitnet_b1_58-large,1bitLLM/bitnet_b1_58-3B,HF1BitLLM/Llama3-8B-1.58-100B-tokens,tiiuae/Falcon3-1B-Instruct-1.58bit,tiiuae/Falcon3-3B-Instruct-1.58bit,tiiuae/Falcon3-7B-Instruct-1.58bit,tiiuae/Falcon3-10B-Instruct-1.58bit}
                        Model used for inference
  --model-dir MODEL_DIR, -md MODEL_DIR
                        Directory to save/load the model
  --log-dir LOG_DIR, -ld LOG_DIR
                        Directory to save the logging info
  --quant-type {i2_s,tl1}, -q {i2_s,tl1}
                        Quantization type
  --quant-embd          Quantize the embeddings to f16
  --use-pretuned, -p    Use the pretuned kernel parameters
</pre>
## Usage
### Basic usage
```bash
# Run inference with the quantized model
python run_inference.py -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf -p "You are a helpful assistant" -cnv
```
<pre>
usage: run_inference.py [-h] [-m MODEL] [-n N_PREDICT] -p PROMPT [-t THREADS] [-c CTX_SIZE] [-temp TEMPERATURE] [-cnv]

Run inference

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        Path to model file
  -n N_PREDICT, --n-predict N_PREDICT
                        Number of tokens to predict when generating text
  -p PROMPT, --prompt PROMPT
                        Prompt to generate text from
  -t THREADS, --threads THREADS
                        Number of threads to use
  -c CTX_SIZE, --ctx-size CTX_SIZE
                        Size of the prompt context
  -temp TEMPERATURE, --temperature TEMPERATURE
                        Temperature, a hyperparameter that controls the randomness of the generated text
  -cnv, --conversation  Whether to enable chat mode or not (for instruct models.)
                        (When this option is turned on, the prompt specified by -p will be used as the system prompt.)
</pre>

### Benchmark
We provide scripts to run the inference benchmark providing a model.

```  
usage: e2e_benchmark.py -m MODEL [-n N_TOKEN] [-p N_PROMPT] [-t THREADS]  
   
Setup the environment for running the inference  
   
required arguments:  
  -m MODEL, --model MODEL  
                        Path to the model file. 
   
optional arguments:  
  -h, --help  
                        Show this help message and exit. 
  -n N_TOKEN, --n-token N_TOKEN  
                        Number of generated tokens. 
  -p N_PROMPT, --n-prompt N_PROMPT  
                        Prompt to generate text from. 
  -t THREADS, --threads THREADS  
                        Number of threads to use. 
```  
   
Here's a brief explanation of each argument:  
   
- `-m`, `--model`: The path to the model file. This is a required argument that must be provided when running the script.  
- `-n`, `--n-token`: The number of tokens to generate during the inference. It is an optional argument with a default value of 128.  
- `-p`, `--n-prompt`: The number of prompt tokens to use for generating text. This is an optional argument with a default value of 512.  
- `-t`, `--threads`: The number of threads to use for running the inference. It is an optional argument with a default value of 2.  
- `-h`, `--help`: Show the help message and exit. Use this argument to display usage information.  
   
For example:  
   
```sh  
python utils/e2e_benchmark.py -m /path/to/model -n 200 -p 256 -t 4  
```  
   
This command would run the inference benchmark using the model located at `/path/to/model`, generating 200 tokens from a 256 token prompt, utilizing 4 threads.  

For the model layout that do not supported by any public model, we provide scripts to generate a dummy model with the given model layout, and run the benchmark on your machine:

```bash
python utils/generate-dummy-bitnet-model.py models/bitnet_b1_58-large --outfile models/dummy-bitnet-125m.tl1.gguf --outtype tl1 --model-size 125M

# Run benchmark with the generated model, use -m to specify the model path, -p to specify the prompt processed, -n to specify the number of token to generate
python utils/e2e_benchmark.py -m models/dummy-bitnet-125m.tl1.gguf -p 512 -n 128
```

### Convert from `.safetensors` Checkpoints

```sh
# Prepare the .safetensors model file
huggingface-cli download microsoft/bitnet-b1.58-2B-4T-bf16 --local-dir ./models/bitnet-b1.58-2B-4T-bf16

# Convert to gguf model
python ./utils/convert-helper-bitnet.py ./models/bitnet-b1.58-2B-4T-bf16
```

### FAQ (Frequently Asked Questions)📌 

#### Q1: The build dies with errors building llama.cpp due to issues with std::chrono in log.cpp?

**A:**
This is an issue introduced in recent version of llama.cpp. Please refer to this [commit](https://github.com/tinglou/llama.cpp/commit/4e3db1e3d78cc1bcd22bcb3af54bd2a4628dd323) in the [discussion](https://github.com/abetlen/llama-cpp-python/issues/1942) to fix this issue.

#### Q2: How to build with clang in conda environment on windows?

**A:** 
Before building the project, verify your clang installation and access to Visual Studio tools by running:
```
clang -v
```

This command checks that you are using the correct version of clang and that the Visual Studio tools are available. If you see an error message such as:
```
'clang' is not recognized as an internal or external command, operable program or batch file.
```

It indicates that your command line window is not properly initialized for Visual Studio tools.

• If you are using Command Prompt, run:
```
"C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\Tools\VsDevCmd.bat" -startdir=none -arch=x64 -host_arch=x64
```

• If you are using Windows PowerShell, run the following commands:
```
Import-Module "C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\Tools\Microsoft.VisualStudio.DevShell.dll" Enter-VsDevShell 3f0e31ad -SkipAutomaticLocation -DevCmdArguments "-arch=x64 -host_arch=x64"
```

These steps will initialize your environment and allow you to use the correct Visual Studio tools.
