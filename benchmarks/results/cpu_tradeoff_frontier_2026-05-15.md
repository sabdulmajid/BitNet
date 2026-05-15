# CPU Tradeoff Frontier Audit, 2026-05-15

I2_SR is a speed-oriented proof of row-scale ternary runtime semantics. It improves decode speed versus FP16 and is faster than Q4_K_M in the audited run, but it is larger than Q4_K_M and has much worse PPL. It should not be claimed as a quality/storage win over mature Q4 quantization.



## Headline Ratios

Compared with FP Q4_K_M, row-scale `I2_SR` has `1.288133x` file size, `1.268574x` RSS at ctx 512, `2.298818x` prefill throughput, `1.190617x` decode throughput, and `3.032323x` PPL.



Pareto frontier over PPL, file size, RSS, prefill, and decode: `FP F16, FP Q8_0, FP Q4_K_M, row TQ2_0, row I2_S, row I2_SR`.



## Rows

| artifact | file MiB | RSS512 GiB | PPL | prefill tok/s | decode tok/s | file/FP | PPL/FP | decode/FP | file/Q4 | PPL/Q4 | decode/Q4 | dominated by |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FP F16 | 2.950350e+03 | 2.948406 | 12.280800 | 114.468162 | 5.555998 | 1.000000 | 1.000000 | 1.000000 | 3.137448 | 0.958599 | 0.346965 | - |
| FP Q8_0 | 1.570291e+03 | 1.600719 | 12.305600 | 124.864246 | 10.131914 | 0.532239 | 1.002019 | 1.823599 | 1.669872 | 0.960535 | 0.632726 | - |
| FP Q4_K_M | 940.366150 | 0.985497 | 12.811200 | 92.077037 | 16.013125 | 0.318730 | 1.043189 | 2.882133 | 1.000000 | 1.000000 | 1.000000 | - |
| row TQ2_0 | 1.218612e+03 | 1.257366 | 38.822400 | 169.460897 | 18.675323 | 0.413040 | 3.161227 | 3.361290 | 1.295891 | 3.030348 | 1.166251 | - |
| row I2_S | 1.211317e+03 | 1.250355 | 38.883200 | 218.172685 | 18.973629 | 0.410567 | 3.166178 | 3.414981 | 1.288133 | 3.035094 | 1.184880 | - |
| row I2_SR | 1.211317e+03 | 1.250175 | 38.847700 | 211.668328 | 19.065496 | 0.410567 | 3.163287 | 3.431516 | 1.288133 | 3.032323 | 1.190617 | - |


