# Compact Prediction Evidence

This bundle preserves all labels and class predictions for `9,815` aligned examples.
It omits logits, prompts, and private paths. Accuracies and paired tests are exactly reconstructible.

| model | accuracy | source trace SHA-256 |
| --- | ---: | --- |
| fp16 | 0.808151 | `6e708242b8c086f0b2aa9b1c7805fc088b4391de3b8442e6f5b66fa44c8f5590` |
| fixed_historical | 0.738462 | `81151391f29321532326b70d3d258c756d1b6835b00c36072f240eef98f97edf` |
| gamma60_historical | 0.729903 | `d18a994d994e0800c7b48644dd123d4acc82110262323f9d0fac03b9cb0959b3` |
| adaptive_seed1234 | 0.755782 | `bc867707dcac2043c926e960289f65a0abd516edb50bf21b0a88b86d22baffef` |
| adaptive_seed1235 | 0.756903 | `2253844dab85411e205da59287122ab66a38db2fe25b6ec7a88a2ef06106a3de` |
| adaptive_seed1236 | 0.753337 | `38b72394b88a7bd114c97ce48ab4598c578c1596697ee90d5303844258874fa5` |
| fixed60_seed1234 | 0.758635 | `2e9edbbcb8c9b858e4139307aacc73da951dcd799bc07ab191fbfabddcbf9f3e` |
| fixed60_seed1235 | 0.754457 | `ed007e6ffdefd77ab81e358bda3fe5d00e7f4eb5aeb086770c1af5039eb57671` |
| fixed60_seed1236 | 0.758023 | `4e18d9f52fdbe8ba1e68bc0898d9f72d83ab76a116fec5baf2ff936121e1100d` |
