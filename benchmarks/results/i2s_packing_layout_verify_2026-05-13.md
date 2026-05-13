# I2S Packing Layout Verification

Reference GGUF: `models/qwen2.5-1.5b-klonly-row-notie-static-ternary-dense/qwen15b_klonly_row_notie_static_ternary_i2_s_rowscale.gguf`
Candidate GGUF: `models/qwen2.5-1.5b-klonly-row-notie-static-ternary-dense/qwen15b_klonly_row_notie_static_ternary_i2_sr_x86act.gguf`

Passed: `True`
Tensors checked: `5`

This check compares only the packed ternary code payload. It does not require scale bytes to match.

| tensor | passed | qtype ref/cand | code bytes | sha256 prefix | first 16 bytes |
| --- | --- | ---: | ---: | --- | --- |
| blk.0.attn_q.weight | true | 36/40 | 589824 | efe477cf31e2 | `820a4956046984896a25868aa0169084` |
| blk.0.attn_k.weight | true | 36/40 | 98304 | 33de83a9758a | `41aa5a8909158989068140aa95282099` |
| blk.0.ffn_gate.weight | true | 36/40 | 3440640 | d6d0c7ad564c | `689a260a512104981140888a5122a968` |
| blk.0.ffn_down.weight | true | 36/40 | 3440640 | de6f51557c79 | `5a9916a896196a5296aa692414084044` |
| blk.27.ffn_down.weight | true | 36/40 | 3440640 | e7a74528be95 | `a8866a5a044866459690991a025a044a` |
