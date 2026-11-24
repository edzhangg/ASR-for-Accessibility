# Qwen 1,000-step validation results

Complete

All models use the pilot's frozen 1,000 recordings. Times are minutes.
The 100-step scores come from the earlier pilot. Both later checkpoints come from one fresh training run per configuration.

| Model | Scored | WER % | CER % | Mean speaker WER % |
|---|---:|---:|---:|---:|
| zero_shot | 1000 | 8.7291 | 5.1644 | 12.5791 |
| E4_L1_100 | 1000 | 8.3557 | 5.0047 | 11.765 |
| E4_L2_100 | 1000 | 9.5694 | 5.9972 | 32.3963 |
| E3_L1_100 | 1000 | 8.3965 | 5.0505 | 11.5951 |
| E3_L2_100 | 1000 | 8.3499 | 5.0763 | 11.6388 |
| E4_L1_500 | 1000 | 8.2215 | 5.1128 | 12.304 |
| E4_L1_1000 | 1000 | 7.8656 | 4.7698 | 11.9787 |
| E4_L2_500 | 1000 | 7.9122 | 4.9013 | 11.3646 |
| E4_L2_1000 | 1000 | 7.7489 | 4.6782 | 11.7031 |
| E3_L1_500 | 1000 | 7.9764 | 4.8391 | 11.5303 |
| E3_L1_1000 | 1000 | 7.813 | 4.8015 | 11.6295 |
| E3_L2_500 | 1000 | 7.8364 | 4.6735 | 11.4454 |
| E3_L2_1000 | 1000 | 7.5446 | 4.5525 | 10.8515 |

| Model | Category | Scored | WER % | CER % |
|---|---|---:|---:|---:|
| zero_shot | Digital Assistant Commands | 334 | 12.7638 | 6.8276 |
| zero_shot | Novel Sentences | 333 | 7.4673 | 3.8065 |
| zero_shot | Spontaneous Speech Prompts | 333 | 8.4986 | 5.4136 |
| E4_L1_100 | Digital Assistant Commands | 334 | 8.8523 | 5.0197 |
| E4_L1_100 | Novel Sentences | 333 | 7.5149 | 4.0484 |
| E4_L1_100 | Spontaneous Speech Prompts | 333 | 8.5896 | 5.3948 |
| E4_L2_100 | Digital Assistant Commands | 334 | 8.8008 | 5.05 |
| E4_L2_100 | Novel Sentences | 333 | 12.176 | 7.7042 |
| E4_L2_100 | Spontaneous Speech Prompts | 333 | 8.7079 | 5.4717 |
| E3_L1_100 | Digital Assistant Commands | 334 | 9.5728 | 5.656 |
| E3_L1_100 | Novel Sentences | 333 | 7.61 | 4.0666 |
| E3_L1_100 | Spontaneous Speech Prompts | 333 | 8.4895 | 5.3423 |
| E3_L2_100 | Digital Assistant Commands | 334 | 8.8008 | 5.3227 |
| E3_L2_100 | Novel Sentences | 333 | 7.4673 | 3.9069 |
| E3_L2_100 | Spontaneous Speech Prompts | 333 | 8.6078 | 5.5111 |
| E4_L1_500 | Digital Assistant Commands | 334 | 7.5656 | 4.8985 |
| E4_L1_500 | Novel Sentences | 333 | 7.1106 | 3.8613 |
| E4_L1_500 | Spontaneous Speech Prompts | 333 | 8.7625 | 5.6667 |
| E4_L1_1000 | Digital Assistant Commands | 334 | 7.9259 | 4.848 |
| E4_L1_1000 | Novel Sentences | 333 | 6.6587 | 3.6057 |
| E4_L1_1000 | Spontaneous Speech Prompts | 333 | 8.3167 | 5.2336 |
| E4_L2_500 | Digital Assistant Commands | 334 | 6.4848 | 3.9996 |
| E4_L2_500 | Novel Sentences | 333 | 6.849 | 3.7289 |
| E4_L2_500 | Spontaneous Speech Prompts | 333 | 8.5714 | 5.5505 |
| E4_L2_1000 | Digital Assistant Commands | 334 | 6.3304 | 3.8481 |
| E4_L2_1000 | Novel Sentences | 333 | 6.6112 | 3.4185 |
| E4_L2_1000 | Spontaneous Speech Prompts | 333 | 8.4349 | 5.3498 |
| E3_L1_500 | Digital Assistant Commands | 334 | 7.6686 | 4.7066 |
| E3_L1_500 | Novel Sentences | 333 | 6.8728 | 3.8065 |
| E3_L1_500 | Spontaneous Speech Prompts | 333 | 8.4531 | 5.2879 |
| E3_L1_1000 | Digital Assistant Commands | 334 | 6.6392 | 4.1511 |
| E3_L1_1000 | Novel Sentences | 333 | 6.5874 | 3.633 |
| E3_L1_1000 | Spontaneous Speech Prompts | 333 | 8.4895 | 5.4023 |
| E3_L2_500 | Digital Assistant Commands | 334 | 7.6686 | 4.5248 |
| E3_L2_500 | Novel Sentences | 333 | 6.5161 | 3.3455 |
| E3_L2_500 | Spontaneous Speech Prompts | 333 | 8.3712 | 5.2467 |
| E3_L2_1000 | Digital Assistant Commands | 334 | 6.6392 | 4.3329 |
| E3_L2_1000 | Novel Sentences | 333 | 5.874 | 3.0215 |
| E3_L2_1000 | Spontaneous Speech Prompts | 333 | 8.3439 | 5.2223 |

| Stage | Minutes |
|---|---:|
| E4_L1 | 53.38 |
| E4_L1_500 | 13.83 |
| E4_L1_1000 | 13.12 |
| E4_L2 | 75.97 |
| E4_L2_500 | 13.04 |
| E4_L2_1000 | 12.79 |
| E3_L1 | 52.64 |
| E3_L1_500 | 13.23 |
| E3_L1_1000 | 13.31 |
| E3_L2 | 75.72 |
| E3_L2_500 | 13.16 |
| E3_L2_1000 | 13.36 |
| comparison | 0.02 |
