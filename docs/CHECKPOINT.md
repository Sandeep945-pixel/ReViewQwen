# Released checkpoint and research provenance

## Verified adapter metadata

The demo uses [domsoos/reviewqwen-large](https://huggingface.co/domsoos/reviewqwen-large), pinned to revision `aaf14042b119db8e90da75b963de3882338e7987`.

| Property | Released adapter |
| --- | --- |
| Base model | `Qwen/Qwen2-VL-7B-Instruct` |
| Adaptation | LoRA / PEFT |
| Rank | 40 |
| Alpha | 64 |
| Dropout | 0.2 |
| Target modules | `q_proj`, `v_proj` |

Source: [adapter_config.json at the pinned revision](https://huggingface.co/domsoos/reviewqwen-large/blob/aaf14042b119db8e90da75b963de3882338e7987/adapter_config.json). The base model and processor are loaded from their public repository without a pinned base revision; the demo is not a fully frozen reproduction environment.

## Distinguishing artifacts

The paper describes rank 8, alpha 16, dropout 0.1, and a 524-pixel image size. The checked-in `qwen/finetune_qwen2.py` instead specifies rank 40, alpha 64, dropout 0.2, and 424 pixels. The released adapter confirms the latter LoRA settings; its configuration does not establish the original image preprocessing or which experiment produced the paper's results.

The inference notebook uses 524-pixel images and contains example placeholders. The new CLI is a separate demo entry point: it uses a brief classification/explanation prompt, explicit local inputs, deterministic decoding, and a pinned adapter revision. These choices are not presented as a recovered final evaluation protocol.

## Existing dataset split status

At repository commit `24c7b1d844c1a978d8fa7fb518be678f3b7c7dd4`, the public folders contain 1,434 training metadata files and 329 testing metadata files. Git blob comparison finds that 122 testing metadata files are byte-identical to files in the training folder. The paper describes a 2,365-example corpus with a 75/5/20 split.

This records the state of the uploaded folders, not which data were used in the published experiment. A definitive reproduction requires the original split manifest and final experiment configuration. The demo update preserves existing data and does not claim to resolve the split discrepancy or reproduce the reported benchmark.

## Metric interpretation

README values are transcribed from Table II. In that table, 51.15% is the base Qwen2-VL **precision**, not its accuracy. The abstract also describes 51.15% to 88.00% as an accuracy comparison. Until the underlying result files and denominators are reconciled, the README uses the explicitly labelled precision/recall/F1 table rather than promoting the ambiguous accuracy headline.

The paper's classification comparisons and its focused human/LLM explanation assessments are different evaluations. Neither proves that the released demo will be correct on every customer example. No new accuracy measurement has been made during this repository update.

## Validation and next checks

Completed: Python syntax checks, fictional-input validation, missing/corrupt-input checks, label parsing, multimodal message ordering, and mocked adapter/model-generation checks. These tests use no real model weights and are not scientific evaluations.

Not completed: downloading/loading the full 7B model, GPU inference, dependency installation on a target GPU machine, training reproduction, and reassessment of the original benchmark. Before reporting new performance, record exact model/dependency versions, input manifests, preprocessing, generation settings, and treatment of failed or unparseable predictions.

Implementation references: [Qwen2-VL in Transformers 4.49](https://huggingface.co/docs/transformers/v4.49.0/model_doc/qwen2_vl) and [PEFT model loading](https://huggingface.co/docs/peft/v0.13.0/en/package_reference/peft_model).
