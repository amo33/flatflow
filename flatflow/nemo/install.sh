#!/usr/bin/env bash
set -e

apt-get update && apt-get install -y libsndfile1 ffmpeg
git clone https://github.com/NVIDIA/NeMo
cp flatflow/nemo/nemo/collections/nlp/modules/common/megatron/attention.py NeMo/nemo/collections/nlp/modules/common/megatron/attention.py
cp flatflow/nemo/nemo/collections/nlp/models/language_modeling/megatron_gpt_sft_model.py NeMo/nemo/collections/nlp/models/language_modeling/megatron_gpt_sft_model.py
cp flatflow/nemo/examples/nlp/language_modeling/tuning/conf/megatron_gpt_finetuning_config.yaml NeMo/examples/nlp/language_modeling/tuning/conf/megatron_gpt_finetuning_config.yaml
cp flatflow/nemo/examples/nlp/language_modeling/tuning/megatron_gpt_finetuning.py NeMo/examples/nlp/language_modeling/tuning/megatron_gpt_finetuning.py
cd NeMo
./reinstall.sh
