###############################################
## Installations after clone the repo##
cd lmms-eval
pip install -e .
## make sure llava is installed ##
git clone https://github.com/LLaVA-VL/LLaVA-NeXT
cd LLaVA-NeXT
pip install -e .
###############################################

# Run llavaone 0.5 for mme
accelerate launch --num_processes=1 \
-m lmms_eval \
--model llava_onevision \
--model_args pretrained=lmms-lab/llava-onevision-qwen2-0.5b-si,conv_template=qwen_1_5,model_name=llava_qwen \
--tasks mme \
--batch_size 1 \
--log_samples \
--log_samples_suffix llava_onevision \
--output_path ./logs/
# Run llavaone 0.5 for gqa
accelerate launch --num_processes=1 \
-m lmms_eval \
--model llava_onevision \
--model_args pretrained=lmms-lab/llava-onevision-qwen2-0.5b-si,conv_template=qwen_1_5,model_name=llava_qwen \
--tasks gqa \
--batch_size 1 \
--log_samples \
--log_samples_suffix llava_onevision \
--output_path ./logs/

# ferret,wildvision
# mmmu,mme,mmbench,seedbench_2_plus,llava_wilder_small,llava_bench_coco,llava_in_the_wild,vibe_eval
accelerate launch --num_processes=1 \
-m lmms_eval \
--model llava_onevision \
--model_args pretrained=lmms-lab/llava-onevision-qwen2-0.5b-si,conv_template=qwen_1_5,model_name=llava_qwen \
--tasks llava_wilder_small \
--batch_size 1 \
--log_samples \
--log_samples_suffix llava_onevision \
--output_path ./logs/

# gqa,vqav2,ok_vqa,realworldqa
accelerate launch --num_processes=1 \
-m lmms_eval \
--verbosity=INFO \
--model vlr \
--model_args pretrained=lmms-lab/llava-onevision-qwen2-0.5b-si,conv_template=qwen_1_5,model_name=llava_qwen \
--tasks mmmu_val \
--batch_size 1 \
--log_samples \
--log_samples_suffix llava_onevision \
--output_path ./logs/

accelerate launch --num_processes=1 \
-m lmms_eval \
--model vlr \
--model_args pretrained=lmms-lab/llava-onevision-qwen2-0.5b-si,conv_template=qwen_1_5,model_name=llava_qwen \
--tasks realworldqa \
--batch_size 1 \
--log_samples \
--log_samples_suffix vlr \
--output_path ./logs/

lm_eval --model vlr \
    --model_args pretrained=lmms-lab/llava-onevision-qwen2-0.5b-si,conv_template=qwen_1_5,model_name=llava_qwen \
    --tasks mmmu_val \
    --device 0 \
    --batch_size 1 \
    --output_path ./ \
    --use_cache ./eval_cache

accelerate launch --num_processes=1 \
-m lmms_eval \
--model vlr \
--model_args pretrained=lmms-lab/llava-onevision-qwen2-0.5b-si,conv_template=qwen_1_5,model_name=llava_qwen \
--tasks realworldqa \
--batch_size 1 \
--log_samples \
--log_samples_suffix vlr \
--output_path ./logs/

accelerate launch --num_processes=1 \
-m lmms_eval \
--model vlr \
--model_args pretrained=lmms-lab/llava-onevision-qwen2-0.5b-si,conv_template=qwen_1_5,model_name=llava_qwen \
--tasks vqav2_val_lite \
--batch_size 1 \
--log_samples \
--log_samples_suffix vlr \
--output_path ./logs/

accelerate launch --num_processes=1 \
-m lmms_eval \
--model vlr \
--model_args pretrained=lmms-lab/llava-onevision-qwen2-0.5b-si,conv_template=qwen_1_5,model_name=llava_qwen \
--tasks vqav2_val \
--batch_size 1 \
--log_samples \
--log_samples_suffix vlr \
--output_path ./logs/

accelerate launch --num_processes=4 \
-m lmms_eval \
--model llava_onevision \
--model_args pretrained=lmms-lab/llava-onevision-qwen2-0.5b-si,conv_template=qwen_1_5,model_name=llava_qwen \
--tasks vqav2_val \
--batch_size 1 \
--log_samples \
--log_samples_suffix llava_onevision \
--output_path ./logs/

accelerate launch \
    --num_processes=1 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="liuhaotian/llava-v1.6-34b,conv_template=mistral_direct" \
    --tasks realworldqa \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_v1.5_mme_mmbenchen \
    --output_path ./logs/

# Evaluating Llama-3-LLaVA-NeXT-8B on multiple datasets
accelerate launch --num_processes=8 \
  -m lmms_eval \
  --model llava \
  --model_args pretrained=lmms-lab/llama3-llava-next-8b,conv_template=llava_llama_3 \
  --tasks ai2d,chartqa,docvqa_val,mme,mmbench_en_dev \
  --batch_size 1 \
  --log_samples \
  --log_samples_suffix llava_next \
  --output_path ./logs/

# Evaluating LLaVA-NeXT-72B on multiple datasets
accelerate launch --num_processes=1 \
  -m lmms_eval \
  --model llava \
  --model_args pretrained=lmms-lab/llava-next-72b,conv_template=qwen_1_5,model_name=llava_qwen,device_map=auto \
  --tasks ai2d,chartqa,docvqa_val,mme,mmbench_en_dev \
  --batch_size 1 \
  --log_samples \
  --log_samples_suffix llava_next \
  --output_path ./logs/

accelerate launch --num_processes=1 \
  -m lmms_eval \
  --model llava \
  --model_args pretrained=lmms-lab/llava-next-qwen-32b,conv_template=qwen_1_5,model_name=llava_qwen,device_map=auto \
  --tasks rqa \
  --batch_size 1 \
  --log_samples \
  --log_samples_suffix llava_next \
  --output_path ./logs/