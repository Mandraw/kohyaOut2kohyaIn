import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--input', help='Input JSON file name', required=True)
parser.add_argument('-o', '--output', help='Output JSON file name', required=True)
args = parser.parse_args()

with open(args.input, 'r') as f:
    input_data = json.load(f)

output_data = {
    "pretrained_model_name_or_path": input_data.get('ss_sd_model_name', None),
    "v2": input_data.get('ss_v2', None),
    "v_parameterization": None,
    "logging_dir": None,
    "train_data_dir": None,
    "reg_data_dir": input_data.get('ss_reg_dataset_dirs', None),
    "output_dir": input_data.get('ss_output_name', None),
    "max_resolution": input_data.get('max_resolution', None),
    "learning_rate": input_data.get('ss_learning_rate', None),
    "lr_scheduler": input_data.get('ss_lr_scheduler', None),
    "lr_warmup": input_data.get('ss_lr_warmup_steps', None),
    "train_batch_size": input_data.get('ss_batch_size_per_device', None),
    "epoch": input_data.get('ss_epoch', None),
    "save_every_n_epochs": None,
    "mixed_precision": input_data.get('ss_mixed_precision', None),
    "save_precision": None,
    "seed": input_data.get('ss_seed', None),
    "num_cpu_threads_per_process": None,
    "cache_latents": input_data.get('ss_cache_latents', None),
    "caption_extension": None,
    "enable_bucket": input_data.get('ss_enable_bucket', None),
    "gradient_checkpointing": input_data.get('ss_gradient_checkpointing', None),
    "full_fp16": input_data.get('ss_full_fp16', None),
    "no_token_padding": None,
    "stop_text_encoder_training": None,
    "use_8bit_adam": input_data.get('ss_optimizer') == 'bitsandbytes.optim.adamw.AdamW8bit',
    "xformers": None,
    "save_model_as": input_data.get('ss_sd_model_hash', None),
    "shuffle_caption": input_data.get('ss_shuffle_caption', None),
    "save_state": None,
    "resume": None,
    "prior_loss_weight": input_data.get('ss_prior_loss_weight', None),
    "text_encoder_lr": input_data.get('ss_text_encoder_lr', None),
    "unet_lr": input_data.get('ss_unet_lr', None),
    "network_dim": input_data.get('ss_network_dim', None),
    "lora_network_weights": None,
    "color_aug": input_data.get('ss_color_aug', None),
    "flip_aug": input_data.get('ss_flip_aug', None),
    "clip_skip": input_data.get('ss_clip_skip', None),
    "gradient_accumulation_steps": input_data.get('ss_gradient_accumulation_steps', None),
    "mem_eff_attn": None,
    "output_name": input_data.get('ss_output_name', None),
    "model_list": None,
    "max_token_length": input_data.get('ss_max_token_length', None),
    "max_train_epochs": None,
    "max_data_loader_n_workers": None,
    "network_alpha": input_data.get('ss_network_alpha', None),
    "training_comment": input_data.get('ss_training_comment', None),
    "keep_tokens": input_data.get('ss_keep_tokens', None),
    "lr_scheduler_num_cycles": input_data.get('ss_lr_scheduler_num_cycles', None),
    "lr_scheduler_power": input_data.get('ss_lr_scheduler_power', None),
    "persistent_data_loader_workers": None,
    "bucket_no_upscale": input_data.get('ss_bucket_no_upscale', None),
    "random_crop": input_data.get('ss_random_crop', None),
    "bucket_reso_steps": None,
    "caption_dropout_every_n_epochs": input_data.get('ss_caption_dropout_every_n_epochs', None),
    "caption_dropout_rate": input_data.get('ss_caption_dropout_rate', None),
    "optimizer": input_data.get('ss_optimizer', None),
    "optimizer_args": None,
    "noise_offset": input_data.get('ss_noise_offset', None),
    "LoRA_type": "LoCon",
    "conv_dim": None,
    "conv_alpha": None
}
with open(args.output, 'w') as f:
    json.dump(output_data, f, indent=4)
print(f'Output saved to {args.output}')
