python ./original_code/evaluate_all_samples_jp.py \
  --features_dir data_argo/features/ \
  --adv_cfg_path config.simpl_cfg \
  --model_path saved_models/simpl_av1_ckpt.tar \
  --mode val \
  --output_csv output/argo/argo_val_eval_metrics_loss_0903_v1.0.csv