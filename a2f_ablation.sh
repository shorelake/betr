python -m torch.distributed.run --nproc_per_node=2 --master_port=29541 \
  -m tools.train_betr_ablation --experiment baseline --seed 42
python -m torch.distributed.run --nproc_per_node=2 --master_port=29541 \
  -m tools.train_betr_ablation --experiment a2f --seed 42

python -m torch.distributed.run --nproc_per_node=2 --master_port=29541 \
  -m tools.train_betr_ablation --experiment gt --seed 42

python -m torch.distributed.run --nproc_per_node=2 --master_port=29541 \
  -m tools.train_betr_ablation --experiment gt-defcn --seed 42

python -m torch.distributed.run --nproc_per_node=2 --master_port=29541 \
  -m tools.train_betr_ablation --experiment legacy-o2m --seed 42

