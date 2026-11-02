_base_ = ['../_base_/d2mamba_ddr_1028.py']

# Fold 4 of the reconstructed image-level five-fold DDR split (see splits/).
train_dataloader = dict(dataset=dict(manifest_file='splits/fold4_train.csv'))
val_dataloader = dict(dataset=dict(manifest_file='splits/fold4_val.csv'))
test_dataloader = dict(dataset=dict(manifest_file='splits/fold4_test.csv'))
work_dir = 'work_dirs/d2mamba_ddr_fold4'
