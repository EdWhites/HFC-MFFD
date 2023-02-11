# ---------log_path----------- #
log_path='./train.log'

# ---------output_model_path----------- #
output_model_path='./et_and_at_model'
    
# ---------train epoch------------ #
num_epoch = 30

# --------- usage ------------ #
usage='binary_classification'

# --------- patch size --------- #
patch_size=9

# ---------- batch size --------- #
batch_size=32

# --------- mean and var path --------#
mean_path='./mean.npy'
var_path='./var.npy'

# --------- data path -----------#
train_datasets_path='./train'
test_datasets_path='./test'

# -------- argument for multi-loss -----#
arfa = 0.6 # loss_all=arfa*loss_patch+(1-arfa)*loss_attention

# -------- learning rate -----#
lr_et = 1e-4
lr_mlp = 1e-4 #for patch
lr_at = 1e-6 #for origin image

