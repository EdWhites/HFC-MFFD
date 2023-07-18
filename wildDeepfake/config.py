# ---------log_path----------- #
log_path='./train.log'

# ---------sift_log_path----------- #
log_path_sift='./train_sift.log'

# ---------output_model_path----------- #
output_model_path='./et_and_at_model'
output_sift_model_path='./sift_model'
    
# ---------train epoch------------ #
num_epoch = 30

# --------- usage ------------ #
usage='binary_classification'

# --------- image size --------- #
image_size=224

# --------- patch size --------- #
patch_size=128

# ---------- batch size --------- #
batch_size=32
batch_size_sift=128

# --------- mean and var path --------#
mean_path='./mean.npy'
var_path='./var.npy'

# --------- data path -----------#
train_datasets_path='./datasets/train'
test_datasets_path='./datasets/test'

# -------- argument for multi-loss -----#
arfa = 0.6 # loss_all=arfa*loss_patch+(1-arfa)*loss_attention

# -------- learning rate -----#
lr_et = 1e-4
lr_mlp = 1e-4 # for patch
lr_at = 1e-5 # for attention_model
lr_sift=1e-4 # for sift attention_model

# ------- sift predictor path ------- #
predictor_path="./shape_predictor_68_face_landmarks.dat"
