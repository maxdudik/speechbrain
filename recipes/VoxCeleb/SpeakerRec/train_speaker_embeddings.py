# ################################
# Model: Speaker identification with ECAPA
# Authors: Hwidong Na & Mirco Ravanelli
# ################################

# Basic parameters
seed: 1986
__set_seed: !apply:torch.manual_seed [!ref <seed>]
output_folder: !ref results/ecapa_augment/<seed>
save_folder: !ref <output_folder>/save
train_log: !ref <output_folder>/train_log.txt

# Data for augmentation
NOISE_DATASET_URL: https://www.dropbox.com/scl/fi/a09pj97s5ifan81dqhi4n/noises.zip?rlkey=j8b0n9kdjdr32o1f06t0cw5b7&dl=1
RIR_DATASET_URL: https://www.dropbox.com/scl/fi/linhy77c36mu10965a836/RIRs.zip?rlkey=pg9cu8vrpn2u173vhiqyu743u&dl=1

# Data files
data_folder: ../../../dataset/dataset/vox_dnd  # e.g. /path/to/Voxceleb
data_folder_noise: !ref <data_folder>/noise # The noisy sequences for data augmentation will automatically be downloaded here.
data_folder_rir: !ref <data_folder>/rir # The impulse responses used for data augmentation will automatically be downloaded here.
train_annotation: !ref <save_folder>/train.csv
valid_annotation: !ref <save_folder>/dev.csv
noise_annotation: !ref <save_folder>/noise.csv
rir_annotation: !ref <save_folder>/rir.csv

# Use the following links for the official voxceleb splits:
# VoxCeleb1 (cleaned): https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt
# VoxCeleb1-H (cleaned): https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/list_test_hard2.txt
# VoxCeleb1-E (cleaned): https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/list_test_all2.txt.
# VoxCeleb1-E and VoxCeleb1-H lists are drawn from the VoxCeleb1 training set.
# Therefore you cannot use any files in VoxCeleb1 for training if you are using these lists for testing.
verification_file: https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt

split_ratio: [90, 10]
skip_prep: False
ckpt_interval_minutes: 15 # save checkpoint every N min

# Training parameters
number_of_epochs: 10
batch_size: 16
lr: 0.001
base_lr: 0.00000001
max_lr: !ref <lr>
step_size: 65000
sample_rate: 16000
sentence_len: 3.0 # seconds
shuffle: True
random_chunk: True

# Feature parameters
n_mels: 80
left_frames: 0
right_frames: 0
deltas: False

# Number of speakers
out_n_neurons: 27 #1211 for vox1  # 5994 for vox2, 7205 for vox1+vox2

num_workers: 2
dataloader_options:
    batch_size: !ref <batch_size>
    shuffle: !ref <shuffle>
    num_workers: !ref <num_workers>

# Functions
compute_features: !new:speechbrain.lobes.features.Fbank
    n_mels: !ref <n_mels>
    left_frames: !ref <left_frames>
    right_frames: !ref <right_frames>
    deltas: !ref <deltas>

embedding_model: !new:speechbrain.lobes.models.ECAPA_TDNN.ECAPA_TDNN
    input_size: !ref <n_mels>
    channels: [1024, 1024, 1024, 1024, 3072]
    kernel_sizes: [5, 3, 3, 3, 1]
    dilations: [1, 2, 3, 4, 1]
    groups: [1, 1, 1, 1, 1]
    attention_channels: 128
    lin_neurons: 192

classifier: !new:speechbrain.lobes.models.ECAPA_TDNN.Classifier
    input_size: 192
    out_neurons: !ref <out_n_neurons>

epoch_counter: !new:speechbrain.utils.epoch_loop.EpochCounter
    limit: !ref <number_of_epochs>

############################## Augmentations ###################################

# Download and prepare the dataset of noisy sequences for augmentation
prepare_noise_data: !name:speechbrain.augment.preparation.prepare_dataset_from_URL
    URL: !ref <NOISE_DATASET_URL>
    dest_folder: !ref <data_folder_noise>
    ext: wav
    csv_file: !ref <noise_annotation>


# Add noise to input signal
add_noise: !new:speechbrain.augment.time_domain.AddNoise
    csv_file: !ref <noise_annotation>
    snr_low: 0
    snr_high: 15
    noise_sample_rate: !ref <sample_rate>
    clean_sample_rate: !ref <sample_rate>
    num_workers: !ref <num_workers>

# Download and prepare the dataset of room impulse responses for augmentation
prepare_rir_data: !name:speechbrain.augment.preparation.prepare_dataset_from_URL
    URL: !ref <RIR_DATASET_URL>
    dest_folder: !ref <data_folder_rir>
    ext: wav
    csv_file: !ref <rir_annotation>

# Add reverberation to input signal
add_reverb: !new:speechbrain.augment.time_domain.AddReverb
    csv_file: !ref <rir_annotation>
    reverb_sample_rate: !ref <sample_rate>
    clean_sample_rate: !ref <sample_rate>
    num_workers: !ref <num_workers>

# Frequency drop: randomly drops a number of frequency bands to zero.
drop_freq: !new:speechbrain.augment.time_domain.DropFreq
    drop_freq_low: 0
    drop_freq_high: 1
    drop_freq_count_low: 1
    drop_freq_count_high: 3
    drop_freq_width: 0.05

# Time drop: randomly drops a number of temporal chunks.
drop_chunk: !new:speechbrain.augment.time_domain.DropChunk
    drop_length_low: 1000
    drop_length_high: 2000
    drop_count_low: 1
    drop_count_high: 5

# Augmenter: Combines previously defined augmentations to perform data augmentation
wav_augment: !new:speechbrain.augment.augmenter.Augmenter
    parallel_augment: True
    concat_original: True
    min_augmentations: 4
    max_augmentations: 4
    augment_prob: 1.0
    augmentations: [
        !ref <add_noise>,
        !ref <add_reverb>,
        !ref <drop_freq>,
        !ref <drop_chunk>]

mean_var_norm: !new:speechbrain.processing.features.InputNormalization
    norm_type: sentence
    std_norm: False

modules:
    compute_features: !ref <compute_features>
    embedding_model: !ref <embedding_model>
    classifier: !ref <classifier>
    mean_var_norm: !ref <mean_var_norm>

compute_cost: !new:speechbrain.nnet.losses.LogSoftmaxWrapper
    loss_fn: !new:speechbrain.nnet.losses.AdditiveAngularMargin
        margin: 0.2
        scale: 30

# compute_error: !name:speechbrain.nnet.losses.classification_error

opt_class: !name:torch.optim.Adam
    lr: !ref <lr>
    weight_decay: 0.000002

lr_annealing: !new:speechbrain.nnet.schedulers.CyclicLRScheduler
    base_lr: !ref <base_lr>
    max_lr: !ref <max_lr>
    step_size: !ref <step_size>

# Logging + checkpoints
train_logger: !new:speechbrain.utils.train_logger.FileTrainLogger
    save_file: !ref <train_log>

error_stats: !name:speechbrain.utils.metric_stats.MetricStats
    metric: !name:speechbrain.nnet.losses.classification_error
        reduction: batch

checkpointer: !new:speechbrain.utils.checkpoints.Checkpointer
    checkpoints_dir: !ref <save_folder>
    recoverables:
        embedding_model: !ref <embedding_model>
        classifier: !ref <classifier>
        normalizer: !ref <mean_var_norm>
        counter: !ref <epoch_counter>
        lr_annealing: !ref <lr_annealing>
