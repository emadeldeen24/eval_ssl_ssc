exp="Supervised_training"
train_mode="supervised_100per"
device="cuda:1"
data_percentage="100"
sleep_model="attnsleep"
ssl_method="simclr"

python3 main.py \
    --device $device \
    --experiment_description $exp \
    --run_description "AttnSleep" \
    --fold_id 0 \
    --train_mode $train_mode \
    --data_percentage $data_percentage \
    --sleep_model $sleep_model \
    --ssl_method $ssl_method \
    --augmentation "timeShift_permute" \
    --dataset "sleep_edf"