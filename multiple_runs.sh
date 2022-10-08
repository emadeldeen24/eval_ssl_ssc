exp="SimCLR_baselines"
train_mode="ft_1per_withoutTemporal"
device="cuda:0"
data_percentage="1"
ssl_method="simclr"

start=0
end=4

sleep_model="dsn"
for i in $(eval echo {$start..$end})
do
   python3 main.py \
   --device $device \
   --experiment_description $exp \
   --run_description "dsn" \
   --fold_id $i \
   --train_mode $train_mode \
   --data_percentage $data_percentage \
   --sleep_model $sleep_model \
   --ssl_method $ssl_method \
   --augmentation "noise_permute" \
   --dataset "sleep_edf"
done


sleep_model="attnsleep"
for i in $(eval echo {$start..$end})
do
   python3 main.py \
   --device $device \
   --experiment_description $exp \
   --run_description "AttnSleep" \
   --fold_id $i \
   --train_mode $train_mode \
   --data_percentage $data_percentage \
   --sleep_model $sleep_model \
   --ssl_method $ssl_method \
   --augmentation "timeShift_permute" \
   --dataset "sleep_edf"
done


sleep_model="cnn1d"
for i in $(eval echo {$start..$end})
do
   python3 main.py \
   --device $device \
   --experiment_description $exp \
   --run_description "cnn1d_deeper" \
   --fold_id $i \
   --train_mode $train_mode \
   --data_percentage $data_percentage \
   --sleep_model $sleep_model \
   --ssl_method $ssl_method \
   --augmentation "negate_permute" \
   --dataset "sleep_edf"
done
