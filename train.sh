python main.py \
--data_dir YOUR_TRAIN_DATA_PATH \
--data_list "data_train.list"  \
--save ours \
--epochs 100 \
--batch_size 18 \
--lr 0.0001 \
--decay 10,20,30,40,50 \
--gamma 1.0,0.5,0.25,0.125,0.0625 \
--max_depth 10.0 \
--cut_mask \
--rgb_noise 0.05 \
--noise 0.01 \
--num_threads 16 \

