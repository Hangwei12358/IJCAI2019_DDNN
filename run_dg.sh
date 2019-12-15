time CUDA_VISIBLE_DEVICES=0 python main_dg.py  --now_model_name ae_spatial_LSTM_CNN --n_lstm_layer 1 --n_lstm_hidden 64 --batch_size 64 --n_epoch 100 --slidwin balancedUp --dataset dg

time CUDA_VISIBLE_DEVICES=0 python main_dg.py  --now_model_name ae_spatial_LSTM_CNN --n_lstm_layer 2 --n_lstm_hidden 128 --batch_size 64 --n_epoch 100 --slidwin balancedUp --dataset dg
