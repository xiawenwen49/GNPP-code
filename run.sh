dataset=Wikipedia # Wikipedia, Reddit, CollegeMsg, Synthetic_hawkes_neg, Synthetic_hawkes_pos, Synthetic_poisson
model=GNPP
batch_size=1
epochs=100
gpu=2
num_heads=1
optim=adam
time_encoder_type=he
time_encoder_dimension=128
desc=he
with_neig=1

python -m tge.main --model ${model} --num_heads ${num_heads} --dataset ${dataset} --epochs ${epochs} --batch_size=${batch_size} \
--gpu ${gpu} --optim ${optim} --time_encoder_type ${time_encoder_type} --time_encoder_dimension ${time_encoder_dimension} \
--with_neig ${with_neig} --desc ${desc}


