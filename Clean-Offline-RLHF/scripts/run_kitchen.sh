#!/bin/bash
# export WANDB_API_KEY='****'
export WANDB_MODE='online'

################### CS-MLP ######################

# CS-MLP-Kitchen-complete-v0
for ((seed=0; seed<3; seed+=1))
do
    name=CS-MLP-IQL-Kitchen-complete-v0
    mkdir -p ./logs/$name
    device="cuda:1"
    env_type=kitchen
    dataset=complete_v0
    reward_model_path="./rlhf/reward_model_logs/kitchen-complete-v0/CS-MLP/epoch_50_query_2000_len_200_seed_999/models/reward_model.pt"
    reward_model_type=mlp
    config_path=./configs/offline/iql/$env_type/$dataset.yaml
    nohup python -u algorithms/offline/iql_p.py --device $device --seed $seed \
    --reward_model_path $reward_model_path --config_path $config_path \
    --reward_model_type $reward_model_type --seed $seed --name $name \
    >./logs/$name/$seed.log 2>&1 &

    echo "$name $seed training start!"
done

# CS-MLP-Hopper-Kitchen-partial-v0
for ((seed=0; seed<3; seed+=1))
do
    name=CS-MLP-IQL-Kitchen-partial-v0
    mkdir -p ./logs/$name
    device="cuda:2"
    env_type=kitchen
    dataset=partial_v0
    reward_model_path="./rlhf/reward_model_logs/kitchen-partial-v0/CS-MLP/epoch_50_query_2000_len_200_seed_999/models/reward_model.pt"
    reward_model_type=mlp
    config_path=./configs/offline/iql/$env_type/$dataset.yaml
    nohup python -u algorithms/offline/iql_p.py --device $device --seed $seed \
    --reward_model_path $reward_model_path --config_path $config_path \
    --reward_model_type $reward_model_type --seed $seed --name $name \
    >./logs/$name/$seed.log 2>&1 &

    echo "$name $seed training start!"
done

# CS-MLP-Kitchen-mixed-v0
for ((seed=0; seed<3; seed+=1))
do
    name=CS-MLP-IQL-Kitchen-mixed-v0
    mkdir -p ./logs/$name
    device="cuda:0"
    env_type=kitchen
    dataset=mixed_v0
    reward_model_path="./rlhf/reward_model_logs/kitchen-mixed-v0/CS-MLP/epoch_50_query_2000_len_200_seed_999/models/reward_model.pt"
    reward_model_type=mlp
    config_path=./configs/offline/iql/$env_type/$dataset.yaml
    nohup python -u algorithms/offline/iql_p.py --device $device --seed $seed \
    --reward_model_path $reward_model_path --config_path $config_path \
    --reward_model_type $reward_model_type --seed $seed --name $name \
    >./logs/$name/$seed.log 2>&1 &

    echo "$name $seed training start!"
done

