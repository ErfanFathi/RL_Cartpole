if [ -d "logs" ]; then
    rm -rf logs
fi

if [ -d "plots" ]; then
    rm -rf plots
fi

if [ -d "videos" ]; then
    rm -rf videos
fi

if [ ! -d "plots" ]; then
    mkdir plots
fi

if [ ! -d "videos" ]; then
    mkdir videos
fi

if [ ! -d "logs" ]; then
    mkdir logs
fi

# alpha values
alpha_values=(0.1 0.2 0.3)

# gamma values
gamma_values=(0.995 0.9 0.85)

# epsilon values
epsilon_values=(0.1 0.2 0.3)

# Run the experiments
for alpha in "${alpha_values[@]}"
do
    # for loop for gamma values
    for gamma in "${gamma_values[@]}"
    do
        # for loop for epsilon values
        for epsilon in "${epsilon_values[@]}"
        do
            touch logs/sarsa_${alpha}_${gamma}_${epsilon}.txt
            # run the main.py file - sarsa algorithm
            (python3 main.py --algorithm sarsa --alpha $alpha --gamma $gamma --epsilon $epsilon --num_episodes 70000 --num_steps 500 --num_bins 100 --seed 2)>>logs/sarsa_${alpha}_${gamma}_${epsilon}.txt
            echo "sarsa_${alpha}_${gamma}_${epsilon} done"
            # run the main.py file - q_learning algorithm
            (python3 main.py --algorithm q_learning --alpha $alpha --gamma $gamma --epsilon $epsilon --num_episodes 70000 --num_steps 500 --num_bins 100 --seed 2)>>logs/q_learning_${alpha}_${gamma}_${epsilon}.txt
            echo "q_learning_${alpha}_${gamma}_${epsilon} done"
        done
    done
done
