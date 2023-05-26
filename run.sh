if [ ! -d "plots" ]; then
    mkdir plots
fi

if [ ! -d "videos" ]; then
    mkdir videos
fi

alpha_values=(0.1 0.3 0.5 0.7 0.9)
echo ${alpha_values[0]}

# First experiment
python3 main.py --algorithm q_learning --alpha ${alpha_values[0]} --gamma 0.995 --epsilon 0.1 --num_episodes 1000 --num_steps 500 --num_bins 100 --seed 2