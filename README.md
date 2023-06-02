# Reinforcement Learning on CartPole-v1
This project implements the Q-learning and SARSA algorithms to solve the CartPole-v1 environment from OpenAI Gym. The Q-learning algorithm learns an optimal action-value function, while the SARSA algorithm learns an action-value function based on the current policy. The goal is to balance a pole on a cart by applying appropriate forces.

## Usage
- 1. Clone the repository or download the source code files.
  ```bash
    git clone git@github.com:ErfanFathi/RL_Project.git
  ```
- 2. Install the required packages.
  ```bash
    pip3 install -r requirements.txt
  ```
- 3. Run the script with the desired parameters. Use the following command to see the available options:
  ```bash
    python3 main.py --help
  ```  
    This script uses command-line arguments to configure the learning parameters and other settings. You can specify the following options:
    
    - `--algorithm`: The algorithm to use for learning. Valid options are `q_learning` and `sarsa`. Default is `q_learning`.
    -  `--alpha`: The learning rate. Default is `0.1`.
    -  `--gamma`: The discount factor. Default is `0.995`.
    -  `--epsilon`: The probability of choosing a random action. Default is `0.1`.
    -  `--num-episodes`: The number of episodes to run. Default is `1000`.
    -  `--num-steps`: The maximum number of steps per episode. Default is `500`.
    -  `--num-bins`: The number of bins to use for discretizing the state space. Default is `20`.
 - e.g.:
   ```bash
    python3 main.py --algorithm q_learning --alpha 0.2 --gamma 0.99 --num_episodes 2000
   ```
  
- 4. The script will execute the chosen algorithm on the CartPole-v1 environment. It will print the name of the generated file containing the results.

- 5. After the execution, a plot of the rewards obtained during the learning process will be saved in the `plots` directory as a PNG file.

- 6. Additionally, frames of the agent's behavior will be rendered and saved as a GIF file in the `videos` directory. This provides a visual representation of the learned policy.

## Result
<p align="center">
<img src = "https://github.com/ErfanFathi/RL_Project/blob/main/plots/q_learning_alpha_0.2_gamma_0.995_epsilon_0.2_num_episodes_10000_num_steps_500_num_bins_20.png" width="480" height="300"</img>
<img src = "https://github.com/ErfanFathi/RL_Project/blob/main/videos/q_learning_alpha_0.2_gamma_0.995_epsilon_0.2_num_episodes_10000_num_steps_500_num_bins_20.gif" width="480" height="300"</img>
</p>

## Finale

Feel free to use, modify this code. And please feel free to fork the code 
from Github and send pull requests.

Report any comment or bugs to:<br />
fathierfan97@gmail.com<br />

Regards,<br />
Erfan Fathi<br />
