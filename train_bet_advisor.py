import math
import random
from collections import deque
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from jinja2.runtime import Macro
from torch.distributions import Normal
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import joblib
from SalePredictor import SalePredictor
import SAC

def calculate_betting_cost(selected_games, action, multiple_factor):
    """
    Calculate the total betting cost based on selected games, action values, and multiple factor.
    """
    cost = 2
    for game_index in selected_games:
        a = action[game_index]
        if a < -0.3:
            cost = cost  # Only bet on one outcome
        elif a > 0.3:
            cost = cost * 3  # Bet on all three outcomes
        else:
            cost = cost * 2  # Bet on two outcomes
    cost *= multiple_factor
    return cost


def calculate_reward(state, issue_result, action):
    prize = issue_result[1]
    issue_result = issue_result[0]
    # 去除issue_result中的空格
    # 转换倍数
    if action[14] < 0:
        multiple_factor = 1
    else:
        multiple_factor = int(np.ceil(action[14] * 49 + 1))
    multiple_factor = 1
    # 选择action值最小的9场比赛进行投注
    selected_games = np.argsort(action[:14])[:9]

    # 存储投注的方案
    betting_options = []

    for game_index in range(14):
        odds = state[game_index * 9 + 6: game_index * 9 + 9]  # 获取每场比赛的后三个赔率，即胜平负的赔率
        game_action = action[game_index]
        
        if game_action < -0.3:
            min_odd_index = np.argmin(odds)
            betting_options.append(["3", "1", "0"][min_odd_index])
        elif game_action > 0.3:
            betting_options.append("*")  # 表示全包
        else:
            min_odd_indices = np.argsort(odds)[:2]
            outcomes = ["3", "1", "0"]
            betting_options.append(",".join([outcomes[i] for i in min_odd_indices]))

    # 计算投注的总成本
    total_cost = calculate_betting_cost(selected_games, action, multiple_factor)

    # 检查是否中奖
    win = True
    for game_index in selected_games:
        game_result = issue_result[game_index]
        if game_result == "*":
            continue  # 比赛延期，任何投注都算赢
        if betting_options[game_index] != "*" and game_result not in betting_options[game_index].split(","):
            win = False
            break

    # 如果中奖，则返回奖金，否则返回成本为负数（表示亏损）
    if win:
        if prize>10000:
            prize=0.8*prize
        reward = prize * multiple_factor - total_cost
    else:
        reward = -total_cost

        # 标准化 reward 到 [-40000, 40000] 的范围

    # reward=math.atan(reward)
    # scaled_reward = np.tanh((reward+40000) / 50000.0)
    scaled_reward = reward / 50000
    return scaled_reward,reward


def train_sac(agent, replay_buffer, batch_size, gamma=0.0, tau=0.005):
    """
    Train the SAC agent using dynamically calculated rewards.
    """
    if len(replay_buffer) < batch_size:
        return  # Wait until buffer is populated enough

    # Sample a batch of data
    states, features_2_results = replay_buffer.sample(batch_size)
    features_2 = states[:, :14 * 9]
    sale_predictions = states[:, 14 * 9:]

    # Generate actions and log probabilities from the Actor
    actions, log_probs = agent.actor.sample_action(features_2, sale_predictions)

    # Calculate rewards dynamically
    rewards = torch.tensor(
        [calculate_reward(state, f2_result, action)[0] for state, f2_result, action in
         zip(states.cpu().numpy(), features_2_results, actions.detach().cpu().numpy())],
        dtype=torch.float32
    ).to(states.device)



    # Assume next_states are the same as current states in offline learning
    next_states = states.clone()

    # Compute target Q values
    with torch.no_grad():
        next_actions, next_log_probs = agent.actor.sample_action(next_states[:, :14 * 9], next_states[:, 14 * 9:])
        next_q1 = agent.target_critic1(next_states[:, :14 * 9], next_states[:, 14 * 9:], next_actions)
        next_q2 = agent.target_critic2(next_states[:, :14 * 9], next_states[:, 14 * 9:], next_actions)
        next_q = torch.min(next_q1, next_q2) - agent.alpha * next_log_probs
        # next_q = torch.clamp(next_q, min=-10, max=10)
        target_q = rewards + gamma * next_q
        # target_q = torch.clamp(target_q, min=-10000, max=10000)



    # Update Critic 1
    q1 = agent.critic1(features_2, sale_predictions, actions)
    loss_fn = nn.SmoothL1Loss()  # Use Huber Loss for better stability
    critic1_loss = loss_fn(q1, target_q)
    agent.critic1_optimizer.zero_grad()
    critic1_loss.backward(retain_graph=True)
    # Gradient clipping for Critic 1
    # torch.nn.utils.clip_grad_norm_(agent.critic1.parameters(), max_norm=50.0)
    agent.critic1_optimizer.step()

    # Update Critic 2
    q2 = agent.critic2(features_2, sale_predictions, actions)
    critic2_loss = loss_fn(q2, target_q)
    agent.critic2_optimizer.zero_grad()
    critic2_loss.backward(retain_graph=True)
    # Gradient clipping for Critic 2
    # torch.nn.utils.clip_grad_norm_(agent.critic2.parameters(), max_norm=50.0)
    agent.critic2_optimizer.step()

    # Update Actor
    # actions_pred, log_probs_pred = agent.actor.sample_action(features_2, sale_predictions)
    q1_pred = agent.critic1(features_2, sale_predictions, actions)
    q2_pred = agent.critic2(features_2, sale_predictions, actions)
    q_pred = torch.min(q1_pred, q2_pred)
    actor_loss = (agent.alpha * log_probs - q_pred).mean()
    agent.actor_optimizer.zero_grad()
    actor_loss.backward()
    # Gradient clipping for Actor
    # torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), max_norm=50.0)
    agent.actor_optimizer.step()

    # Update alpha (temperature parameter)
    agent.log_alpha.data = torch.clamp(agent.log_alpha.data, min=np.log(0.1), max=np.log(0.5))
    alpha_loss = -(agent.log_alpha * (log_probs + agent.target_entropy).detach()).mean()
    agent.alpha_optimizer.zero_grad()
    alpha_loss.backward()
    agent.alpha_optimizer.step()

    # Update target networks
    agent._update_target_networks(tau)

    return critic1_loss.item(), critic2_loss.item(), actor_loss.item(), alpha_loss.item()




def offline_sac_training(agent, dataset, num_epochs=1000, batch_size=32, gamma=0.0, tau=0.005):
    """
    Train SAC agent using an offline dataset.
    :param agent: SACAgent instance
    :param dataset: Offline dataset containing (state, features_2_result)
    :param num_epochs: Number of training epochs
    :param batch_size: Batch size for training
    :param gamma: Discount factor
    :param tau: Soft update factor for target networks
    :param alpha: Temperature parameter for entropy
    """
    replay_buffer = SAC.ReplayBuffer(buffer_size=len(dataset))
    for data in dataset:
        replay_buffer.add(data.T)

    for epoch in range(num_epochs):
        alpha_losses,critic1_losses, critic2_losses, actor_losses = [], [], [],[]

        # Train with mini-batches
        for _ in range(len(replay_buffer) // batch_size):
            loss = train_sac(agent, replay_buffer, batch_size, gamma, tau)
            if loss is not None:
                c1_loss, c2_loss, a_loss,alpha_loss  = loss
                critic1_losses.append(c1_loss)
                critic2_losses.append(c2_loss)
                actor_losses.append(a_loss)
                alpha_losses.append(alpha_loss)

        # Evaluate model on entire dataset
        total_reward = 0.0
        for state, issue_result in replay_buffer.buffer:
            # Prepare input
            features_2 = torch.tensor(state[:14 * 9], dtype=torch.float32).unsqueeze(0)  # Add batch dim
            sale_prediction = torch.tensor(state[14 * 9:], dtype=torch.float32).unsqueeze(0)

            # Get deterministic action (mean of the action distribution)
            with torch.no_grad():
                action_mean, _ = agent.actor(features_2, sale_prediction)
                action = action_mean.squeeze(0).numpy()  # Remove batch dim

            # Calculate reward for the action
            _,reward = calculate_reward(state, issue_result, action)
            total_reward += reward

        # Logging
        if epoch%5==0:
            print(f"Epoch {epoch + 1}/{num_epochs}, "
                  f"Critic1 Loss: {np.mean(critic1_losses):.4f}, "
                  f"Critic2 Loss: {np.mean(critic2_losses):.4f}, "
                  f"Actor Loss: {np.mean(actor_losses):.4f}, "
                  f"Alpha: {agent.alpha.item():.4f}, "
                  f"Alpha Loss: {np.mean(alpha_losses):.4f}",
                  f"Total Reward: {total_reward:.2f}"
                  )








def load_data(odds_file="odds_euromean.csv", history_file="history_data.csv", start=16097, end=24178):
    # Load odds and history data from CSV files
    odds_df = pd.read_csv(odds_file)
    history_df = pd.read_csv(history_file)
    history_df = history_df[history_df["issue_id"] >= start]
    history_df = history_df[history_df["issue_id"] < end]
    return odds_df, history_df


def feature_engineering(odds_df, history_df):
    # Merge data based on issue ID (extracted from Match ID)
    odds_df["issue_id"] = odds_df["Match ID"].str.split("_").str[0].astype(int)
    odds_df["match_seq"] = odds_df["Match ID"].str.split("_").str[1].astype(int)

    # Calculate rolling mean and variance for sale amounts
    history_df["mean_30_sale_amount_R9"] = history_df["sale_amount_R9"].rolling(window=30).mean()
    history_df["var_30_sale_amount_R9"] = history_df["sale_amount_R9"].rolling(window=30).var()
    history_df["mean_30_sale_amount_14"] = history_df["sale_amount_14"].rolling(window=30).mean()
    history_df["var_30_sale_amount_14"] = history_df["sale_amount_14"].rolling(window=30).var()
    history_df.bfill(inplace=True)

    # Merge odds and history data
    merged_df = pd.merge(history_df, odds_df, how="inner", on="issue_id")
    merged_df = merged_df.sort_values(["issue_id", "match_seq"]).reset_index(drop=True)

    # Calculate host and guest match rates
    host_guest_counts = merged_df.groupby(["Host ID", "Guest ID"]).size().reset_index(name='count')
    merged_df = pd.merge(merged_df, host_guest_counts, left_on=["Host ID", "Guest ID"],
                         right_on=["Host ID", "Guest ID"], how='left')
    merged_df["rate_host"] = merged_df["count"]
    merged_df["rate_guest"] = merged_df["count"]

    # Calculate odds ranges
    merged_df["range3"] = merged_df["Max3"] - merged_df["Min3"]
    merged_df["range1"] = merged_df["Max1"] - merged_df["Min1"]
    merged_df["range0"] = merged_df["Max0"] - merged_df["Min0"]

    # Extract features needed for the model
    match_features = ["rate_host", "rate_guest", "Odds_48H_of_3", "Odds_48H_of_1", "Odds_48H_of_0", "range3", "range1",
                      "range0", "Odds_24H_of_3", "Odds_24H_of_1", "Odds_24H_of_0"]
    issue_df = merged_df[
        ['issue_id', 'results', "sale_amount_R9", "sale_amount_14", "prize_R9", "prize_14_1", "prize_14_2",
         "mean_30_sale_amount_R9", "var_30_sale_amount_R9", "mean_30_sale_amount_14",
         "var_30_sale_amount_14"]].drop_duplicates().reset_index(drop=True)
    # Create match-specific features for each of the 14 matches
    for i in range(1, 15):
        for feature in match_features:
            new_column = merged_df.loc[merged_df["match_seq"] == i, feature].values
            if new_column.shape[0] != issue_df.shape[0]:
                print("error!!!!")
            issue_df[f"match_{i}_{feature}"] = new_column
    issue_features_1 = ["mean_30_sale_amount_R9", "var_30_sale_amount_R9", "mean_30_sale_amount_14",
                        "var_30_sale_amount_14"]
    issue_features_2 = ["results", "prize_R9"]

    for i in range(1, 15):
        for feature in ["rate_host", "rate_guest", "Odds_48H_of_3", "Odds_48H_of_1", "Odds_48H_of_0", "range3",
                        "range1",
                        "range0"]:
            issue_features_1.append(f"match_{i}_{feature}")
        for feature in ["Odds_48H_of_3", "Odds_48H_of_1", "Odds_48H_of_0", "range3", "range1", "range0",
                        "Odds_24H_of_3", "Odds_24H_of_1", "Odds_24H_of_0"]:
            issue_features_2.append(f"match_{i}_{feature}")

    return issue_df[issue_features_1], issue_df[issue_features_2]




def predict_sales(features_1, model_path, scaler_path):
    # Load model
    input_dim = 14 * 8 + 4  # Example: match features + extra features
    output_dim = 2
    model = SalePredictor(input_dim=input_dim, output_dim=output_dim)
    model.load_state_dict(torch.load(model_path))

    # Set the model to evaluation mode
    device = torch.device('cpu')
    model.to(device)
    model.eval()

    # Load the saved scaler to normalize input data in the same way as during training
    sale_predictor_scaler_x = joblib.load(scaler_path)

    # Scale the features_1
    features_1_scaled = sale_predictor_scaler_x.transform(features_1)

    # Convert the input data to a torch tensor
    input_tensor = torch.FloatTensor(features_1_scaled).to(device)

    # Perform the prediction
    with torch.no_grad():
        sale_predictions = model(input_tensor)
    assert sale_predictions.shape[1] == 2, "输出维度不匹配"
    return sale_predictions


def main():
    # Load data
    odds_df, history_df = load_data(end=24065)

    # Feature engineering
    features_1, features_2 = feature_engineering(odds_df, history_df)
    # Load the scaler for feature_2
    scaler_2 = StandardScaler()
    issue_results = features_2[["results", "prize_R9"]].values
    features_2 = features_2.drop(columns=['results', 'prize_R9'])
    features_2 = scaler_2.fit_transform(features_2)

    # Predict sales
    model_path = 'sale_predictor2690.pth'
    scaler_path = 'x_scaler.pkl'
    sale_predictions = predict_sales(features_1, model_path, scaler_path)
    sale_predictions = sale_predictions.numpy().astype(np.float32)
    # Prepare the SAC environment state
    offline_dataset = np.hstack((features_2, sale_predictions[:, :1], issue_results))

    # # 初始化 SAC Agent
    agent = SAC.SACAgent(state_dim=127, action_dim=15)
    # # 开始训练
    offline_sac_training(agent, offline_dataset, num_epochs=3000, batch_size=64)



if __name__ == "__main__":
    main()
