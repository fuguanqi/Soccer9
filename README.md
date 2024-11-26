# Soccer9
基于深度学习和SAC强化学习的14选9投注帮手

# Football Lottery Prediction and Betting Advisor

This repository provides a comprehensive solution for predicting sales and generating betting strategies for China's football lottery games: **14 Matches (\u8db3\u5f6914\u573a)** and its derivative **14 Select 9 (14\u90099)**. The project leverages machine learning and reinforcement learning models to analyze historical data and provide actionable insights.

---

## Features
1. **Sales Prediction**: A deep learning model (`sale_predictor`) predicts the sales revenue for both games based on historical and odds data.
2. **Betting Strategy Advisor**: A reinforcement learning model (`bet_advisor`) generates optimized betting strategies for the 14 Select 9 game.

---

## Game Rules

### 14 Matches (\u8db3\u5f6914\u573a)
- **Objective**: Predict the results (Win/Draw/Loss) of 14 selected football matches.
  - **Win (3)**: Home team wins.
  - **Draw (1)**: Match ends in a tie.
  - **Loss (0)**: Away team wins.
- **Betting Options**:
  - **Single Selection**: Choose one outcome per match.
  - **Multiple Selection**: Choose multiple outcomes per match to increase winning chances.
- **Cost**: \uffe52 per ticket.
- **Winning Criteria**:
  - **First Prize**: Correctly predict all 14 matches.
  - **Second Prize**: Correctly predict 13 matches.
- **Prize Pool**: Floating pool based on sales revenue. Unclaimed prizes roll over to the next issue.

### 14 Select 9 (14\u90099)
- **Objective**: Predict the results (Win/Draw/Loss) of any 9 matches out of the 14.
- **Betting Options**:
  - Same as 14 Matches but only for 9 matches.
- **Cost**: \uffe52 per ticket.
- **Winning Criteria**: Correctly predict all 9 selected matches.
- **Prize Pool**: Separate from the 14 Matches pool and only has one prize level.

---

## Project Overview

### Data Description

#### `odds_euromean.csv`
Contains odds-related data for each match:
- **Match ID**: Unique identifier combining issue ID and match number.
- **Odds**: Predicted probabilities for Win (3), Draw (1), Loss (0) at different times before the match (12-48 hours).
- **Statistical Fields**: Mean, variance, max, min, and range for each outcome's odds.

#### `history_data.csv`
Contains historical sales and results data:
- **Issue ID**: Unique identifier for each lottery issue.
- **Results**: Match outcomes (`3`, `1`, `0`, `*` for cancellations).
- **Sales Data**: Sales revenue for both games.
- **Prize Data**: Count and prize values for winners of each category.

---

## Models

### 1. Sales Prediction Model (`sale_predictor`)

#### **Input Features**
1. **Historical Sales Data**:
   - Mean and variance of sales for the last 30 issues for both games.
2. **Team and Odds Data**:
   - Appearance frequency of teams as host/guest.
   - Odds and statistical features from `odds_euromean.csv`.

#### **Outputs**
1. **`sale_amount_R9_predict`**: Predicted sales revenue for 14 Select 9.
2. **`sale_amount_14_predict`**: Predicted sales revenue for 14 Matches.

---

### 2. Betting Advisor Model (`bet_advisor`)

A Soft Actor-Critic (SAC) reinforcement learning model suggests optimal betting strategies based on predicted sales and odds data.

#### **Input Features**
1. **Predicted Sales**: Output of `sale_predictor`.
2. **Odds Data**: Odds and ranges for Win (3), Draw (1), and Loss (0) outcomes.

#### **Outputs**
1. **Betting Actions**:
   - Action values dictate the number of outcomes to bet on for each match:
     - **< -0.3**: Bet only on the lowest-odds outcome.
     - **-0.3 to 0.3**: Bet on the two lowest-odds outcomes.
     - **> 0.3**: Bet on all three outcomes.
2. **Bet Multiples**: Suggested investment multiplier.

#### **Reward Function**
- Net profit calculated as winnings minus cost for each issue.

---

## Workflow

1. **Data Preparation**:
   - Preprocess `odds_euromean.csv` and `history_data.csv`.
   - Extract statistical features for model inputs.

2. **Train `sale_predictor`**:
   - Use deep learning to predict sales for both games.

3. **Train `bet_advisor`**:
   - Use SAC reinforcement learning to optimize betting strategies.

4. **Generate Betting Suggestions**:
   - Combine model predictions and odds data to recommend bets.

---

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/football-lottery-predictor.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Train the models:
   ```bash
   python train_sale_predictor.py
   python train_bet_advisor.py
   ```
4. Generate predictions and betting advice:
   ```bash
   python generate_predictions.py
   python generate_betting_advice.py
   ```

---

## Contributions
Feel free to contribute by submitting issues or pull requests to enhance the project.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
