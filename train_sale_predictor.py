import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import joblib

from SalePredictor import SalePredictor

# Step 1: Load and Preprocess Data
odds_file = "odds_euromean.csv"
history_file = "history_data.csv"
odds_df = pd.read_csv(odds_file)
history_df = pd.read_csv(history_file)
history_df = history_df[history_df["issue_id"] < 24065]
# history_df = history_df[history_df["issue_id"] < 17001]

# Step 2: Feature Engineering
def feature_engineering(odds_df, history_df):
    # Merge data based on issue ID (extracted from Match ID)

    odds_df["issue_id"] = odds_df["Match ID"].str.split("_").str[0].astype(int)
    odds_df["match_seq"] = odds_df["Match ID"].str.split("_").str[1].astype(int)
    history_df["mean_30_sale_amount_R9"] = history_df["sale_amount_R9"].rolling(window=30).mean()
    history_df["var_30_sale_amount_R9"] = history_df["sale_amount_R9"].rolling(window=30).var()
    history_df["mean_30_sale_amount_14"] = history_df["sale_amount_14"].rolling(window=30).mean()
    history_df["var_30_sale_amount_14"] = history_df["sale_amount_14"].rolling(window=30).var()
    history_df.bfill(inplace=True)

    merged_df = pd.merge(history_df, odds_df, how="inner", on="issue_id")

    # Recent 30 issues mean and variance calculations
    merged_df = merged_df.sort_values(["issue_id", "match_seq"]).reset_index(drop=True)

    # Fill NA values created by rolling operations

    # Rate calculations for Host and Guest ID
    def calc_host_guest_rate(issue_id, host_or_guest):
        count = merged_df[(merged_df["Host ID"] == host_or_guest) | (merged_df["Guest ID"] == host_or_guest)].shape[0]
        return count

    merged_df["rate_host"] = merged_df.apply(lambda row: calc_host_guest_rate(row["issue_id"], row["Host ID"]),
                                             axis=1)
    merged_df["rate_guest"] = merged_df.apply(lambda row: calc_host_guest_rate(row["issue_id"], row["Guest ID"]),
                                              axis=1)
    merged_df["range3"] = merged_df["Max3"] - merged_df["Min3"]
    merged_df["range1"] = merged_df["Max1"] - merged_df["Min1"]
    merged_df["range0"] = merged_df["Max0"] - merged_df["Min0"]

    # Extract odds features needed for each match
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

    # Final feature set
    issue_features = ["mean_30_sale_amount_R9", "var_30_sale_amount_R9", "mean_30_sale_amount_14",
                      "var_30_sale_amount_14"]
    for i in range(1, 15):
        for feature in match_features:
            issue_features.append(f"match_{i}_{feature}")

    return issue_df[issue_features], issue_df[["sale_amount_R9", "sale_amount_14"]]


X, y = feature_engineering(odds_df, history_df)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

y_scaler = StandardScaler()
y_train = y_scaler.fit_transform(y_train)
y_test = y_scaler.transform(y_test)

# Save the scaler
joblib.dump(scaler, 'x_scaler.pkl')
joblib.dump(y_scaler, 'y_scaler.pkl')

# Step 3: Build the Sale Predictor Model


# Model Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = X_train.shape[1]
output_dim = y_train.shape[1]
model = SalePredictor(input_dim=input_dim, output_dim=output_dim).to(device)
criterion = nn.MSELoss()
shared_optimizer = optim.Adam(model.shared_block.parameters(), lr=0.0001)  # Shared block with lower learning rate
final_optimizer = optim.Adam(model.final_block.parameters(), lr=0.001)  # Final block with higher learning rate

# Learning rate scheduler
shared_scheduler = torch.optim.lr_scheduler.StepLR(shared_optimizer, step_size=4, gamma=0.7)
final_scheduler = torch.optim.lr_scheduler.StepLR(final_optimizer, step_size=4, gamma=0.7)
min_lr_shared = 3e-5  # Minimum learning rate
min_lr_final = 3e-5  # Minimum learning rate

# Create Dataset and DataLoader for Mini-batch Gradient Descent
from torch.utils.data import DataLoader, TensorDataset

train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

X_train_tensor = torch.FloatTensor(X_train).to(device)
y_train_tensor = torch.FloatTensor(y_train).to(device)
X_test_tensor = torch.FloatTensor(X_test).to(device)
y_test_tensor = torch.FloatTensor(y_test).to(device)

num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        shared_optimizer.zero_grad()
        final_optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        shared_optimizer.step()
        final_optimizer.step()
        total_loss += loss.item()  # 累加每个 batch 的损失

    avg_train_loss = total_loss / len(train_loader)  # 计算平均损失
    if shared_optimizer.param_groups[0]['lr'] > min_lr_shared:
        shared_scheduler.step()
    else:
        shared_optimizer.param_groups[0]['lr'] = min_lr_shared

    if final_optimizer.param_groups[0]['lr'] > min_lr_final:
        final_scheduler.step()
    else:
        final_optimizer.param_groups[0]['lr'] = min_lr_final

    if (epoch + 1) % 1 == 0:
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            test_loss = criterion(test_outputs, y_test_tensor)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.8f}, Test Loss: {test_loss.item():.8f}')

        # Save the model
        torch.save(model.state_dict(), 'sale_predictor.pth')

# Step 4: Model Evaluation
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor).cpu().numpy()
    predictions = y_scaler.inverse_transform(predictions)
    mse = mean_squared_error(y_test, predictions)
    print(f'Mean Squared Error on Test Set: {mse:.8f}')
