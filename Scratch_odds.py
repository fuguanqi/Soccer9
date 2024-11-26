import requests
from bs4 import BeautifulSoup
import csv
import time
import random
import datetime
import numpy as np
import pandas as pd
import sys


def fetch_with_retries(url, headers, data={},max_attempts=5 ):
    attempts = 0
    while attempts < max_attempts:
        try:
            if data=={}:
                response = requests.get(url, headers=headers)
            else:
                response = requests.post(url, data=data, headers=headers)
            if response.status_code == 200:
                return response
            elif response.status_code == 429:
                print("Too many requests. Pausing to avoid ban...")
                time.sleep(random.uniform(10, 20))
            else:
                print(
                    f"Request failed with status code: {response.status_code}, retrying... ({attempts + 1}/{max_attempts})")
                if response.status_code==418:
                    sys.exit(418)
                time.sleep(random.uniform(2, 4))
        except requests.exceptions.RequestException as e:
            print(f"Request error occurred: {e}. Retrying... ({attempts + 1}/{max_attempts})")
            time.sleep(random.uniform(2, 4))
        attempts += 1
    return None


def fetch_issue_data(url,issue):
    headers = {
        'User-Agent': random.choice([
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1 Safari/605.1.15',
            'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.182 Safari/537.36',
            'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:88.0) Gecko/20100101 Firefox/88.0',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.85 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        ]),
        'Referer': 'https://cp.zgzcw.com/',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive'
    }

    response = fetch_with_retries(url, headers)
    if response is None or response.status_code != 200:
        print(f"Failed to retrieve issue data after maximum attempts.")
        return []

    response.encoding = 'utf-8'
    try:
        json_data = response.json()
    except requests.exceptions.JSONDecodeError as e:
        print(f"Failed to decode JSON response: {e}.")
        return []

    matches = json_data.get('matchInfo', [])
    issue_data = []

    # 查找有缺失数据的比赛
    missing_indices = []
    for idx, match in enumerate(matches):
        play_id = match.get('playId').strip().replace('-', '')
        host_id = match.get('hostId').strip().replace('-', '')
        guest_id = match.get('guestId').strip().replace('-', '')
        if play_id == '0' or not host_id or not guest_id:
            missing_indices.append(idx)

    # 如果有缺失数据，获取补充信息的URL
    if missing_indices:
        indices = list(range(0, 14))
        good_indices = [x for x in indices if x not in missing_indices]

        play_id = matches[good_indices[0]].get('playId').strip().replace('-', '')

        post_data={"lc":'20'+str(issue)}
        post_url=f"https://fenxi.zgzcw.com/dynamic/{play_id}/play/zucai/bsls"
        response_b = fetch_with_retries(post_url, data=post_data,headers=headers)
        if response_b and response_b.status_code == 200:

            soup_b = BeautifulSoup(response_b.text, 'html.parser')
            lines = soup_b.find_all('li')
            for idx in missing_indices:
                if idx < len(lines):
                    relative_url = lines[idx].find('a').get('href')
                    if relative_url:
                        play_id = relative_url.split('/')[1]
                        full_url = f"https://fenxi.zgzcw.com{relative_url}"
                        response_c = fetch_with_retries(full_url, headers)
                        if response_c and response_c.status_code == 200:
                            soup_c = BeautifulSoup(response_c.text, 'html.parser')
                            host_name_div = soup_c.find('div', class_='host-name')
                            guest_name_div = soup_c.find('div', class_='visit-name')
                            if host_name_div and guest_name_div:
                                host_id = host_name_div.find('a').get('href').split('/')[-1]
                                guest_id = guest_name_div.find('a').get('href').split('/')[-1]
                                matches[idx]['playId'] = play_id
                                matches[idx]['hostId'] = host_id
                                matches[idx]['guestId'] = guest_id


    # 提取比赛数据
    for match in matches:
        play_id = match.get('playId').strip().replace('-', '')
        host_id = match.get('hostId').strip().replace('-', '')
        guest_id = match.get('guestId').strip().replace('-', '')
        lottery_end_date_str = match.get('lotteryEndDate')
        if play_id and lottery_end_date_str:
            # 将日期字符串转换为时间戳
            lottery_end_date = int(
                time.mktime(datetime.datetime.strptime(lottery_end_date_str, "%Y-%m-%d %H:%M:%S").timetuple()))
            issue_data.append(
                {'playId': play_id, 'lotteryEndDate': lottery_end_date, 'hostId': host_id, 'guestId': guest_id})

    return issue_data


def fetch_odds_data(play_id, target_times):
    if play_id == '0':
        return None

    odds_url = f"https://fenxi.zgzcw.com/{play_id}/bjop/zhishu?company_id=0&company=%E5%B9%B3%E5%9D%87%E6%AC%A7%E8%B5%94"

    max_attempts = 5
    attempts = 0
    table = None

    while attempts < max_attempts and table is None:
        headers = {
            'User-Agent': random.choice([
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36',
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1 Safari/605.1.15',
                'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.182 Safari/537.36',
                'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:88.0) Gecko/20100101 Firefox/88.0',
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.85 Safari/537.36',
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            ])
        }

        try:
            response = requests.get(odds_url, headers=headers)

            if response.status_code == 429:
                print("Too many requests. Pausing to avoid ban...")
                time.sleep(random.uniform(10, 20))
                continue

            response.encoding = 'utf-8'
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                table = soup.find('table', class_='dxzkt-tab')
                if table is None:
                    print(f"No odds table found for playId {play_id}, retrying... ({attempts + 1}/{max_attempts})")
                    time.sleep(random.uniform(2, 4))  # Pause before retrying to avoid spamming the server
            else:
                print(
                    f"Request failed with status code: {response.status_code}, retrying... ({attempts + 1}/{max_attempts})")
                if response.status_code==418:
                    sys.exit(218)
                time.sleep(random.uniform(2, 4))
        except requests.RequestException as e:
            print(f"Request error: {e}, retrying... ({attempts + 1}/{max_attempts})")
            time.sleep(random.uniform(2, 4))

        attempts += 1

    if table is None:
        print(f"Failed to retrieve odds table for playId {play_id} after {max_attempts} attempts.")
        return None

    rows = table.find_all('tr')[2:]  # Skip header rows
    odds_data = []
    for row in rows:
        cols = row.find_all('td')
        timestamp_str = cols[1].text.strip()
        timestamp = int(time.mktime(datetime.datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S").timetuple()))
        odds = [float(cols[i].text.strip().replace('↑', '').replace('↓', '')) for i in range(3, 6)]
        odds_data.append({'timestamp': timestamp, 'odds': odds})
    if not odds_data:
        return None
    results = []
    for target_time in target_times:
        closest_record = min(odds_data, key=lambda x: abs(x['timestamp'] - target_time))
        results.append(closest_record['odds'])

    odds_values = [record['odds'] for record in odds_data]

    results.append(np.mean(odds_values, axis=0).tolist())
    results.append(np.var(odds_values, axis=0).tolist())
    results.append(np.max(odds_values, axis=0).tolist())
    results.append(np.min(odds_values, axis=0).tolist())


    print(f"Odds scraped for playId {play_id}.")
    return results


def save_to_csv(data, filename, header_written=True):
    with open(filename, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not header_written:
            writer.writerow(
                ['Match ID', 'Host ID', 'Guest ID', 'Odds_12H_of_3', 'Odds_12H_of_1', 'Odds_12H_of_0', 'Odds_18H_of_3',
                 'Odds_18H_of_1', 'Odds_18H_of_0', 'Odds_24H_of_3', 'Odds_24H_of_1', 'Odds_24H_of_0', 'Odds_30H_of_3',
                 'Odds_30H_of_1', 'Odds_30H_of_0', 'Odds_36H_of_3', 'Odds_36H_of_1', 'Odds_36H_of_0', 'Odds_42H_of_3',
                 'Odds_42H_of_1', 'Odds_42H_of_0', 'Odds_48H_of_3', 'Odds_48H_of_1', 'Odds_48H_of_0', 'Mean3', 'Mean1',
                 'Mean0', 'Variance3', 'Variance1', 'Variance0', 'Max3', 'Max1', 'Max0', 'Min3', 'Min1', 'Min0', ])
        writer.writerows(data)


def main():
    while True:

        # Step 1: 读取csv文件，提取第一列数据
        csv_file_path = 'odds_euromean.csv'  # 请将这里替换为你的CSV文件的实际路径
        df = pd.read_csv(csv_file_path)

        # 假设第一列名为 'col1'，可以通过 df.columns 来查看列名
        first_column = df.iloc[:, 0]  # 取出第一列数据

        # Step 2: 将第一列按下划线分割为 issueID 和 matchID
        df[['issueID', 'matchID']] = first_column.str.split('_', expand=True)

        # Step 3: 找到在 issues 列表中但不在 issueID 中的数据
        issues=list(range(16097, 16199))+list(range(17001, 17196))+list(range(18001, 18177))+list(range(19001, 19182))+list(range(20001, 20084))+list(range(21001, 21160))+list(range(22001, 22153))+list(range(23001, 23175))+list(range(24001, 24178))
        bad_issues=[17111,18094,20008,22014,23139,24093,24173]
        issues = [iss for iss in issues if iss not in bad_issues]
        scrathed_issueIDs = df['issueID'].tolist()

        # 找到在 issues 中但不在 issueID 中的项目
        unscratched_issues = [issue for issue in issues if str(issue) not in scrathed_issueIDs]
        if not unscratched_issues:
            break
        for issue in unscratched_issues:
            issue_url = f"https://cp.zgzcw.com/lottery/zcplayvs.action?lotteryId=13&issue={issue}"
            issue_data = fetch_issue_data(issue_url,issue)
            all_data = []
            print(f"scrathing matches' odds  for issue {issue}.")
            for idx, match in enumerate(issue_data):
                play_id = match['playId']
                host_id = match['hostId']
                guest_id = match['guestId']
                lottery_end_date = match['lotteryEndDate']

                target_times = [
                    lottery_end_date - i * 3600 for i in [12, 18, 24, 30, 36, 42, 48]
                ]

                odds_data = fetch_odds_data(play_id, target_times)
                if odds_data:  # 检查数据是否成功抓取
                    flattened_odds = [value for odds in odds_data for value in odds]
                    all_data.append([f"{issue}_{idx + 1}"] + [host_id, guest_id] + flattened_odds)
                else:
                    print(f"Skipping playId {play_id} due to missing odds data.")
                    all_data = []
                    break

                time.sleep(random.uniform(3, 4))  # 增加每次请求之间的暂停，防止被反爬虫检测
            if issue == 16097:
                save_to_csv(all_data, 'odds_euromean.csv', header_written=False)
            else:
                save_to_csv(all_data, 'odds_euromean.csv', header_written=True)
            # time.sleep(random.uniform(1, 3))  # 增加每次请求之间的暂停，防止被反爬虫检测

            print("Data has been successfully saved to odds_mean_euro.csv!")



if __name__ == "__main__":
    main()
