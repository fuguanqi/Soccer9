import requests  # 需要安装 requests 库来发送HTTP请求。可以通过 `pip install requests` 进行安装。
from bs4 import BeautifulSoup  # 需要安装 beautifulsoup4 库来进行HTML解析。可以通过 `pip install beautifulsoup4` 进行安装。
import csv  # csv 库是Python标准库，不需要额外安装。
import time  # time 库是Python标准库，用于增加随机延迟以防止被反爬虫检测。
import random  # random 库是Python标准库，用于生成随机数来增加请求间隔的随机性。


def fetch_data(url):
    # 请求网页内容
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36'
    }
    response = requests.get(url, headers=headers)
    response.encoding = 'utf-8'
    if response.status_code == 200:
        return response.json()
    else:
        print(f"请求失败，状态码: {response.status_code}")
        return None


def parse_data(json_content):
    # 从JSON响应中解析数据
    if not json_content or 'value' not in json_content or 'list' not in json_content['value']:
        print("没有找到指定的表格数据！")
        return []

    data_list = json_content['value']['list']
    data = []

    # 遍历每一项，获取需要的数据
    for item in data_list:
        row_data = [
            item.get('lotteryDrawNum', ''),
            item.get('lotteryDrawTime', ''),
            item.get('lotteryDrawResult', []),
            item.get('totalSaleAmountRj', ''),
            item.get('prizeLevelListRj','')[0].get('stakeCount', ''),
            item.get('prizeLevelListRj','')[0].get('stakeAmount', ''),
            item.get('totalSaleAmount', ''),
            item.get('prizeLevelList', '')[0].get('stakeCount', ''),
            item.get('prizeLevelList', '')[0].get('stakeAmount', ''),
            item.get('prizeLevelList', '')[1].get('stakeCount', ''),
            item.get('prizeLevelList', '')[1].get('stakeAmount', '')


        ]
        data.append(row_data)

    return data


def save_to_csv(data, filename, header_written):
    # 保存数据到CSV文件
    with open(filename, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # 写入表头（只在第一次写入时）
        if not header_written:
            writer.writerow(['game_id', 'result_date', 'results','sale_amount_R9','prize_count_R9','prize_R9','sale_amount_14','prize_count_14_1','prize_14_1','prize_count_14_2','prize_14_2'])
        # 写入数据
        writer.writerows(data)


def main():
    base_url = "https://webapi.sporttery.cn/gateway/lottery/getHistoryPageListV1.qry?gameNo=90&provinceId=0&pageSize=30&isVerify=1&pageNo={}"
    all_data = []
    header_written = False
    filename = 'history_data.csv'

    # 遍历页码，获取所有数据
    for page_no in range(47, 48):
        url = base_url.format(page_no)
        json_content = fetch_data(url)
        if json_content:
            page_data = parse_data(json_content)
            if page_data:
                save_to_csv(page_data, filename, header_written)
                header_written = True
        # 增加随机延迟，防止被反爬机制封锁
        time.sleep(random.uniform(1, 3))

    print("数据已成功保存到 history_data.csv 文件中！")


if __name__ == "__main__":
    main()
