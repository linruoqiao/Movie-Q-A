import requests
from bs4 import BeautifulSoup
import time
import re
import pandas as pd
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

def get_comments(id,start=0):
    url = f"https://movie.douban.com/subject/{id}/comments?start={start}&limit=20&status=P&sort=new_score"
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print("访问失败")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    comment_blocks = soup.find_all("div", class_="comment-item")

    results = []
    for item in comment_blocks:
        # 时间
        time_tag = item.find("span", class_="comment-time")
        time_str = time_tag["title"] if time_tag else ""

        # 评分
        rating = ""
        rating_tag = item.find("span", class_=re.compile("allstar\d+"))
        if rating_tag:
            match = re.search(r'allstar(\d+)', rating_tag["class"][0])
            if match:
                rating = int(match.group(1)) // 10  # allstar50 -> 5星

        # 内容
        content_tag = item.find("span", class_="short")
        content = content_tag.text.strip() if content_tag else ""

        results.append((time_str, rating, content))
    return results

if __name__ == '__main__':
    df_film = pd.read_csv('Top250_Douban.csv',encoding='utf-8',header=0)
    for id in df_film['id'].tolist():
        # continue
        df_comments = pd.DataFrame(columns=['time', 'rating', 'content'])
        # 主循环，获取多页数据
        all_comments = []
        for i in range(0, 3):
            print(f"正在抓取第 {i+1} 页...")
            comments = get_comments(id, start=i * 20)
            all_comments.extend(comments)
            time.sleep(2)  # 防止被封

        # 输出示例
        for t, r, c in all_comments:
            df_comments.loc[len(df_comments)] = [t, r, c]

        # df_comments.to_json(f"DoubanComments/{id}.json", orient='records')
        df_comments.to_csv(f'DoubanComments/{id}_Comments.csv', encoding='utf-8', index=False)
        # print(df_comments)
        # break