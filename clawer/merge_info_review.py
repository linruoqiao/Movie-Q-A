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
    cnt = 0
    for id in df_film['id'].tolist():
        if id != 1292226:
            continue
        df_comments = pd.read_csv(f'DoubanComments/{id}_Comments.csv',encoding='utf-8',header=0)
        info = df_film[df_film['id']==id]
        print(info)
        name = info['title'].to_string(index=False).replace(':',' ')
        # print(name)

        with open(f'DoubanTop250/{name}_Info.txt','w',encoding='utf-8') as f:
            f.write("电影基本信息:\n")
            for col in info.columns:
                if col == 'id':
                    continue
                content = str(info[col].values[0]).replace('\n', ' ').replace('\r', '')
                f.write(col + ': ' + content + '\n')

            f.write('\n六十条豆瓣评论:\n')
            for index, row in df_comments.iterrows():
                time_str = row['time']
                rating = row['rating']
                content = row['content']
                f.write(f"时间: {time_str}\n")
                f.write(f"评分: {rating}\n")
                f.write(f"内容: {content}\n")
                f.write("\n")


