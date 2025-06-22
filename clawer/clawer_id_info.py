import requests
from bs4 import BeautifulSoup
import time
import random
import csv
from fake_useragent import UserAgent

# 设置请求头
ua = UserAgent()
headers = {
    'User-Agent': 'Edge/117.0',
    'Referer': 'https://movie.douban.com/',
    'Host': 'movie.douban.com'
}



# 存储电影信息的列表
movies = []


def get_movie_info(url):
    try:
        # 随机延迟，避免触发反爬
        time.sleep(random.uniform(1, 3))

        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')

            # 提取电影信息
            title = soup.find('span', property='v:itemreviewed').text if soup.find('span',
                                                                                   property='v:itemreviewed') else 'N/A'
            year = soup.find('span', class_='year').text.strip('()') if soup.find('span', class_='year') else 'N/A'
            rating = soup.find('strong', class_='ll rating_num').text if soup.find('strong',
                                                                                   class_='ll rating_num') else 'N/A'
            votes = soup.find('span', property='v:votes').text if soup.find('span', property='v:votes') else 'N/A'

            # 导演和演员
            directors = [a.text for a in soup.find_all('a', rel='v:directedBy')]
            actors = [a.text for a in soup.find_all('a', rel='v:starring')][:5]  # 只取前5个主要演员

            # 类型和地区/时长
            genres = [a.text for a in soup.find_all('span', property='v:genre')]
            info = soup.find('div', id='info').text if soup.find('div', id='info') else ''

            # 简介
            summary = soup.find('span', property='v:summary').text.strip() if soup.find('span',
                                                                                        property='v:summary') else 'N/A'

            # 封面图片
            cover = soup.find('img', rel='v:image')['src'] if soup.find('img', rel='v:image') else 'N/A'

            movie_info = {
                'id': url.split('/')[-2],
                'title': title,
                'year': year,
                'rating': rating,
                'votes': votes,
                'directors': ', '.join(directors),
                'actors': ', '.join(actors),
                'genres': ', '.join(genres),
                'info': info,
                'summary': summary,
                'cover': cover,
                'url': url
            }

            movies.append(movie_info)
            print(f"已获取: {title} ({year})")

        else:
            print(f"请求失败: {url}, 状态码: {response.status_code}")

    except Exception as e:
        print(f"获取电影信息时出错: {url}, 错误: {str(e)}")


def get_movie_links(start_url, num_movies=500):
    collected = 0
    start = 0

    while collected < num_movies:
        url = f"{start_url}?start={start}&filter="
        try:
            time.sleep(random.uniform(2, 5))
            response = requests.get(url, headers=headers)

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                items = soup.find_all('div', class_='info')

                if not items:
                    print("没有找到更多电影，可能已达到限制")
                    break

                for item in items:
                    if collected >= num_movies:
                        break

                    link = item.find('a')['href']
                    get_movie_info(link)
                    collected += 1

                start += len(items)
                print(f"已收集 {collected} 部电影")

            else:
                print(f"获取列表页失败: {url}, 状态码: {response.status_code}")
                break

        except Exception as e:
            print(f"获取列表页时出错: {url}, 错误: {str(e)}")
            break


def save_to_csv(filename='douban_movies.csv'):
    if not movies:
        print("没有电影数据可保存")
        return

    keys = movies[0].keys()
    with open(filename, 'w', newline='', encoding='utf-8-sig') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(movies)
    print(f"已保存 {len(movies)} 部电影信息到 {filename}")


if __name__ == '__main__':
    # 豆瓣电影Top250页面
    top250_url = 'https://movie.douban.com/top250'

    # 获取电影链接和信息
    get_movie_links(top250_url, num_movies=250)  # 先获取Top250
    # # 保存到CSV
    save_to_csv(filename='Top250_Douban.csv')
    #
    # movies.clear()
    get_movie_links(top250_url, num_movies=20)  # 再从探索页面获取250部
    save_to_csv(filename='douban_new_movies_explore.csv')
