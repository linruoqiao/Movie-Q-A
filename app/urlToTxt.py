import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import re
from typing import Optional, Dict, List
import trafilatura
from trafilatura.settings import use_config


class URLTextExtractor:
    def __init__(self):
        # 配置请求头，模拟Edge浏览器访问
        self.headers = {
            'User-Agent': 'Edge/114.0.1823.58'
        }

        # 配置trafilatura (用于更好的正文提取)
        self.trafilatura_config = use_config()
        self.trafilatura_config.set("DEFAULT", "EXTRACTION_TIMEOUT", "0")  # 禁用超时

    def is_valid_url(self, url: str) -> bool:
        """检查URL是否有效"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False

    def fetch_html(self, url: str) -> Optional[str]:
        """获取网页HTML内容"""
        if not self.is_valid_url(url):
            print(f"无效的URL: {url}")
            return None

        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()

            # 检查内容类型是否为HTML
            content_type = response.headers.get('content-type', '')
            if 'text/html' not in content_type:
                print(f"URL不包含HTML内容: {url}")
                return None

            return response.text
        except requests.exceptions.RequestException as e:
            print(f"获取URL内容失败: {url}, 错误: {str(e)}")
            return None

    def extract_with_trafilatura(self, html: str, url: str) -> Optional[str]:
        """使用trafilatura提取正文内容"""
        try:
            extracted = trafilatura.extract(
                html,
                url=url,
                config=self.trafilatura_config,
                include_comments=False,
                include_tables=False,
                include_links=False
            )
            return extracted
        except Exception as e:
            print(f"使用trafilatura提取失败: {str(e)}")
            return None

    def extract_with_bs4(self, html: str) -> Optional[str]:
        """使用BeautifulSoup提取正文内容"""
        try:
            soup = BeautifulSoup(html, 'html.parser')

            # 移除不需要的标签
            for element in soup(['script', 'style', 'nav', 'footer', 'iframe', 'noscript']):
                element.decompose()

            # 获取所有文本
            text = soup.get_text(separator='\n', strip=True)

            # 清理多余的空格和换行
            text = re.sub(r'\n{3,}', '\n\n', text)
            text = re.sub(r'[ \t]{2,}', ' ', text)

            return text.strip()
        except Exception as e:
            print(f"使用BeautifulSoup提取失败: {str(e)}")
            return None

    def clean_text(self, text: str) -> str:
        """清理提取的文本"""
        if not text:
            return ""

        # 移除特殊字符和多余空格
        text = re.sub(r'[\x00-\x1F\x7F-\x9F]', ' ', text)  # 移除控制字符
        text = re.sub(r'[ \t]{2,}', ' ', text)  # 多个空格变一个
        text = re.sub(r'\n{3,}', '\n\n', text)  # 多个换行变两个

        return text

    def extract_from_url(self, url: str) -> Dict[str, str]:
        """
        从URL提取文本内容

        返回:
            {
                "url": 原始URL,
                "text": 提取的文本内容,
                "method": 使用的提取方法,
                "error": 错误信息(如果有)
            }
        """
        result = {
            "url": url,
            "text": "",
            "method": None,
            "error": None
        }

        if not self.is_valid_url(url):
            result["error"] = "无效的URL"
            return result

        html = self.fetch_html(url)
        if not html:
            result["error"] = "无法获取HTML内容"
            return result

        # 首先尝试使用trafilatura提取
        text = self.extract_with_trafilatura(html, url)
        if text:
            result["text"] = self.clean_text(text)
            result["method"] = "trafilatura"
            return result

        # 如果trafilatura失败，回退到BeautifulSoup
        text = self.extract_with_bs4(html)
        if text:
            result["text"] = self.clean_text(text)
            result["method"] = "beautifulsoup"
            return result

        result["error"] = "无法提取文本内容"
        return result

    def batch_extract(self, urls: List[str]) -> List[Dict[str, str]]:
        """批量提取多个URL的文本内容"""
        return [self.extract_from_url(url) for url in urls]


# 使用示例
# if __name__ == "__main__":
#     extractor = URLTextExtractor()
#
#     url = "https://movie.douban.com/subject/36251574/"
#     result = extractor.extract_from_url(url)
#
#     print(f"URL: {result['url']}")
#     # print(f"提取方法: {result['method']}")
#     with open('test.txt', 'w', encoding='utf-8') as f:
#         f.write(result['text'])
#     print(f"提取结果: {result['text']}...")  # 只打印前200个字符

    # urls = [
    #     "https://example.com",
    #     "https://example.org",
    #     "https://example.net"
    # ]
    # results = extractor.batch_extract(urls)
    #
    # for res in results:
    #     print(f"\nURL: {res['url']}")
    #     print(f"提取方法: {res['method']}")
    #     print(f"文本长度: {len(res['text'])} 字符")