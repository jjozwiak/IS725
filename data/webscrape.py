import requests
from bs4 import BeautifulSoup

mlb_pr_urls = [
    'https://www.mlb.com/whitesox/news/topic/white-sox-press-releases',
    'https://www.mlb.com/news/topic/orioles-press-releases',
    'https://www.mlb.com/news/topic/nationals-press-releases'
]

with open('pr_urls.txt') as f:
    for (index, line) in enumerate(f.readlines()):
        r = requests.get(line)
        soup = BeautifulSoup(r.content, 'html5lib')
        title = soup.find('h1', attrs={'class': 'article-item__headline'})
        title_text = title.text
        article = soup.find('div', attrs={'class': 'article-item__body'})
        article_file = open('pr_' + str(index) + '.txt', "w")
        article_file.write(article.text)
        article_file.close()