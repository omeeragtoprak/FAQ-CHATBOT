from bs4 import BeautifulSoup
import requests
import json

urls = [
    "https://www.garantibbva.com.tr/sikca-sorulan-sorular/hesaplar",
    "https://www.garantibbva.com.tr/sikca-sorulan-sorular/odemeler",
    "https://www.garantibbva.com.tr/sikca-sorulan-sorular/yatirim",
    "https://www.garantibbva.com.tr/sikca-sorulan-sorular/krediler",
    "https://www.garantibbva.com.tr/sikca-sorulan-sorular/kredi-kartlari",
    "https://www.garantibbva.com.tr/sikca-sorulan-sorular/uye-isyeri-pos",
    "https://www.garantibbva.com.tr/sikca-sorulan-sorular/dijital-bankacilik",
    "https://www.garantibbva.com.tr/sikca-sorulan-sorular/sigorta",
    "https://www.garantibbva.com.tr/sikca-sorulan-sorular/emeklilik",
    "https://www.garantibbva.com.tr/sikca-sorulan-sorular/mortgage",
    "https://www.garantibbva.com.tr/sikca-sorulan-sorular/yatirimci-iliskileri",
    "https://www.garantibbva.com.tr/sikca-sorulan-sorular/yetenek-ve-kultur"
]

faq_list = []

for url in urls:
    page = requests.get(url)
    soup = BeautifulSoup(page.text, 'html.parser')
    
    headings = soup.find_all('h3')
    paragraphs = soup.find_all("div", class_="rich-text")
    
    for index, (heading, paragraph) in enumerate(zip(headings, paragraphs)):
        faq_list.append({
            'index': len(faq_list),
            'question': heading.text.strip(),
            'answer': paragraph.text.strip(),
            'url': url
        })

with open('soru_cevap.json', 'w', encoding='utf-8') as f:
    json.dump(faq_list, f, ensure_ascii=False, indent=4)

