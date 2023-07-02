import streamlit as st
import requests
from trafilatura import fetch_url, extract
from trafilatura.settings import use_config
from bs4 import BeautifulSoup
import sqlite3

API_KEY = ...
SEARCH_ENGINE_ID = ...

title = st.text_input(label='Academic Web Scraper', placeholder='Enter university name')
if title:
    # Data description
    # Academics
    # - general academic information
    # - information about popular majors
    #     - business
    #     - social science and history
    #     - computer science
    #     - engineering
    #     - biology
    #     - pyschology
    #     - visual and performing arts
    #     - communication and journalism

    # Alumni
    # - famous alumni
    # - post graduation results (like on LinkedIn)

    # Campus
    # - campus information
    # - famous buildings
    # - landmarks
    # - school location and city

    # History
    # - history of the school
    # - important dates and milestones

    # Student life
    # - student life
    # - student organizations
    # - student athletics and intramural sports
    # - social clubs
    # - student housing
    # - greek life
    categories = ['Academics', 'Alumni', 'Campus', 'History', 'Student Life']
    search_queries = {
        'Academics' : ['academic information', 'business', 'social science and history', 'computer science', 'engineering', 'biology', 'psychology', 'visual and performing arts', 'communication and journalism'],
        'Alumni' : ['notable alumni', 'career outcomes'],
        'Campus' : ['campus', 'famous buildings', 'location and city', 'landmarks'],
        'History' : ['history about', 'important milestones'],
        'Student Life' : ['student life', 'student organizations', 'student athletics and intramural sports', 'student social clubs', 'student housing', 'greek life']
    }

    @st.cache
    def get_all_information(school_name):
        
        def get_urls_and_titles(query):    
            urls = []
            titles = []
            page = 1
            start = (page - 1) * 10 + 1
            url = f"https://www.googleapis.com/customsearch/v1?key={API_KEY}&cx={SEARCH_ENGINE_ID}&q={query}&start={start}"
            data = requests.get(url).json()
            search_items = data.get("items")
            for search_item in search_items:
                title = search_item.get("title")
                link = search_item.get("link")
                urls.append(link)
                titles.append(title)
            return [urls, titles]

        def extract_paragraphs(url):
            config = use_config()
            config.set("DEFAULT", "EXTRACTION_TIMEOUT", "0")
            downloaded = fetch_url(url)
            result = extract(downloaded, config=config, output_format='xml', include_links=True, include_formatting=True)
            if result is None:
                return []

            soup = BeautifulSoup(result, 'lxml')
            paragraphs = []
            for p in soup.find_all('p'):
                text = p.get_text(strip=True, separator='\n')
                if '.' in text:
                    paragraphs.append(text)
            return paragraphs
        
        result = {'Academics':[], 'Alumni':[], 'Campus':[], 'History':[], 'Student Life':[]}
        for category in categories:
            queries = search_queries[category]
            for q in queries:
                urls = get_urls_and_titles(f'{school_name} {q}')[0]
                for url in urls:
                    paragraphs = extract_paragraphs(url)
                    if paragraphs:
                        result[category].append((url, paragraphs))
        return result

    result = get_all_information(title)
    con = sqlite3.connect('dataset.db')
    cur = con.cursor()

    values = []
    k = 0
    for category in categories:
        st.subheader(category)
        data = result[category]
        for pair in data:
            url, paragraphs = pair
            st.write(url)
            for p in paragraphs:
                cb = st.checkbox(p, key=k)
                k += 1
                if cb:
                    values.append((category, p))
            st.markdown('''---''')

    b = st.button(label='Add checked passages to dataset')
    if b:
        sql = 'INSERT INTO paragraph_dataset VALUES (?, ?)'
        cur.executemany(sql, values)
        con.commit()
        st.write(f'Added {len(values)} rows ✅')