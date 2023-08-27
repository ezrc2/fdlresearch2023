from transformers import BertTokenizer, AutoTokenizer, BartForConditionalGeneration
import torch
import requests
from trafilatura import fetch_url, extract
from trafilatura.settings import use_config
from bs4 import BeautifulSoup
from google_images_search import GoogleImagesSearch
import urllib
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait 
from selenium.webdriver.support import expected_conditions as EC
import re
from rake_nltk import Rake
import spacy
import pickle

from keys import API_KEY, SEARCH_ENGINE_ID

categories = ['academics', 'alumni', 'campus', 'history', 'studentlife']
possible_majors = ['accounting', 'agricultural science', 'anthropology', 'architecture', 'art', 'biology', 'business', 'chemistry', 'communications', 'computer science', 'criminal justice', 'culinary arts', 
                   'dental studies', 'design', 'economics', 'education', 'engineering', 'english', 'environmental science', 'film', 'finance', 'foreign language', 'history', 'information science', 'kinesiology', 
                   'law', 'math', 'music', 'nursing', 'nutrition', 'performing arts', 'pharmacy', 'philosophy', 'physics', 'political science', 'psychology', 'religion', 'sociology', 'statistics']

nlp = spacy.load('en_core_web_md')
def similarity(word1, word2):
    token1 = nlp(word1)[0]
    token2 = nlp(word2)[0]
    return token1.similarity(token2)

# Find the university's common name
# For example, if 'NYU' is searched, this will return 'New York University'
def find_university_name(search_name):
    url = 'https://api.ror.org/organizations?query='
    response = requests.get(url + search_name)
    json_data = response.json()
    if json_data['items']:
        return json_data['items'][0]['name']
    return None  # invalid name


# Reference: https://pypi.org/project/Google-Images-Search/
# def get_image_urls(school_name):
#     gis = GoogleImagesSearch(API_KEY, IMAGE_SEARCH_ENGINE_ID)
#     _search_params = {
#         'q': 'UIUC campus',
#         'num': 3,
#         'fileType': 'jpg|gif|png',
#         'rights': 'cc_publicdomain|cc_attribute|cc_sharealike|cc_noncommercial|cc_nonderived'
#     }

#     campus_urls = []
#     gis.search(search_params=_search_params)
#     for image in gis.results():
#         campus_urls.append(image.url)

#     _search_params['q'] = 'UIUC logo'
#     logo_urls = []
#     gis.search(search_params=_search_params)
#     for image in gis.results():
#         logo_urls.append(image.url)

#     return [campus_urls, logo_urls]  # has to pass by reference

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
    try:    
        config = use_config()
        config.set("DEFAULT", "EXTRACTION_TIMEOUT", "0")
        downloaded = fetch_url(url)
        result = extract(downloaded, config=config, output_format='xml',
                        include_links=True, include_formatting=True)
    except:
        return []
    if result is None:
        return []

    soup = BeautifulSoup(result, 'lxml')
    paragraphs = []
    for p in soup.find_all('p'):
        text = p.get_text(strip=True, separator='\n')
        if '.' in text:
            paragraphs.append(text)
    return paragraphs

def find_popular_majors(school_name):
    service = Service()
    options = webdriver.ChromeOptions()
    options.binary_location = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36")
    driver = webdriver.Chrome(service=service, options=options)

    url = f'https://www.usnews.com/search/education#gsc.tab=0&gsc.q={school_name}%20academics%20majors&gsc.sort='
    driver.get(url)

    try:
        elem = WebDriverWait(driver, 40).until(EC.presence_of_element_located((By.TAG_NAME, 'li')))
        content = driver.find_elements(By.TAG_NAME, 'li')
        print('Content found')
    except:
        print('Website timed out')

    result = ''
    for c in content:
        if 'https://www.usnews.com/best-colleges/' in c.text:
            result = c.text
            break
    url = re.findall('https?://\S+', result)[0]
    headers = {'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.111 Safari/537.36'}
    page = requests.get(url, headers=headers)
    soup = BeautifulSoup(page.text, 'lxml')
    a = soup.find('h1', class_='Heading-sc-1w5xk2o-0 cSCNqo')
    text = a.find_next_sibling('p').text
    
    majors = text.split('include:')[1].split('. The average')[0].strip().split(';')
    if majors[-1][:4] == ' and':
        majors[-1] = majors[-1][4:]
    majors = [m.strip() for m in majors]

    final_majors = []
    for major_str in majors:
        lower = major_str.lower()
        if 'and related' in lower:
            i = lower.index('and related')
            major_str = major_str[:i].strip()
                
        if 'and support' in lower:
            i = lower.index('and support')
            major_str = major_str[:i].strip()
            
        if ',' in major_str and 'and' not in lower and 'studies' not in lower:
            lst = major_str.split(',')
            for m in lst:
                if m and 'and related' not in m.lower():
                    final_majors.append(m.strip())
        
        else:
            final_majors.append(major_str)
    
    r = Rake()
    major_keywords = []
    for fm in final_majors:
        r.extract_keywords_from_text(fm)
        keywords = r.get_ranked_phrases()
        words = ['science', 'studies']
        for w in words:
            if len(keywords) > 1 and w in keywords[0] and w not in keywords[1:]:
                for i in range(1, len(keywords)):
                    keywords[i] = keywords[i] + ' ' + w
        major_keywords.append(keywords)
    
    majors_to_search = set()
    for mks in major_keywords:
        for mk in mks:
            for pm in possible_majors:
                if similarity(pm, mk) > 0.7:
                    majors_to_search.add(mk)
                    
    return majors_to_search

def gather_university_information(school_name):
    majors_to_search = find_popular_majors(school_name)
    print('------majors search finished------')

    search_queries = {
        'alumni': ['notable alumni', 'career outcomes'],
        'campus_general': ['campus information'],
        'campus_famousbuildingslandmarks': ['famous buildings', 'landmarks'],
        'campus_locationcity': ['location and city'],
        'history_general': ['history about'],
        'history_milestones': ['important milestones'],
        'studentlife_clubsorganizations': ['student organizations', 'student social clubs'],
        'studentlife_athleticsintramurals': ['student athletics and intramural sports'],
        'studentlife_greeklife': ['greek life']
    }

    academics_result = {}
    for major in majors_to_search:
        academics_result[major] = []
        urls = get_urls_and_titles(f'{school_name} {major}')[0]
        for url in urls:
            paragraphs = extract_paragraphs(url)
            if paragraphs:
                academics_result[major].extend(paragraphs)
    
    non_academics_result = {}
    for query_category in search_queries.keys():
        queries = search_queries[query_category]
        non_academics_result[query_category] = []
        for q in queries:
            urls = get_urls_and_titles(f'{school_name} {q}')[0]
            for url in urls:
                paragraphs = extract_paragraphs(url)
                if paragraphs:
                    non_academics_result[query_category].extend(paragraphs)
    
    with open('academics_result.pkl', 'wb') as f:
        pickle.dump(academics_result, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open('non_academics_result.pkl', 'wb') as f:
        pickle.dump(non_academics_result, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open('majors_to_search.pkl', 'wb') as f:
        pickle.dump(majors_to_search, f)

    return academics_result, non_academics_result, majors_to_search





device = torch.device('cpu')
model = torch.load('bert_multiclass.pt', map_location=device)
tokenizer = BertTokenizer.from_pretrained(
    "bert-base-uncased", do_lower_case=True, force_download=True, resume_download=False)
model.eval()

bart_model = BartForConditionalGeneration.from_pretrained(
    "facebook/bart-large-cnn")
summary_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")


def model_pred(input_text):
    input_text_tokenized = tokenizer.encode(input_text,
                                            truncation=True,
                                            padding=True,
                                            return_tensors="pt").to(device)
    prediction = model(input_text_tokenized)
    prediction_logits = prediction[0]
    softmax = torch.nn.Softmax(dim=1)
    idx = torch.argmax(softmax(prediction_logits))
    return categories[idx]


def summarize_text(input_text):
    max_length = 300  # len(input_text.split())
    min_length = 20  # max_length // 4
    inputs = summary_tokenizer(
        [input_text], max_length=1024, return_tensors="pt", truncation=True)
    summary_ids = bart_model.generate(
        inputs["input_ids"], num_beams=2, min_length=min_length, max_length=max_length)
    summary = summary_tokenizer.batch_decode(
        summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False, truncation=True)[0]
    return summary


def call_wikidata_api(query):
    # https://www.jcchouinard.com/wikidata-api-python/
    def fetch_wikidata(params):
        url = 'https://www.wikidata.org/w/api.php'
        try:
            return requests.get(url, params=params)
        except:
            return 'API error'
        
    params = {
        'action': 'wbsearchentities',
        'format': 'json',
        'search': query,
        'language': 'en'
    }
    json_data = fetch_wikidata(params).json()
    id_ = json_data['search'][0]['id']
    params = {
        'action': 'wbgetentities',
        'ids': id_,
        'format': 'json',
        'languages': 'en'
    }

    data = fetch_wikidata(params).json()
    try:
        title = data['entities'][id_]['labels']['en']['value']
    except:
        title = 'not found'
    try:
        alternate_names = [v['value']
                           for v in data['entities'][id_]['aliases']['en']]
    except:
        alternate_names = 'not found'
    try:
        description = data['entities'][id_]['descriptions']['en']['value']
    except:
        description = 'not found'
    try:
        twitter = data['entities'][id_]['claims']['P2002'][0]['mainsnak']['datavalue']['value']
    except:
        twitter = 'not found'
    try:
        instagram = data['entities'][id_]['claims']['P2003'][0]['mainsnak']['datavalue']['value']
    except:
        instagram = 'not found'
    try:
        subreddit = data['entities'][id_]['claims']['P3984'][0]['mainsnak']['datavalue']['value']
    except:
        subreddit = 'not found'
    try:
        official_websites = [v['mainsnak']['datavalue']['value']
                             for v in data['entities'][id_]['claims']['P856']]
    except:
        official_websites = 'not found'
    try:
        logo_file = data['entities'][id_]['claims']['P154'][0]['mainsnak']['datavalue']['value'].replace(
            ' ', '_')
        temp_url = f'www.wikidata.org/wiki/{id_}#/media/File:{logo_file}'
        page = urllib.request.urlopen('https://' + temp_url)
        soup = BeautifulSoup(page, features='lxml')
        images = soup.findAll('img')
        logo_image_url = 'https:' + images[0]['src']
        campus_image_url = 'https:' + images[1]['src']
    except:
        logo_image_url = 'not found'
        campus_image_url = 'not found'


    alternate_names = ', '.join(alternate_names)

    result = (title, description, alternate_names, twitter, instagram, subreddit, official_websites, logo_image_url, campus_image_url)

    return result
