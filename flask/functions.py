from transformers import BertTokenizer, AutoTokenizer, BartForConditionalGeneration
import torch
import requests
from trafilatura import fetch_url, extract
from trafilatura.settings import use_config
from bs4 import BeautifulSoup
from google_images_search import GoogleImagesSearch
import urllib

from keys import API_KEY, SEARCH_ENGINE_ID

categories = ['academics', 'alumni', 'campus', 'history', 'studentlife']
query_categories = ['academics_overall', 'academics_business', 'academics_socialsciencehistory', 'academics_computerscience', 'academics_engineering', 'academics_biology', 'academics_psychology', 'academics_visualperformingarts', 'academics_communicationjournalism', 
                    'alumni', 
                    'campus_general', 'campus_famousbuildingslandmarks', 'campus_locationcity', 
                    'history_general', 'history_milestones', 
                    'studentlife_clubsorganizations', 'studentlife_athleticsintramurals', 'studentlife_greeklife']

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

def gather_university_information(school_name):

    search_queries = {
        'academics_overall': ['academics overall'],
        'academics_business': ['business'],
        'academics_socialsciencehistory': ['social science and history'],
        'academics_computerscience': ['computer science'],
        'academics_engineering': ['engineering'],
        'academics_biology': ['biology'],
        'academics_psychology': ['psychology'],
        'academics_visualperformingarts': ['visual and performing arts'],
        'academics_communicationjournalism': ['communication and journalism'],
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

    result = {}
    for query_category in query_categories:
        queries = search_queries[query_category]
        for q in queries:
            urls = get_urls_and_titles(f'{school_name} {q}')[0]
            for url in urls:
                paragraphs = extract_paragraphs(url)
                if paragraphs:
                    if query_category in result:
                        result[query_category].extend(paragraphs)
                    else:
                        result[query_category] = paragraphs
    return result


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
        map_image_url = 'https:' + images[2]['src']

    except:
        logo_image_url = 'not found'
        campus_image_url = 'not found'
    # try:
    #     campus_file = data['entities'][id_]['claims']['P18'][0]['mainsnak']['datavalue']['value'].replace(
    #         ' ', '_')
    #     temp_url = f'www.wikidata.org/wiki/{id_}#/media/File:{campus_file}'
    #     page = urllib.request.urlopen('https://' + temp_url)
    #     soup = BeautifulSoup(page, features='lxml')
    #     image = soup.findAll('img')[0]
    #     campus_image_url = 'https:' + image['src']
    # except:
    #     campus_image_url = 'not found'

    # result = {
    #     'wikidata_id': id_,
    #     'title': title,
    #     'description': description,
    #     'alternate_names': alternate_names,
    #     'twitter': twitter,
    #     'instagram': instagram,
    #     'subreddit': subreddit,
    #     'official_websites': official_websites,
    #     'logo_image_url': logo_image_url,
    #     'campus_image_url': campus_image_url
    # }

    alternate_names = ', '.join(alternate_names)

    result = (title, description, alternate_names, twitter, instagram, subreddit, official_websites, logo_image_url, campus_image_url)

    return result
