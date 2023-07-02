from flask import Flask, render_template, request, url_for
from functions import *
import sqlite3

app = Flask(__name__)

categories = ['academics', 'alumni', 'campus', 'history', 'studentlife']
query_categories = ['academics_overall', 'academics_business', 'academics_socialsciencehistory', 'academics_computerscience', 'academics_engineering', 'academics_biology', 'academics_psychology', 'academics_visualperformingarts', 'academics_communicationjournalism', 
                    'alumni', 
                    'campus_general', 'campus_famousbuildingslandmarks', 'campus_locationcity', 
                    'history_general', 'history_milestones', 
                    'studentlife_clubsorganizations', 'studentlife_athleticsintramurals', 'studentlife_greeklife']

@app.route('/')
def main():
    return render_template('search.html')

@app.route('/result', methods=['GET', 'POST'])
def search_result():
    if request.method == 'POST':
        con = sqlite3.connect("wiki_data.db")
        cur = con.cursor()
        res = cur.execute('SELECT DISTINCT institution_name FROM raw_paragraphs;')
        paragraph_institution_names = []
        for x in res.fetchall():
            paragraph_institution_names.append(x[0])

        query = request.form.get('search_query')
        #name = find_university_name(query)
        wikidata_result = call_wikidata_api(query)
        name = wikidata_result[0]

        if not name:
            return render_template('error_page.html', query=query)
        
        elif name in paragraph_institution_names:
            # generate wiki page

            res = cur.execute('SELECT DISTINCT title FROM wiki_page;')
            wiki_institution_names = []
            for x in res.fetchall():
                wiki_institution_names.append(x[0])

            results = {}
            if name in wiki_institution_names:
                print('------summaries already present------')
                res = cur.execute(f"SELECT * FROM wiki_page where title='{name}'")
                x = res.fetchall()
                for i, query_category in enumerate(query_categories):
                    results[query_category] = x[0][i+1]
            else: # summarize the paragraphs
                print('------summarizing------')
                paragraphs_to_summarize = {}
                res = cur.execute(f"SELECT * FROM raw_paragraphs where institution_name='{name}'")
                for x in res.fetchall():
                    category, paragraph = x[0], x[1]
                    paragraphs_to_summarize[category] = paragraph
                
                values = [name]
                for query_category, text in paragraphs_to_summarize.items():
                    results[query_category] = text
                    values.append(text)
                #     summary = summarize_text(text)
                #     values.append(summary)
                #     results[query_category] = summary

                # values = tuple(values)
                
                # sql = """INSERT INTO wiki_page (title, academics_overall, academics_business, academics_socialsciencehistory, academics_computerscience, academics_engineering, academics_biology, academics_psychology, academics_visualperformingarts, academics_communicationjournalism, 
                #     alumni, 
                #     campus_general, campus_famousbuildingslandmarks, campus_locationcity, 
                #     history_general, history_milestones, 
                #     studentlife_clubsorganizations, studentlife_athleticsintramurals, studentlife_greeklife) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
                # cur.execute(sql, values)
                # con.commit()
                # print('------inserted summaries------')

            # res = cur.execute(f'SELECT * FROM image_urls WHERE institution_name={name}')
            # images = {'campus':[], 'logo':[]}
            # for x in res.fetchall():
            #     url, image_type = x[1], x[2]
            #     images[image_type].append(url)
            

            return render_template('wiki_page.html', results=results, wikidata_result=wikidata_result, name=name)
        
        else:
            # scrape paragraphs and image urls
            print(name)
            result = gather_university_information(name)
            print('------finished scraping information------')
            
            paragraphs_to_keep = {}
            
            for query_category in query_categories:
                paragraphs_to_keep[query_category] = ''
                paragraphs = result[query_category]
                for p in paragraphs:
                    #pred = model_pred(p)
                    #category = query_category.split('_')[0] # main category
                    if True:#pred.lower() == category.lower():
                        paragraphs_to_keep[query_category] += p

            values = []
            for query_category, text in paragraphs_to_keep.items():
                values.append((query_category, text, name))
            sql = 'INSERT INTO raw_paragraphs (label, paragraph, institution_name) VALUES (?, ?, ?)'
            cur.executemany(sql, values)
            con.commit()
            print('------inserted paragraphs into database------')

            # campus_urls, logo_urls = get_image_urls(name)
            # sql = 'INSERT INTO image_urls (institution_name, url, image_type) VALUES (?, ?, ?)'
            # values = []
            # for url in campus_urls:
            #     values.append((name, url, 'campus'))
            # cur.executemany(sql, values)
            # con.commit()
            # values = []
            # for url in logo_urls:
            #     values.append((name, url, 'logo'))
            # cur.executemany(sql, values)
            # con.commit()
            # msg2 = 'Images successfully gathered '
            # print('------inserted images------')

            # wikidata_result = call_wikidata_api(name)
            # sql = 'INSERT INTO wikidata_info (title, description, alternate_names, twitter, facebook, instagram, subreddit, website, logo_image_url, campus_image_url) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'
            # cur.execute(sql, wikidata_result)
            # con.commit()
            # print('------inserted wikidata information into database------')
            msg1 = 'Information successfully gathered.'

            return render_template('temp_page.html', name=name)#, msg1=msg1)#, msg2=msg2)