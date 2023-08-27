from flask import Flask, render_template, request, url_for
from functions import *
import sqlite3

app = Flask(__name__)

categories = ['academics', 'alumni', 'campus', 'history', 'studentlife']
query_categories = [#'academics', 
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
        res = cur.execute('SELECT DISTINCT institution_name FROM raw_paragraphs_non_academics;')
        paragraph_institution_names = []
        for x in res.fetchall():
            paragraph_institution_names.append(x[0])

        query = request.form.get('search_query')
        wikidata_result = call_wikidata_api(query)
        name = wikidata_result[0]

        if not name:
            return render_template('error_page.html', query=query)
        
        elif name in paragraph_institution_names:
            
            # generate wiki page
            res = cur.execute('SELECT DISTINCT title FROM wiki_page_non_academics;')
            wiki_institution_names = []
            for x in res.fetchall():
                wiki_institution_names.append(x[0])

            results = {}
            if name in wiki_institution_names:
                print('------summaries already present------')
                # res = cur.execute(f"SELECT * FROM wiki_page_academics where title='{name}'")
                # x = res.fetchall()
                # for i, query_category in enumerate(query_categories):
                #     results[query_category] = x[0][i+1]

                res = cur.execute(f"SELECT * FROM raw_paragraphs_page_non_academics where title='{name}'")
                x = res.fetchall()
                for i, query_category in enumerate(query_categories):
                    results[query_category] = x[0][i+1]

            else: # summarize the paragraphs
                print('------summarizing------')
                # academics_paragraphs_to_summarize = {}
                # res = cur.execute(f"SELECT * FROM raw_paragraphs_academics where institution_name='{name}'")

                non_academics_paragraphs_to_summarize = {}
                res = cur.execute(f"SELECT * FROM raw_paragraphs_non_academics where institution_name='{name}'")
                for x in res.fetchall():
                    category, paragraph = x[0], x[1]
                    non_academics_paragraphs_to_summarize[category] = paragraph
                
                values = [name]
                for query_category, text in non_academics_paragraphs_to_summarize.items():
                    results[query_category] = text
                    # values.append(text)
                    # summary = summarize_text(text)
                    # values.append(summary)
                    # results[query_category] = summary

                # values = tuple(values)
                # sql = """INSERT INTO wiki_page (title, 
                #     alumni, 
                #     campus_general, campus_famousbuildingslandmarks, campus_locationcity, 
                #     history_general, history_milestones, 
                #     studentlife_clubsorganizations, studentlife_athleticsintramurals, studentlife_greeklife) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
                # cur.execute(sql, values)
                # con.commit()
                print('------inserted summaries------')

            return render_template('wiki_page.html', results=results, wikidata_result=wikidata_result, name=name)
        
        else:
            # scrape paragraphs
            print(name)
            academics_result, non_academics_result, majors_to_search = gather_university_information(name)
    
            print('------finished scraping information------')

            #pred = model_pred(p)
            #category = query_category.split('_')[0] # main category
            #if pred.lower() == category.lower():

            values = []
            for major, paragraphs in academics_result.items():
                values.append((name, ' '.join(paragraphs), major))
            sql = 'INSERT INTO raw_paragraphs_academics (institution_name, paragraph, major) VALUES (?, ?, ?)'
            cur.executemany(sql, values)
            con.commit()
            
            values = []
            for query_category, paragraphs in non_academics_result.items():
                values.append((query_category, ' '.join(paragraphs), name))
            sql = 'INSERT INTO raw_paragraphs_non_academics (label, paragraph, institution_name) VALUES (?, ?, ?)'
            cur.executemany(sql, values)
            con.commit()
            print('------inserted paragraphs into database------')

            values = []
            for major in majors_to_search:
                values.append((name, major))
            sql = 'INSERT INTO popular_majors (institution_name, major) VALUES (?, ?)'
            cur.executemany(sql, values)
            con.commit()
            print('------inserted popular majors into database------')

            return render_template('temp_page.html', name=name)