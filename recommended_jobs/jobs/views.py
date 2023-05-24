from django.shortcuts import render, redirect
from .forms import MyFileForm
from jobs.models import MyFile
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors
import os
from django.conf import settings
import PyPDF2
import codecs
import PyPDF2
import textract
import io
import re

from bs4 import BeautifulSoup as bts
import requests
import sys
import json
import csv
import string


def upload_file(request):
    if request.method == 'POST':
        form = MyFileForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('process_file')
    else:
        form = MyFileForm()
    return render(request, 'jobs/upload_file.html', {'form': form})

def process_file(request):
    def getAndResultURL(url):
        result = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        soup = bts(result.content, 'html.parser')
        return soup

    def extract_text(html):
        soup = bts(html, 'html.parser')
        text = ""

        for p in soup.find_all(['p', 'li']):
            text += p.get_text() + "\n"

        return text

    def remove_punctuation(text):
        return text.translate(str.maketrans("", "", string.punctuation))

    def find_matching_skills(text):
        file_path = os.path.join('jobs/documents','skill_list.txt')
        with open(file_path, 'r') as file:
            skills = [line.strip().lower() for line in file]

        matched_skills = set()

        text = remove_punctuation(text).lower()

        for skill in skills:
            if re.search(r'\W' + re.escape(skill) + r'\W', ' ' + text + ' '):
                matched_skills.add(skill)

        result = ', '.join(matched_skills)
        return result

    JOB_PAGE = []
    for pg in range(1, 18):
        JOB_PAGE.append("https://www.kariyer.net/is-ilanlari/yazilim+uzmani-2?pst=2024&pkw=Yazilim%20Uzmani&cp="+str(pg))

    JOB_LINK = []
    for pages in JOB_PAGE[::]:
        html = getAndResultURL(pages)
        for links in html.findAll("div", {"class": "list-items"}):
            JOB_LINK.append("https://www.kariyer.net/" + links.a["href"])

    disctionaryArr = []
    for detail in JOB_LINK[:]:
        html = getAndResultURL(detail)
        job_titles = html.find('a', {'class': 'link position'}).text.strip()
        company_name = html.find('a', {'class': 'link company'}).text.strip()
        requirement = html.find('div', {'class': 'genel-nitelikler'})
        clean_html = extract_text(str(requirement))
        finall = find_matching_skills(clean_html)

        # Skill kısmı boş olanları filtrele
        if finall:
            link1 = detail
            img_url = ''
            img_element = html.find('div', {'class': 'logo-wrapper'})
            if img_element and img_element.find('img'):
                img_src = img_element.find('img')['src']
            match = re.search(r'https://img-kariyer\.mncdn\.com/UploadFiles/Clients/Logolar/.+', img_src)
            if match:
                img_url = match.group(0)

            job_dict = {
                'title': job_titles,
                'description': company_name,
                "skill": finall,
                "link": link1,
                'img url': img_url,
            }
            disctionaryArr.append(job_dict)

    data = disctionaryArr
    output_file = os.path.join('jobs/documents', 'processed_requirement.csv')
    output = csv.writer(open(output_file, "w"))
    output.writerow(data[0].keys())  # header row
    for row in data:
        output.writerow(row.values())





    uploaded_file = MyFile.objects.latest('id')  # En son yüklenen dosyayı alın
    file_path = uploaded_file.file.path
    file_path_csv = os.path.join(settings.BASE_DIR, 'jobs/documents', 'processed_requirement.csv')
    if file_path.endswith('.txt'):
        with codecs.open(file_path, 'r', encoding='utf-8') as file:
            cv_text = file.read()
    elif file_path.endswith('.pdf'):
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            num_pages = len(pdf_reader.pages)

            text = ''
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                text += page.extract_text ()
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        #text = textract.process(text, method='tesseract', encoding='utf-8')
        cv_text = text
    else:
        raise ValueError("Desteklenmeyen dosya formatı.")

    df = pd.read_csv(file_path_csv, encoding="ISO-8859-9")
    tfidf_vectorizer = TfidfVectorizer(stop_words="english")
    job_data_tfidf = tfidf_vectorizer.fit_transform(df["skill"])
    cv_tfidf = tfidf_vectorizer.transform([cv_text])
    cosine_similarities = cosine_similarity(cv_tfidf, job_data_tfidf).flatten()

    def get_recommendation(top_indices, df_all, scores):
        recommendation = pd.DataFrame(columns=['JobID', 'title', 'description', 'skill', 'score', 'link'])
        count = 0
        for i in top_indices:
            recommendation.at[count, 'JobID'] = df_all.index[i]
            recommendation.at[count, 'title'] = df_all['title'][i]
            recommendation.at[count, 'description'] = df_all['description'][i]
            recommendation.at[count, 'skill'] = df_all['skill'][i]
            recommendation.at[count, 'score'] = scores[count]
            recommendation.at[count, 'link'] = df_all['link'][i]
            count += 1
        return recommendation

    def KNN(scraped_data, cv):
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')

        n_neighbors = 100
        KNN = NearestNeighbors(n_neighbors=n_neighbors, p=2)
        KNN.fit(tfidf_vectorizer.fit_transform(scraped_data))
        NNs = KNN.kneighbors(tfidf_vectorizer.transform(cv))
        top = NNs[1][0][1:]
        index_score = NNs[0][0][1:]

        knn = get_recommendation(top, df, index_score)
        return knn

    knn_recommendations = KNN(df['skill'], [cv_text])
    top_indices = sorted(range(len(cosine_similarities)), key=lambda i: cosine_similarities[i], reverse=True)[:100]
    list_scores = [cosine_similarities[i] for i in top_indices]
    recommendations = get_recommendation(top_indices, df, list_scores)
    
    return render(request, 'jobs/process_file.html', {
        'file_content': cv_text,
        'knn_recommendations': knn_recommendations,
        'recommendations': recommendations
    })