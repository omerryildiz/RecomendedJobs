from django.shortcuts import render,redirect
from django.http.response import HttpResponse
from jobs.models import Document
from .forms import DocumentForm
import csv
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
def index(requests):
    return HttpResponse("index")

def upload_document(request):
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('read_file')
    else:
        form = DocumentForm()
    return render(request, 'jobs/upload_document.html', {'form': form})

def read_file(file_path):
    df = pd.read_csv("jobs/requirements/processed_requirement.csv", encoding="ISO-8859-9")
    with open(file_path, "r", errors="ignore") as f:
        cv_text = f.read()
    tfidf_vectorizer = TfidfVectorizer(stop_words="english")
    job_data_tfidf = tfidf_vectorizer.fit_transform(df["skill"])
    cv_tfidf = tfidf_vectorizer.transform([cv_text])
    cosine_similarities = cosine_similarity(cv_tfidf, job_data_tfidf).flatten()
    recommended_jobs(df,cv_text,cosine_similarities)

def get_recommendation(top_indices, df_all, scores):
    recommendation = pd.DataFrame(columns=['JobID', 'title', 'description', 'skill', 'score'])
    count = 0
    for i in top_indices:
        recommendation.at[count, 'JobID'] = df_all.index[i]
        recommendation.at[count, 'title'] = df_all['title'][i]
        recommendation.at[count, 'description'] = df_all['description'][i]
        recommendation.at[count, 'skill'] = df_all['skill'][i]
        recommendation.at[count, 'score'] = scores[count]
        count += 1
    return recommendation

def KNN(scraped_data, cv):
    df = pd.read_csv("processed_requirement.csv", encoding="ISO-8859-9")
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')

    n_neighbors = 100
    KNN = NearestNeighbors(n_neighbors=n_neighbors, p=2)
    KNN.fit(tfidf_vectorizer.fit_transform(scraped_data))
    NNs = KNN.kneighbors(tfidf_vectorizer.transform(cv))
    top = NNs[1][0][1:]
    index_score = NNs[0][0][1:]

    knn = get_recommendation(top, df, index_score)
    return knn

def recommended_jobs(df,cv_text,cosine_similarities):
    knn_recommendations = KNN(df['skill'], [cv_text])
    top_indices = sorted(range(len(cosine_similarities)), key=lambda i: cosine_similarities[i], reverse=True)[:100]
    list_scores = [cosine_similarities[i] for i in top_indices]
    recommendations = get_recommendation(top_indices, df, list_scores)
    recommendations.to_csv("requirements/job_recommendations_cosine_similarity.csv", index=False)
    knn_recommendations.to_csv("requirements/job_recommendations_knn.csv", index=False)   
def upload_success(request):
    documents = Document.objects.all()
    latest_document = documents.latest('uploaded_at')
    file_path = latest_document.document.path
    latest_document.read_content()
    return render(request, 'jobs/upload_success.html', {'documents': documents})

def read_csv(request):
    # CSV dosyasının yolu
    file_path = 'requirements/job_recommendations_knn.csv'

    # CSV dosyasını oku
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        data = list(csv_reader)

    # Verileri HTML şablonuna aktar
    context = {'data': data}
    return render(request, 'jobs/template.html', context)

def show_results(request):
    return read_csv(request)