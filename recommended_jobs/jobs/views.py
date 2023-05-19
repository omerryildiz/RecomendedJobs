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
    uploaded_file = MyFile.objects.latest('id')  # En son yüklenen dosyayı alın
    file_path = uploaded_file.file.path
    csv_file_path = os.path.join(settings.BASE_DIR, file_path)

    # Dosyayı okuyun
    df = pd.read_csv(csv_file_path, encoding="ISO-8859-9")
    with open(file_path, 'r') as file:
        cv_text = file.read()
        
    tfidf_vectorizer = TfidfVectorizer(stop_words="english")
    job_data_tfidf = tfidf_vectorizer.fit_transform(df["skill"])
    cv_tfidf = tfidf_vectorizer.transform([cv_text])
    cosine_similarities = cosine_similarity(cv_tfidf, job_data_tfidf).flatten()
    
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
   