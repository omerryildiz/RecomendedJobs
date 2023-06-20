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


#***********************
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn.manifold import TSNE
from bokeh.plotting import figure, show, output_notebook
from bokeh.models import HoverTool, ColumnDataSource, CategoricalColorMapper
from bokeh.palettes import Spectral10
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

def index(request):
    return render(request,'index.html')
def about(request):
     return render(request,'about.html')
def upload_file(request):
    if request.method == 'POST':
        form = MyFileForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('process_file')
          
    else:
        form = MyFileForm()
    return render(request, 'upload_file.html', {'form': form})

def process_file(request):
    
    uploaded_file = MyFile.objects.latest('id')  # En son yüklenen dosyayı alın
    file_path = uploaded_file.file.path
    file_path_csv = os.path.join(settings.BASE_DIR, 'jobs/documents', 'job_ads_with_labels.csv')
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
    df['skill'].fillna('unknown', inplace=True)
    df['label'].fillna('unknown', inplace=True)
    
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(df['skill'])
    y = df['label'] 
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = KNeighborsClassifier(n_neighbors=5)  # n_neighbors parametresini ihtiyaca göre ayarlayabilirsiniz
    model.fit(X_train, y_train)

    cv_vector = vectorizer.transform([cv_text])
    predicted_class = model.predict(cv_vector)
    filtered_df = df[df['label'] == predicted_class[0]]
    filtered_X = vectorizer.transform(filtered_df['skill'])
    cos_sim = cosine_similarity(cv_vector, filtered_X).flatten()
    # En iyi 10 iş ilanını getir
    top_10_indices = cos_sim.argsort()[-10:][::-1]
    top_10_jobs = filtered_df.iloc[top_10_indices]
    
    top_10_jobs.to_csv('documents/recommended_jobs.csv', index=False)
    file_path_csv_rec = 'documents/recommended_jobs.csv'
    df = pd.read_csv(file_path_csv_rec, encoding="utf-8")
    df['skill'].fillna('unknown', inplace=True)
    df['label'].fillna('unknown', inplace=True)

    columns = ['title', 'description', 'skill', 'link', 'img_url', 'label']
    data = df[columns].values.tolist()

    return render(request, 'process_file.html', {'data': data})
#     tsne = TSNE(n_components=2, random_state=0)
#     X_2d = tsne.fit_transform(X.toarray())
    
#     df_viz = pd.DataFrame(columns=['x', 'y', 'description'])
#     df_viz['x'], df_viz['y'], df_viz['description'] = X_2d[:, 0], X_2d[:, 1], df['description']
#     df_viz['color'] = ['red' if i in top_10_indices else 'blue' for i in range(len(df_viz))]
#     source = ColumnDataSource(data=df_viz)
#     output_notebook()  # output bokeh plots to notebook
#     hover_tsne = HoverTool(tooltips=[("description", "@description")])
#     tools_tsne = ['pan', 'wheel_zoom', 'reset', hover_tsne]
#     plot_tsne = figure(width=700, height=700, tools=tools_tsne, title='Jobs')
#     plot_tsne.circle('x', 'y', size=7, source=source, color='color', name="df_viz", alpha=0.8)
    
#     show(plot_tsne)
    
    
    
    
    
    
#     tfidf_vectorizer = TfidfVectorizer(stop_words="english")
#     job_data_tfidf = tfidf_vectorizer.fit_transform(df["skill"])
#     cv_tfidf = tfidf_vectorizer.transform([cv_text])
#     cosine_similarities = cosine_similarity(cv_tfidf, job_data_tfidf).flatten()

#     def get_recommendation(top_indices, df_all, scores):
#         recommendation = pd.DataFrame(columns=['JobID', 'title', 'description', 'skill', 'score', 'link'])
#         count = 0
#         for i in top_indices:
#             recommendation.at[count, 'JobID'] = df_all.index[i]
#             recommendation.at[count, 'title'] = df_all['title'][i]
#             recommendation.at[count, 'description'] = df_all['description'][i]
#             recommendation.at[count, 'skill'] = df_all['skill'][i]
#             recommendation.at[count, 'score'] = scores[count]
#             recommendation.at[count, 'link'] = df_all['link'][i]
#             count += 1
#         return recommendation

#     def KNN(scraped_data, cv):
#         tfidf_vectorizer = TfidfVectorizer(stop_words='english')

#         n_neighbors = 100
#         KNN = NearestNeighbors(n_neighbors=n_neighbors, p=2)
#         KNN.fit(tfidf_vectorizer.fit_transform(scraped_data))
#         NNs = KNN.kneighbors(tfidf_vectorizer.transform(cv))
#         top = NNs[1][0][1:]
#         index_score = NNs[0][0][1:]

#         knn = get_recommendation(top, df, index_score)
#         return knn

#     knn_recommendations = KNN(df['skill'], [cv_text])
#     top_indices = sorted(range(len(cosine_similarities)), key=lambda i: cosine_similarities[i], reverse=True)[:100]
#     list_scores = [cosine_similarities[i] for i in top_indices]
#     recommendations = get_recommendation(top_indices, df, list_scores)
    
#     return render(request, 'jobs/process_file.html', {
#         'file_content': cv_text,
#         'knn_recommendations': knn_recommendations,
#         'recommendations': recommendations
#     })