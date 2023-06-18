from django.urls import path
from .views import upload_file,index,about,process_file

urlpatterns = [
    path('upload/', upload_file, name='upload_file'),
    path('', index, name='index'),
    path('about/', about, name='about'),
    
    path('process/', process_file, name='process_file'),
]