#http://127.0.0.1:8000/
from django.urls import path
from . import views
urlpatterns = [
    path("",views.index),
    path("anasayfa",views.index),
    path('upload/', views.upload_document, name='upload_document'),
    path('upload/success/', views.upload_success, name='upload_success'),
    path('results/', views.show_results, name='show_results'),
    #path('dosya/oku/<int:dosya_id>/', file_read, name='dosya-oku')
    #path('upload_document/upload_success/',views.upload_success,name='upload_success')
    
]
