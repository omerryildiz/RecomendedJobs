from django.db import models


#import numpy as np
#import seaborn as sns
import os


class FilesAdmin(models.Model):
    description= models.CharField(max_length=500,blank=True)
    document = models.FileField(upload_to='documents/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    def __str__(self):
        return self.description

class Document(models.Model):
    description = models.CharField(max_length=255, blank=True)
    document = models.FileField(upload_to='documents/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    def read_content(self):
        with open(self.document.path, 'r',encoding="utf-8") as file:
            content = file.read()
            
        return content
        
  