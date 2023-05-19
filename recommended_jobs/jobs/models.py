from django.db import models

class MyFile(models.Model):
    file = models.FileField(upload_to='uploads/')