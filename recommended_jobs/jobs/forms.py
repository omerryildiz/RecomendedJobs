from django import forms
from .models import MyFile

class MyFileForm(forms.ModelForm):
    
    class Meta:
        model = MyFile
        fields = ("file",)
