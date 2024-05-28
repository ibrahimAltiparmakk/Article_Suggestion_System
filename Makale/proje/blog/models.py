from django.db import models
from django.contrib.auth.models import User
from django.db.models import JSONField  # Correct import for JSONField

class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    interest1 = models.CharField(max_length=100, blank=True)
    interest2 = models.CharField(max_length=100, blank=True)
    interest3 = models.CharField(max_length=100, blank=True)
    ft_vector = JSONField(default=list)  # FastText vektör temsili
    scibert_vector = JSONField(default=list)  # SciBERT vektör temsili

    def __str__(self):
        return self.user.username

class Article(models.Model):
    title = models.CharField(max_length=255)
    abstract = models.TextField()
    fulltext = models.TextField()  # Tam metni saklamak için sütun
    keywords = models.TextField()
    ft_vector = JSONField(default=list)  # FastText vektörleri
    scibert_vector = JSONField(default=list)  # SciBERT vektörleri

    def __str__(self):
        return self.title
    
class KrapivinArticle(models.Model):
    title = models.CharField(max_length=255)
    abstract = models.TextField()
    keywords = models.TextField()

    def __str__(self):
        return self.title

