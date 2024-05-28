from django.urls import path
from . import views
from .views import update_profile
from .views import recommend_articles
from .views import recommend_interest_based_articles

# http://127.0.0.1:8000/

urlpatterns = [
     path("",views.index,name="Kayıt"),
     path("index",views.index,),
     path("blogs",views.blogs,name="Giriş"),
     path("anasayfa",views.anasayfa,name="anasayfa"),
     path("logout/",views.logout_view, name='logout'),
     path("profil",views.profil,name="profil"),
     path('update-profile/', update_profile, name='update_profile'),
     path('article/<int:id>/', views.article_detail, name='article_detail'),  # Yeni eklenen detay sayfası
     path('recommendations/', views.recommend_articles, name='recommendations'),
     path('recommend-interest-based/', recommend_interest_based_articles, name='recommend_interest_based'),
     path('update-interest-vectors/', views.update_interest_vectors, name='update_interest_vectors'),
     path('recommend-articles/', views.recommend_articles, name='recommend_articles'),

]