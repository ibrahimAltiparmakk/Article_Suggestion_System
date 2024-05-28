from django.shortcuts import render ,redirect
from django.http.response import HttpResponse
from .forms import RegisterForm
from .forms import LoginForm
from django.contrib.auth import authenticate, login ,logout
from django.http import HttpResponseRedirect
from .forms import ProfileUpdateForm
from .forms import SearchForm
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModel
import fasttext
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .models import Article , Profile
from django.contrib.auth.decorators import login_required
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import string
from .models import KrapivinArticle
from django.shortcuts import get_object_or_404
from django.urls import reverse


ft_model = fasttext.load_model(r'C:\\Users\\sarib\\OneDrive\\Masaüstü\\Makale\\cc.en.300.bin')  # Model yolu doğru olmalı

# Create your views here.

def index(request):
    if request.method == 'POST':
        form = RegisterForm(request.POST)
        if form.is_valid():
            user = form.save()  # Kullanıcıyı veritabanına kaydeder
         #   login(request, user)  # Kullanıcıyı otomatik olarak giriş yapar
            return redirect('anasayfa')  # Başarılı kayıt sonrası anasayfaya yönlendir
    else: 
        form = RegisterForm()
    return render(request, 'blog/index.html', {'form': form})

def blogs(request):
    if request.method == 'POST':
        form = LoginForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']
            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('anasayfa')  # Başarılı giriş sonrası yönlendirme
            else:
                return render(request, 'blog/blogs.html', {'form': form, 'error': 'Hatalı kullanıcı adı veya şifre'})
    else:
        form = LoginForm()
    return render(request, 'blog/blogs.html', {'form': form})

dataset = load_dataset("memray/krapivin", split='validation')

@login_required
def anasayfa(request):
    form = SearchForm()
    results = []
    if request.method == 'POST':
        form = SearchForm(request.POST)
        if form.is_valid():
            query = form.cleaned_data['query'].lower()
            results = [(idx, item['title'], item['keywords']) for idx, item in enumerate(dataset) if query in item['keywords'].lower()]
            return render(request, 'blog/anasayfa.html', {'form': form, 'results': results})
    else:
        return render(request, 'blog/anasayfa.html', {'form': form})



def logout_view(request):
    logout(request)
    return redirect('Kayıt')  # Kullanıcıyı anasayfaya yönlendir

@login_required
def profil(request):
    user = request.user
    interests = [user.profile.interest1, user.profile.interest2, user.profile.interest3]
    interests = [interest for interest in interests if interest]  # Boş ilgi alanlarını filtrele
    
    # FastText vektör ortalamasını hesapla
    ft_vectors = [ft_model.get_sentence_vector(interest) for interest in interests]
    ft_average_vector = np.mean(ft_vectors, axis=0) if ft_vectors else np.zeros(ft_model.get_dimension())

    # SciBERT vektör ortalamasını hesapla
    scibert_tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    scibert_model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
    encoded_input = scibert_tokenizer(interests, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = scibert_model(**encoded_input)
    scibert_vectors = model_output.last_hidden_state.mean(dim=1)
    scibert_average_vector = scibert_vectors.mean(dim=0).numpy()

    return render(request, 'blog/profil.html', {
        'user': user,
        'ft_average_vector': ft_average_vector,
        'scibert_average_vector': scibert_average_vector
    })




def get_ft_vector_representation(text):
    cleaned_text = clean_text(text)
    vector = ft_model.get_sentence_vector(cleaned_text)
    return vector

def get_scibert_vector_representation(text):
    scibert_tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    scibert_model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')

    inputs = scibert_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = scibert_model(**inputs)
    vector = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return vector

def calculate_ft_vector_average(interests):
    vectors = [get_ft_vector_representation(interest) for interest in interests if interest]
    if vectors:
        return np.mean(vectors, axis=0)
    return np.zeros(ft_model.get_dimension())  # Vektörler boşsa sıfır vektör dön

def calculate_scibert_vector_average(interests):
    vectors = [get_scibert_vector_representation(interest) for interest in interests if interest]
    if vectors:
        return np.mean(vectors, axis=0)
    return np.zeros(scibert_model.config.hidden_size)  # Vektörler boşsa sıfır vektör dön



def get_vector_representation(text):
    cleaned_text = clean_text(text)
    vector = ft_model.get_sentence_vector(cleaned_text)
    return vector



def update_profile(request):
    if not request.user.is_authenticated:
        return redirect('login')  # Kullanıcı giriş yapmamışsa giriş sayfasına yönlendir

    profile = request.user.profile
    if request.method == 'POST':
        form = ProfileUpdateForm(request.POST, instance=profile)
        if form.is_valid():
            form.save()
            return redirect('profil')  # Güncelleme başarılı ise profil sayfasına yönlendir
    else:
        form = ProfileUpdateForm(instance=profile)

    return render(request, 'blog/update_profile.html', {'form': form})


def article_detail(request, id):
    article = get_object_or_404(Article, id=id)
    cleaned_text = clean_text(article.fulltext)  # Tam metni temizle
    return render(request, 'blog/article_detail.html', {'article': article, 'cleaned_text': cleaned_text})


nlp = spacy.load('en_core_web_sm')  # Dil modelini yükle

def clean_text(text):
    doc = nlp(text)
    cleaned_text = " ".join(token.text for token in doc if not token.is_stop and not token.is_punct)
    return cleaned_text


# FastText modelini yükle

def preprocess_text(text):
    doc = nlp(text)
    return ' '.join([token.text for token in doc if not token.is_stop and not token.is_punct])

# FastText ve SciBERT ile vektörlerin hesaplanması
def get_vectors(texts):
    ft_vectors = [ft_model.get_sentence_vector(text) for text in texts]
    scibert_vectors = []
    for text in texts:
        inputs = scibert_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = scibert_model(**inputs)
        scibert_vectors.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())
    return ft_vectors, scibert_vectors

def calculate_interest_vector(interests):
    vectors = [ft_model.get_word_vector(interest) for interest in interests if interest]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(ft_model.get_dimension())  # Model boyutunu dinamik olarak al

scibert_model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
scibert_tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')

def scibert_encode(texts):
    if texts:  # Liste boş değilse işlem yap
        encoded_input = scibert_tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = scibert_model(**encoded_input)
        return model_output.pooler_output.squeeze().numpy()
    else:
        return np.array([])  # Boş bir numpy dizisi döndür

# recommend_articles görünümünü ham vektör verileri içerecek şekilde değiştir
@login_required
def recommend_articles(request):
    user = request.user
    profile = Profile.objects.get(user=user)

    interests = [profile.interest1, profile.interest2, profile.interest3]
    interests = [interest for interest in interests if interest]

    user_ft_vector = np.mean([np.array(get_ft_vector_representation(interest)) for interest in interests], axis=0)
    user_scibert_vector = np.mean([np.array(get_scibert_vector_representation(interest)) for interest in interests], axis=0)

    articles = Article.objects.all()
    ft_vectors = [np.array(article.ft_vector) for article in articles]
    scibert_vectors = [np.array(article.scibert_vector) for article in articles]

    ft_similarities = cosine_similarity([user_ft_vector], ft_vectors)[0]
    scibert_similarities = cosine_similarity([user_scibert_vector], scibert_vectors)[0]

    ft_sorted_articles = sorted(zip(articles, ft_similarities), key=lambda x: x[1], reverse=True)[:5]
    scibert_sorted_articles = sorted(zip(articles, scibert_similarities), key=lambda x: x[1], reverse=True)[:5]

    ft_recommended_articles = [{'article': article, 'similarity': similarity} for article, similarity in ft_sorted_articles]
    scibert_recommended_articles = [{'article': article, 'similarity': similarity} for article, similarity in scibert_sorted_articles]

    return render(request, 'blog/oneri.html', {
        'ft_recommended_articles': ft_recommended_articles,
        'scibert_recommended_articles': scibert_recommended_articles
    })






# FastText ve SCIBERT modellerini kullanarak kullanıcı ilgi alanlarından vektörler üretme ve kaydetme
@login_required
def update_user_interest_vectors(request):
    profile = request.user.profile
    interests = [profile.interest1, profile.interest2, profile.interest3]
    
    # FastText vektörleri hesaplama ve ortalamasını alma
    ft_vectors = [ft_model.get_word_vector(interest) for interest in interests if interest]
    if ft_vectors:
        ft_average_vector = np.mean(ft_vectors, axis=0)
    else:
        ft_average_vector = np.zeros(ft_model.get_dimension())

    # SCIBERT vektörleri hesaplama ve ortalamasını alma
    scibert_vectors = scibert_encode(interests)
    if len(scibert_vectors) > 0:
        scibert_average_vector = np.mean(scibert_vectors, axis=0)
    else:
        scibert_average_vector = np.zeros(scibert_model.config.hidden_size)

    # Kullanıcı profilinde vektörleri güncelleme
    profile.ft_vector = ft_average_vector
    profile.scibert_vector = scibert_average_vector
    profile.save()

    return render(request, 'profile_updated.html', {'profile': profile})

# views.py dosyasında get_article_vectors fonksiyonunu ham verileri dönecek şekilde değiştir
def get_article_vectors():
    articles = Article.objects.all()
    ft_vectors = {}
    scibert_vectors = {}
    ft_raw_vectors = {}
    scibert_raw_vectors = {}
    
    for article in articles:
        # FastText
        abstract_words = article.abstract.split()
        ft_abstract_vectors = [ft_model.get_word_vector(word) for word in abstract_words]
        ft_vector = np.mean(ft_abstract_vectors, axis=0) if ft_abstract_vectors else np.zeros(ft_model.get_dimension())
        ft_vectors[article.id] = ft_vector
        ft_raw_vectors[article.id] = ft_vector.tolist()  # List olarak kaydetmek için
        
        # SciBERT
        scibert_abstract_vector = scibert_encode([article.abstract])
        scibert_vectors[article.id] = scibert_abstract_vector if scibert_abstract_vector.size else np.zeros(scibert_model.config.hidden_size)
        scibert_raw_vectors[article.id] = scibert_abstract_vector.tolist() if scibert_abstract_vector.size else np.zeros(scibert_model.config.hidden_size).tolist()
    
    return ft_vectors, scibert_vectors, ft_raw_vectors, scibert_raw_vectors




# Kullanıcı ilgi alanları ve makale vektörleri arasındaki benzerlikleri hesaplama ve en iyi önerileri alma
def recommend_articles_to_user(request):
    profile = request.user.profile
    ft_user_vector = profile.ft_vector
    scibert_user_vector = profile.scibert_vector
    
    ft_vectors, scibert_vectors = get_article_vectors()
    
    # FastText için benzerlik hesaplama
    ft_similarities = {article_id: cosine_similarity([ft_user_vector], [vec])[0][0] for article_id, vec in ft_vectors.items()}
    ft_top_articles = sorted(ft_similarities, key=ft_similarities.get, reverse=True)[:5]

    # SCIBERT için benzerlik hesaplama
    scibert_similarities = {article_id: cosine_similarity([scibert_user_vector], [vec])[0][0] for article_id, vec in scibert_vectors.items()}
    scibert_top_articles = sorted(scibert_similarities, key=scibert_similarities.get, reverse=True)[:5]

    ft_recommended_articles = Article.objects.filter(id__in=ft_top_articles)
    scibert_recommended_articles = Article.objects.filter(id__in=scibert_top_articles)

    return render(request, 'oneri.html', {
        'ft_recommended_articles': ft_recommended_articles,
        'scibert_recommended_articles': scibert_recommended_articles
    })



@login_required
def recommend_interest_based_articles(request):
    user = request.user
    interests = [user.profile.interest1, user.profile.interest2, user.profile.interest3]
    interests = [interest for interest in interests if interest]  # Boş ilgi alanlarını filtrele

    # FastText ve SciBERT vektörleri
    combined_vectors = [
        (interest, get_ft_vector_representation(interest), get_scibert_vector_representation(interest))
        for interest in interests
    ]

    return render(request, 'blog/oneri.html', {
        'combined_vectors': combined_vectors
    })


def get_recommended_articles(ft_user_vector):
    dataset = load_dataset("memray/krapivin", split='validation')
    article_vectors = [(article['title'], ft_model.get_sentence_vector(article['abstract'])) for article in dataset]

    # Cosine Similarity hesapla ve en benzer 5 makaleyi bul
    similarities = [(title, cosine_similarity([ft_user_vector], [vector]).flatten()[0]) for title, vector in article_vectors]
    sorted_articles = sorted(similarities, key=lambda x: x[1], reverse=True)[:5]
    
    return sorted_articles


@login_required
def recommend_sciBERT_articles(request):
    user = request.user
    interests = [user.profile.interest1, user.profile.interest2, user.profile.interest3]
    interests = [interest for interest in interests if interest]
    
    scibert_average_vector = calculate_scibert_vector_average(interests)
    recommended_articles = get_sciBERT_recommendations(scibert_average_vector)
    
    return render(request, 'blog/oneri.html', {'recommended_articles': recommended_articles})



def get_sciBERT_recommendations(scibert_average_vector):
    dataset = load_dataset("memray/krapivin", split='validation')
    articles = []
    for article in dataset:
        encoded_input = scibert_tokenizer(article['abstract'], return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            model_output = scibert_model(**encoded_input)
        article_vector = model_output.last_hidden_state.mean(dim=1).squeeze().numpy()
        similarity = cosine_similarity([scibert_average_vector], [article_vector])
        articles.append((article['title'], similarity[0][0]))
    articles.sort(key=lambda x: x[1], reverse=True)
    return articles[:5]



@login_required
def update_interest_vectors(request):
    if request.method == 'POST':
        article_ids = request.POST.getlist('article_ids')
        articles = Article.objects.filter(id__in=article_ids)
        profile = request.user.profile
        current_vectors = [np.array(profile.ft_vector)] if profile.ft_vector else []

        for article in articles:
            if article.ft_vector:
                current_vectors.append(np.array(article.ft_vector))

        if current_vectors:
            new_average_vector = np.mean(current_vectors, axis=0)
        else:
            new_average_vector = np.zeros(ft_model.get_dimension())

        profile.ft_vector = new_average_vector.tolist()
        profile.save()

        # İlgi alanlarına dayalı öneri listesini yenilemek için öneri fonksiyonunu tetikle
        return HttpResponseRedirect(reverse('recommend_articles'))

    return redirect('anasayfa')

