from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .models import Profile

class RegisterForm(UserCreationForm):
    interest1 = forms.CharField(label='İlgi Alanı 1', max_length=100, required=False)
    interest2 = forms.CharField(label='İlgi Alanı 2', max_length=100, required=False)
    interest3 = forms.CharField(label='İlgi Alanı 3', max_length=100, required=False)

    class Meta:
        model = User
        fields = ['username', 'password1', 'password2', 'interest1', 'interest2', 'interest3']

    def __init__(self, *args, **kwargs):
        super(RegisterForm, self).__init__(*args, **kwargs)
        self.fields['username'].label = 'Kullanıcı Adı'
        self.fields['password1'].label = 'Şifre'
        self.fields['password2'].label = 'Şifre Tekrar'

    def save(self, commit=True):
        user = super(RegisterForm, self).save(commit=False)
        if commit:
            user.save()
            user.profile.interest1 = self.cleaned_data['interest1']
            user.profile.interest2 = self.cleaned_data['interest2']
            user.profile.interest3 = self.cleaned_data['interest3']
            user.profile.save()
        return user

class LoginForm(forms.Form):
    username = forms.CharField(label='Kullanıcı Adı ', max_length=64)
    password = forms.CharField(label='Şifre ', widget=forms.PasswordInput)

class ProfileUpdateForm(forms.ModelForm):
    class Meta:
        model = Profile
        fields = ['interest1', 'interest2', 'interest3']


class SearchForm(forms.Form):
    query = forms.CharField(max_length=100, label='Ara')