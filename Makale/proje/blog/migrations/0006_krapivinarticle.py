# Generated by Django 4.2.12 on 2024-05-14 11:19

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('blog', '0005_article'),
    ]

    operations = [
        migrations.CreateModel(
            name='KrapivinArticle',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('title', models.CharField(max_length=255)),
                ('abstract', models.TextField()),
                ('keywords', models.TextField()),
            ],
        ),
    ]