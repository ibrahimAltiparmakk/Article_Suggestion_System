from django.core.management.base import BaseCommand
from datasets import load_dataset
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import fasttext
from ...models import Article

class Command(BaseCommand):
    help = 'Imports articles from the memray/krapivin dataset and computes their vector representations'

    def handle(self, *args, **options):
        dataset = load_dataset("memray/krapivin", split='validation')
        tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
        model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
        ft_model = fasttext.load_model(r'C:\\Users\\sarib\\OneDrive\\Masaüstü\\Makale - Kopya\\cc.en.300.bin')

        for article in dataset:
            title = article['title']
            abstract = article['abstract']
            fulltext = article['fulltext']  # Tam metin
            keywords = article['keywords']

            # FastText vector
            ft_vector = ft_model.get_sentence_vector(abstract).tolist()

            # SciBERT vector
            inputs = tokenizer(abstract, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = model(**inputs)
            scibert_vector = outputs.last_hidden_state.mean(dim=1).squeeze().numpy().tolist()

            # Save to database
            Article.objects.update_or_create(
                title=title,
                defaults={
                    'abstract': abstract,
                    'fulltext': fulltext,
                    'keywords': keywords,
                    'ft_vector': ft_vector,
                    'scibert_vector': scibert_vector
                }
            )
            self.stdout.write(self.style.SUCCESS(f'Imported "{title}"'))
