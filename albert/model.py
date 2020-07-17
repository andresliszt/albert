# -*- coding: utf-8 -*-
"""Word embedding para data extraida de redes sociales."""

import multiprocessing
from pathlib import Path
from typing import Iterable

from gensim.models import Word2Vec
from gensim.models.phrases import Phraser
from gensim.models.phrases import Phrases

from albert import SETTINGS
from albert.utils import CORPUS

# TODO: Typing
# pylint: disable=no-member

W2VEC_CONFIG = {
    "min_count": 20,
    "window": 2,
    "size": 300,
    "sample": 6e-5,
    "alpha": 0.03,
    "min_alpha": 0.0007,
    "negative": 8,
    "workers": multiprocessing.cpu_count() - 1,
}

W2VEC_MODEL = Word2Vec(**W2VEC_CONFIG)

# Entrenamos el modelo con las tablas ya sanitizadas
# TODO : Generalizar con una función anidada los N-grams


def make_trigrams(
    sentences: Iterable, save_model_path: Path, **phrases_kw
):
    """Entrena modelo de bigramas de gensim."""
    bigram = Phrases(sentences, **phrases_kw)
    bigram_phraser = Phraser(bigram)
    tokens = bigram_phraser[sentences]
    trigram = Phrases(tokens, delimiter=b" ")
    trigram_phraser = Phraser(trigram)
    trigram_phraser.save(str(save_model_path))


def train_model_w2vec(
    sentences: Iterable,
    model: Word2Vec,
    trigram_model_path: Path,
    save_model_path: Path,
    epochs: int,
):
    """Entrena Word2Vec."""

    # TODO: Agregar kwargs para pasarlo a los metodos de gensim

    trigram = Phraser.load(str(trigram_model_path))
    sentences = trigram[sentences]
    model.build_vocab(sentences, progress_per=10000)
    model.train(
        sentences, total_examples=model.corpus_count, epochs=epochs
    )
    model.init_sims(replace=True)
    model.save(str(save_model_path))


def make_social_media_trigrams(
    sentences=CORPUS, save_model_path=SETTINGS.TRIGRAM_PATH, **kwargs
):
    """Trigramas usando el iterador para redes sociales `CORPUS`"""

    make_trigrams(sentences, save_model_path, **kwargs)


def train_social_media_w2vec(
    sentences=CORPUS,
    model=W2VEC_MODEL,
    trigram_model_path=SETTINGS.TRIGRAM_PATH,
    save_model_path=SETTINGS.WORD2VEC_PATH,
    epochs=20,
):
    """Entrena modelo W2vec con configuración en `CORPUS`"""
    train_model_w2vec(
        sentences, model, trigram_model_path, save_model_path, epochs
    )


if __name__ == "__main__":
    train_social_media_w2vec()
