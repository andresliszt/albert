# -*- coding: utf-8 -*-
"""Procesamiento de text: stopwords, stem, etc."""
import re
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords as _stopwords
from nltk.stem import SnowballStemmer

from albert import SETTINGS

# TODO: DOCUMENTACIÓN, TYPING

# pylint: disable=anomalous-backslash-in-string
# pylint: disable=no-member


def configure_nltk(
    download_list: List[str], download_path: Union[Path, str]
):
    """Configuración básica para la librería nltk."""
    nltk.data.path.append(download_path)

    for elem in download_list:
        nltk.download(elem, download_path)


configure_nltk(["stopwords", "punkt"], str(SETTINGS.NLTK_DATA))


SPANISH_STEMMER = SnowballStemmer("spanish")
STOPWORDS_NLTK = _stopwords.words("spanish")
CUSTOM_STOPWORDS = [
    "twitter",
    "facebook",
    "instagram",
    "status",
    "see",
    "original",
    "rate",
    "translation",
    "com",
    "pic",
]
STOPWORDS = [*STOPWORDS_NLTK, *CUSTOM_STOPWORDS]


SPECIAL_CHARACTERS = re.compile("[^a-zA-Záéíóúñ]+")
"""Remueve todos los caracteres especiales menos los acentos"""

URL_REGEX = re.compile(
    """((http|https|www.)\:?(\/\/)?)[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*"""
)
"""Remueve las urls presentes en el texto"""

REGEX_DICT = {
    URL_REGEX: "",
    SPECIAL_CHARACTERS: " ",
}


class TextProcessor:
    """Clase para procesar frases/textos de acuerdo a reglas."""

    def __init__(
        self,
        tokenizer: Callable,
        stemmer: Callable,
        stopwords: List[str],
        regex_dict: Dict[Any, str],
    ):

        self.tokenizer = tokenizer
        self.stemmer = stemmer
        self.stopwords = stopwords
        self.regex_dict = regex_dict

    def regex_replacement(self, text: str):

        for pattern, replacement in self.regex_dict.items():
            text = pattern.sub(replacement, text)

        return text

    def full_clean(self, text: Optional[str], joined=True):
        """Aplica todos los métodos de limpieza."""
        if not text:
            return text

        text = self.regex_replacement(text)

        words = self.stem_text(text, joined=joined)

        return words

    def stem_text(self, text: str, joined):
        """Método simple que aplica stem a cada palabra."""

        stopwords = self.stopwords

        words = self.tokenizer(text)

        words = [
            self.stemmer(w) for w in words if w.lower() not in stopwords
        ]

        return " ".join(words) if joined else words


TEXT_PROCESSOR = TextProcessor(
    tokenizer=word_tokenize,
    stemmer=SPANISH_STEMMER.stem,
    stopwords=STOPWORDS,
    regex_dict=REGEX_DICT,
)
