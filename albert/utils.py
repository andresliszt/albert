# -*- coding: utf-8 -*-
"""Utilidades para gensim como callbacks, loggers etc."""
from typing import Any
from typing import List
from typing import Optional

from elasticsearch import Elasticsearch
from elastinga.crud import FacebookSearch
from elastinga.crud import InstagramSearch
from elastinga.crud import TwitterSearch
from gensim.models.callbacks import CallbackAny2Vec
from gensim.test.utils import get_tmpfile

from albert import ELASTIC_CONNECTION
from albert.processing import TEXT_PROCESSOR

# TODO: USAR logger


class EpochSaver(CallbackAny2Vec):
    """Callback para guardar el modelo despues de cada epoca."""

    def __init__(self, path_prefix):
        self.path_prefix = path_prefix
        self.epoch = 0

    def on_epoch_end(self, model):
        output_path = get_tmpfile(
            "{}_epoch{}.model".format(self.path_prefix, self.epoch)
        )
        model.save(output_path)
        self.epoch += 1


class EpochLogger(CallbackAny2Vec):
    """Callback para  información sobre el training."""

    def __init__(self):
        self.epoch = 0

    def on_epoch_begin(self, model):
        print("Epoch #{} inicia".format(self.epoch))

    def on_epoch_end(self, model):
        print("Epoch #{} termina".format(self.epoch))
        self.epoch += 1


class Word2VecCorpus:  # pylint: disable=too-few-public-methods
    """Clase para manjear el stream de la data."""

    CHUNK_SIZE = 10000

    def __init__(
        self,
        connection: Elasticsearch,
        index_lists: List[Any],
        **scan_kw
    ):

        self.connection = connection
        self.search_bases = [
            index(self.connection).search_base for index in index_lists
        ]
        self.scan_kw = scan_kw

    def __iter__(self):

        for search in self.search_bases:

            results_generator = search.scan(**self.scan_kw)

            for result in results_generator:

                text = TEXT_PROCESSOR.full_clean(
                    result.to_dict()["text"], joined=False
                )

                yield text


CORPUS = Word2VecCorpus(
    ELASTIC_CONNECTION, index_lists=[TwitterSearch, InstagramSearch]
)
