# -*- coding: utf-8 -*-
"""Inicialización del paquete."""

from albert._logging import configure_logging
from albert.exc import SocialIndexNotExists
from albert.settings import init_project_settings

SETTINGS = init_project_settings()


logger = configure_logging("albert", SETTINGS, kidnap_loggers=True)


def get_connection():
    """Retorna conexión a elasticsearch.

    Raises:
        SocialIndexNotExists: Si los índices no existen

    """
    import os

    from elastinga.connection import ElasticSearchConnection

    connection_class = ElasticSearchConnection()

    if (
        not connection_class.verify_index_exist(
            os.environ.get("ELASTINGA_INSTAGRAM_INDEX_NAME")
        )
        or not connection_class.verify_index_exist(
            os.environ.get("ELASTINGA_FACEBOOK_INDEX_NAME")
        )
        or not connection_class.verify_index_exist(
            os.environ.get("ELASTINGA_TWITTER_INDEX_NAME")
        )
    ):
        raise SocialIndexNotExists

    return connection_class.connection


ELASTIC_CONNECTION = get_connection()
"""Conexión a elasticsearch. Notar que no es una conexión activa"""
