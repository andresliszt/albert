# -*- coding: utf-8 -*-
"""Manejo de Exceptions y Errores."""

import abc
import functools
from contextlib import contextmanager
from types import ModuleType
from typing import Any
from typing import Callable
from typing import Generator
from typing import Type
from typing import Union


class AlbertErrorMixin(abc.ABC, BaseException):
    """Base class for custom errors and exceptions.

    Example:

        >>> class MyError(AlbertErrorMixin):
                msg_template = "Value ``{value}`` could not be found"
        >>> raise MyError(value="can't touch this")
        (...)
        MyError: Value `can't touch this` could not be found

    """

    @property
    @abc.abstractmethod
    def msg_template(self) -> str:
        """Un template para imprimir cuando una excepción es levantada.

        Ejemplo:
            "Value ``{value}`` no se encuentra "

        """

    def __init__(self, **ctx: Any) -> None:
        self.ctx = ctx
        super().__init__()

    def __str__(self) -> str:
        txt = self.msg_template
        for name, value in self.ctx.items():
            txt = txt.replace("{" + name + "}", str(value))
        txt = txt.replace("`{", "").replace("}`", "")
        return txt


@contextmanager
def change_exception(
    raise_exc: Union[AlbertErrorMixin, Type[AlbertErrorMixin]],
    *except_types: Type[BaseException],
) -> Generator[None, None, None]:
    """Context Manager para remplazar excepciones por propias.

    Ver:
        :func:`pydantic.utils.change_exception`

    """
    try:
        yield
    except except_types as exception:
        raise raise_exc from exception  # type: ignore


class EnvVarNotFound(AlbertErrorMixin, NameError):
    """Levantar cuando no se encuentro una var de entorno."""

    msg_template = (
        "La variable de entorno `{env_var}` no puede ser encontrada"
    )


class SocialIndexNotExists(AlbertErrorMixin, NameError):
    """Levantar cuando los indices de redes sociales no existan."""

    msg_template = "Uno o más índices de redes sociales no existen"
