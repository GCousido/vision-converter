from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from formats.base import DatasetFormat
from formats.neutral_format import NeutralFormat

T = TypeVar("T", bound=DatasetFormat)

class DatasetConverter(ABC, Generic[T]):

    @staticmethod
    @abstractmethod
    def toNeutral(df: T) -> NeutralFormat:
        pass

    @staticmethod
    @abstractmethod
    def fromNeutral(nf: NeutralFormat) -> T:
        pass