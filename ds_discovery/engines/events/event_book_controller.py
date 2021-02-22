import pandas as pd
from typing import Dict
from aistac.properties.decorator_patterns import singleton
from ds_discovery.engines.events.pandas_event_book import PandasEventBook

__author__ = 'Darryl Oatridge'


class EventBookController(object):

    __book_catalog: Dict[str, PandasEventBook] = dict()

    @singleton
    def __new__(cls):
        return super().__new__(cls)

    @property
    def event_book_catalog(self) -> list:
        """Returns the list of event book references in the catalog"""
        return list(self.__book_catalog.keys())

    def is_event_book(self, book_name: str) -> bool:
        """Checks if a book_name reference exists in the book catalog"""
        if book_name in self.event_book_catalog:
            return True
        return False

    def add_event_book(self, book_name: str, reset: bool=None):
        """Returns the event book instance for the given reference name"""
        reset = reset if isinstance(reset, bool) else False
        if not self.is_event_book(book_name=book_name):
            self.__book_catalog.update({book_name: PandasEventBook(book_name=book_name)})
        elif reset:
            self.__book_catalog.get(book_name).reset_state()
        else:
            raise ValueError(f"The book name '{book_name}' already exists in the catalog and does not need to be added")
        return

    def remove_event_books(self, book_name: str) -> bool:
        """removes the event book"""
        book = self.__book_catalog.pop(book_name, None)
        return True if book else False

    def current_state(self, book_name: str, fillna: bool=None) -> [pd.DataFrame, pd.Series]:
        if self.is_event_book(book_name=book_name):
            return self.__book_catalog.get(book_name).current_state(fillna=fillna)
        raise ValueError(f"The book name '{book_name}' can not be found in the catalog")

    def get_modified(self, book_name: str) -> bool:
        """A boolean flag that is raised when a modifier method is called"""
        if self.is_event_book(book_name=book_name):
            return self.__book_catalog.get(book_name).modified
        raise ValueError(f"The book name '{book_name}' can not be found in the catalog")

    def reset_modified(self, book_name: str, modified: bool=None) -> bool:
        """resets the modifier flag to be lowered"""
        if self.is_event_book(book_name=book_name):
            modified = modified if isinstance(modified, bool) else False
            return self.__book_catalog.get(book_name).set_modified(modified=modified)
        raise ValueError(f"The book name '{book_name}' can not be found in the catalog")

    def add_event(self, book_name: str, event: [pd.DataFrame, pd.Series], fix_index: bool=False):
        if self.is_event_book(book_name=book_name):
            fix_index = fix_index if isinstance(fix_index, bool) else False
            return self.__book_catalog.get(book_name).add_event(event=event, fix_index=fix_index)
        raise ValueError(f"The book name '{book_name}' can not be found in the catalog")

    def increment_event(self, book_name: str, event: [pd.DataFrame, pd.Series]):
        if self.is_event_book(book_name=book_name):
            return self.__book_catalog.get(book_name).increment_event(event=event)
        raise ValueError(f"The book name '{book_name}' can not be found in the catalog")

    def decrement_event(self, book_name: str, event: [pd.DataFrame, pd.Series]):
        if self.is_event_book(book_name=book_name):
            return self.__book_catalog.get(book_name).decrement_event(event=event)
        raise ValueError(f"The book name '{book_name}' can not be found in the catalog")
