import pandas as pd
from ds_discovery.engines.events.event_book_controller import EventBookController
from aistac.handlers.abstract_handlers import AbstractSourceHandler, AbstractPersistHandler
from aistac.handlers.abstract_handlers import ConnectorContract


__author__ = 'Darryl Oatridge'


class EventSourceHandler(AbstractSourceHandler):

    def __init__(self, connector_contract: ConnectorContract):
        """ initialise the Handler passing the connector_contract dictionary"""
        super().__init__(connector_contract)
        self._controller = EventBookController()
        if connector_contract.schema != 'eb' or len(connector_contract.netloc) == 0:
            raise ValueError(f"The connector contract uri must be in  the format 'eb://<book_name>' as a minimum")
        self._book_name = connector_contract.netloc
        if not self._controller.is_event_book(book_name=self._book_name):
            self._controller.add_event_book(book_name=self._book_name)

    def supported_types(self) -> list:
        return ['pd.DataFrame']

    def exists(self) -> bool:
        return self._controller.is_event_book(book_name=self._book_name)

    def has_changed(self) -> bool:
        return self._controller.get_modified(book_name=self._book_name)

    def reset_changed(self, changed: bool=False):
        self._controller.reset_modified(book_name=self._book_name, modified=changed)
        return

    def load_canonical(self, **kwargs) -> pd.DataFrame:
        return self._controller.current_state(book_name=self._book_name)


class EventPersistHandler(EventSourceHandler, AbstractPersistHandler):

    def persist_canonical(self, canonical: pd.DataFrame, reset_state: bool=None, **kwargs) -> bool:
        """ persists the canonical into the event book extending or replacing the current state

        :param canonical: the canonical to persist to the event book
        :param reset_state: True - resets the event book (Default)
                            False - merges the canonical to the current state based on their index
        """
        reset_state = reset_state if isinstance(reset_state, bool) else True
        if reset_state:
            self._controller.add_event_book(book_name=self._book_name, reset=True)
        self._controller.add_event(book_name=self._book_name, event=canonical, fix_index=False)
        return True

    def remove_canonical(self, **kwargs) -> bool:
        return self._controller.remove_event_books(book_name=self._book_name)

    def backup_canonical(self, canonical: pd.DataFrame, uri: str, reset_state: bool=None, **kwargs) -> bool:
        """ persists the canonical into the event book extending or replacing the current state

        :param canonical: the canonical to persist to the event book
        :param uri: the uri of the event book
        :param reset_state: True - resets the event book (Default)
                            False - merges the canonical to the current state based on their index
        """
        _schema, _book_name, _ = ConnectorContract.parse_address_elements(uri=uri)
        if _schema != 'eb' or len(_book_name) == 0:
            raise ValueError(f"The connector contract uri must be in  the format 'eb://<book_name>' as a minimum")
        self._controller.add_event_book(book_name=_book_name, reset=True)
        self._controller.add_event(book_name=self._book_name, event=canonical, fix_index=False)
        return True

