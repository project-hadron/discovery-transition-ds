__author__ = 'Darryl Oatridge'

import socketserver
from typing import Any

from aistac import ConnectorContract


class PublicationHandler(object):

    def publish_canonical(self, canonical: Any, **kwargs) -> bool:
        pass

    def remove_subscription(self, **kwargs) -> bool:
        pass


class SubscriptionHandler(object):
    _subscriptions: dict

    def __init__(self, host: str, port: int):
        self._host = host if isinstance(host, str) else "localhost"
        self._port = port if isinstance(port, int) else 6473

    @property
    def host(self) -> str:
        return self._host

    @property
    def port(self) -> int:
        return self._port

    def startup(self, poll_interval: int):
        server = SocketServer((self._host, self._port), SocketHandler)
        server.serve_forever(poll_interval)


class SocketHandler(socketserver.StreamRequestHandler):

    def handler(self):
        pass


class SocketServer(socketserver.ThreadingMixIn, socketserver.ThreadingTCPServer):
    pass
