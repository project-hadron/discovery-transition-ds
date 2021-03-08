from __future__ import absolute_import

import copy
from datetime import datetime

from ds_behavioral.simulator.orderbook import Orderbook

__author__ = "Darryl Oatridge"


class OrderbookManager(object):
    """The order-book manager manages a collection of order-books and maintains
    a record of all fulfilled orders
    """

    def __init__(self, exchange: str=None, day_reference: str=None, orderbooks: dict=None):
        """
        initialisation of the orderbook manager
        :param exchange: the reference name given to this orderbook manager, normally the exchange.
                optional else 'Default'
        :param day_reference: a reference to the day this orderbook manager was for, normally a date string
                optional else None
        :param orderbooks: a dictionary of name:value pair of name:start_price or orderbooks to pre-build
            example: {'OAH7.L': 29.575, 'OB65.L': 17.835, 'OCXC.L': 14.9225, 'ODHJ.L': 9.19}
        """
        self._exchange = 'Default'
        self._order_books = {}
        self._order_agent = {}
        self._fulfilled = {}
        if exchange is not None and exchange:
            self._exchange = exchange
        self._day_reference = datetime.now().strftime("%Y-%m-%d") if day_reference is None or not day_reference else day_reference
        if orderbooks is not None and isinstance(orderbooks, dict) and len(orderbooks) > 0:
            self._build_orderbooks(orderbooks)

    @property
    def exchange(self):
        return self._exchange

    @property
    def day_reference(self):
        return self._day_reference

    @property
    def orderbooks(self) -> list:
        """Returns a list of current orderbook names"""
        return [*self._order_books.keys()]

    def get_current_bid(self, orderbook: str) -> float:
        """Returns the current orderbook bid price"""
        return self.get_bid_ask(orderbook)['bid'][-1]

    def get_current_ask(self, orderbook: str) -> float:
        """returns the current order-book ask price"""
        return self.get_bid_ask(orderbook)['ask'][-1]

    def get_current_price(self, orderbook: str):
        """The current price of this book, This is the same as the current ask price"""
        return self.get_current_ask(orderbook)

    def get_current_volume(self, orderbook: str):
        """Returns the current volume of completed orders for this orderbook"""
        total = 0
        for order in self.get_fulfilled(orderbook):
            for key in order['Sell'].keys():
                for e_order in order['Sell'][key]:
                    total += e_order[0]
        return total

    def get_order_book(self, orderbook: str):
        """Returns a formatted order-book"""
        orderbook = self.get_raw_order_book(orderbook)
        sell_side = orderbook['Sell']
        buy_side = orderbook['Buy']
        sell_list = [{'price': key, 'volume': sell_side[key]['total']} for key in sell_side.keys()]
        buy_list = [{'price': key, 'volume': buy_side[key]['total']} for key in buy_side.keys()]
        buy_volume = 0
        sell_volume = 0
        for idx in buy_list:
            buy_volume += idx['volume']
        for idx in sell_list:
            sell_volume += idx['volume']
        return {
            'buy': sorted(buy_list, key= lambda el : -el['price']),     # reverse sorted list
            'buy_volume': buy_volume,
            'sell': sorted(sell_list, key= lambda el: el['price']),      # sorted list
            'sell_volume': sell_volume,
        }

    def get_fulfilled(self, orderbook) -> list:
        """A dictionary of fulfilled orders by orderbook and timestamp of fulfillment"""
        if orderbook is None or orderbook not in self._fulfilled:
            raise ValueError("The orderbook {} does not exist".format(orderbook))
        return copy.deepcopy(self._fulfilled.get(orderbook))

    def get_order_agent(self, orderbook) -> dict:
        """returns a dictionary of keyed on order_id, referencing the order timestamp and the agent id"""
        if orderbook is None or orderbook not in self._order_agent:
            raise ValueError("The orderbook {} does not exist".format(orderbook))
        return copy.deepcopy(self._order_agent.get(orderbook))

    def get_bid_ask(self, orderbook: str):
        """ the bid ask history of a particular orderbook"""
        if orderbook is None or orderbook not in self._order_books:
            raise ValueError("The orderbook {} does not exist".format(orderbook))
        ob = self._order_books.get(orderbook)
        return ob.bid_ask

    def get_raw_order_book(self, orderbook: str):
        if orderbook is None or orderbook not in self._order_books:
            raise ValueError("The orderbook {} does not exist".format(orderbook))
        ob = self._order_books.get(orderbook)
        return ob.book

    def create_orderbook(self, name: str, start_price: float, start_order: int=-1):
        """ creates an order book using the name as a unique reference

        :param name: the name of the orderbook (unique reference name
        :param start_price:
        :param start_order:
        :return:
        """
        if name in self._order_books:
            raise ValueError("The orderbook with name {} already exists".format(name))
        self._order_books.update({name: Orderbook(name, start_price, start_order)})
        self._order_agent.update({name: {}})
        self._fulfilled.update({name: []})

    def remove_orderbook(self, orderbook: str):
        """Remove the specific orderbook from the manager"""
        if orderbook is None or orderbook not in self._order_books:
            return
        del self._order_books[orderbook]
        del self._order_agent[orderbook]
        del self._fulfilled[orderbook]

    def cancel_order(self, orderbook: str, order_id: int, timestamp: float):
        """Remove an order from the order book"""
        if orderbook is None or orderbook not in self._order_books:
            raise ValueError("The orderbook {} does not exist, create orderbook before removing orders".format(orderbook))
        ob = self._order_books.get(orderbook)
        ob.delete_order(order_id)
        if order_id in self._order_agent.get(orderbook):
            self._order_agent.get(orderbook).get(order_id).get('action').update({timestamp: {'status': 'cancel'}})

    def add_market_order(self, orderbook: str, agent_id: str, side: str, size: int, timestamp: float=None) -> tuple:
        _price = float("inf") if side.upper().startswith('B') else float(1)
        return self.add_limit_order(orderbook, agent_id, side, _price, size, timestamp, status='market')

    def add_limit_order(self, orderbook: str, agent_id: str, side: str, price: float, size: int,
                        timestamp: float=None, status: str=None) -> tuple:
        """Adds an order to the specified orderbook as a limit order. If 'Buy'

        :param orderbook: the name of the orderbook
        :param agent_id: the id or the agent makibng the order
        :param side: if the order is Buy or Sell (can use B and S)
        :param price: the price of the order
        :param size: the volume size of the order
        :param timestamp: the timestamp to be associated with the order (optional: if None set to now())
        :param is_market_order: this order is market order not limit (optional: default False)
        :return: tuple of order_id and any fulfilled orders as a dictionary
        """
        if orderbook is None or orderbook not in self._order_books:
            raise ValueError("The orderbook {} does not exist, create orderbook before adding orders".format(orderbook))
        if not side.upper().startswith(('B', 'S')):
            raise ValueError("side must be one of Buy, Sell, B or S")
        if price is None or not price or price < 0:
            raise ValueError("price must greater than 0")
        if size is None or not size or size <= 0 or size > 40000000:
            raise ValueError("size must be agreater than 0", size)
        if timestamp is None:
            timestamp = datetime.now().timestamp()
        if status is None:
            status = 'limit'
        ob = self._order_books.get(orderbook)
        order_id = ob.create_order(side, price, int(size))
        self._order_agent.get(orderbook).update({order_id: {'agent': agent_id, 'action' : {timestamp: {'status': status}}}})
        fulfilled = ob.fulfilled
        if len(fulfilled.get('Buy')) > 0 or len(fulfilled.get('Sell')) > 0:
            fulfilled.update({'timestamp': timestamp})
            self._fulfilled[orderbook] += [fulfilled]
        return order_id, fulfilled

    def _ob(self, orderbook: str) -> Orderbook:
        return self._order_books[orderbook]

    def _build_orderbooks(self, orderbooks: dict):
        for (symbol, price) in orderbooks.items():
            self.create_orderbook(symbol, price)

