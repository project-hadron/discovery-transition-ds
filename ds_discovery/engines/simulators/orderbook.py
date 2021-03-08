import copy
from heapq import heappush, heappop

__author__ = "Darryl Oatridge"


class Orderbook(object):

    BUY = 'Buy'
    SELL = 'Sell'

    def __init__(self, name: str, start_price: float, start_order: int=-1):
        self._name = name
        self._order_id = start_order
        self._book = {'Sell': {}, 'Buy': {}}
        self._prices = {'Sell': [], 'Buy': []}
        self._fulfilled = {'Sell': {}, 'Buy': {}}
        self._audit = {'ask': [start_price], 'bid': [start_price]}

    @property
    def book(self) -> dict:
        return copy.deepcopy(self._book)

    @property
    def bid_ask(self) -> dict:
        return copy.deepcopy(self._audit)

    @property
    def fulfilled(self) -> dict:
        return copy.deepcopy(self._fulfilled)

    @property
    def prices(self) -> dict:
        return copy.deepcopy(self._prices)

    @property
    def name(self):
        return self._name

    def create_order(self, side: str, price: float, size: int) -> int:
        _order_id = self._add_order(side, price, size)
        self._prepare_match()
        self._reconcile_orders()
        self._set_audit()
        return _order_id

    def delete_order(self, order_id: int):
        # clean the book
        for side in self._book.keys():
            _remove_list = []
            for price, values in self._book.get(side).items():
                if order_id in values.get('ids'):
                    size = values.get('orders').get(order_id)
                    values.get('ids').remove(order_id)
                    del values['orders'][order_id]
                    values['total'] -= size
                if values['total'] < 1:
                    _remove_list += [price]
            for p in _remove_list:
                del self._book[side][p]
                self._prices[side].remove(p if side == Orderbook.SELL else -p)
        self._prepare_match()
        self._reconcile_orders()
        self._set_audit()

    def _add_order(self, side: str, price: float, size: int) -> int:
        if not side.upper().startswith(('B', 'S')):
            raise ValueError('side must be one of Buy, Sell, B or S')
        price = float(price)
        self._order_id += 1
        is_sell = True if side.upper().startswith('S') else False
        _book_name = 'Sell' if is_sell else 'Buy'
        if price not in self._book[_book_name]:
            self._book[_book_name][price] = {'ids': [], 'orders': {}, 'total': 0}
            heappush(self._prices[_book_name], price if is_sell else -price)
        self._book[_book_name][price]['ids'] += [self._order_id]
        self._book[_book_name][price]['orders'].update({self._order_id: size})
        self._book[_book_name][price]['total'] += size
        return self._order_id

    def _prepare_match(self):
        self._fulfilled['Sell'] = {}
        self._fulfilled['Buy'] = {}

    def _reconcile_orders(self):
        while len(self._prices['Buy']) > 0 and self._book['Buy'][-self._prices['Buy'][0]]['total'] == 0:
            del self._book['Buy'][-self._prices['Buy'][0]]
            heappop(self._prices['Buy'])
        while len(self._prices['Sell']) > 0 and self._book['Sell'][self._prices['Sell'][0]]['total'] == 0:
            del self._book['Sell'][self._prices['Sell'][0]]
            heappop(self._prices['Sell'])
        if len(self._prices['Buy']) == 0 or len(self._prices['Sell']) == 0 \
                or -self._prices['Buy'][0] < self._prices['Sell'][0]:
            return
        bb = -self._prices['Buy'][0]
        bs = self._prices['Sell'][0]
        # covers off a sell market order (price = 1)
        _settle_price = bs if bs > 1 else bb
        # covers off both a buy and a sell market order
        if bb == float('inf'):
            _settle_price = self._audit['bid'][-1]
        price = int(100 * _settle_price) / 100.0
        if len(self._book['Buy'][bb]['ids']) != 0:
            buy_id = self._book['Buy'][bb]['ids'][0]
        else:
            return
        buy_size = self._book['Buy'][bb]['orders'][buy_id]
        if len(self._book['Sell'][bs]['ids']) != 0:
            sell_id = self._book['Sell'][bs]['ids'][0]
        else:
            return
        sell_size = self._book['Sell'][bs]['orders'][sell_id]
        # return the smallest
        filled = min(buy_size, sell_size)
        if buy_id not in self._fulfilled['Buy']:
            self._fulfilled['Buy'][buy_id] = []
        if sell_id not in self._fulfilled['Sell']:
            self._fulfilled['Sell'][sell_id] = []
        self._fulfilled['Buy'][buy_id] += [[filled, price]]
        self._fulfilled['Sell'][sell_id] += [[filled, price]]
        self._book['Buy'][bb]['total'] -= filled
        self._book['Sell'][bs]['total'] -= filled
        self._book['Buy'][bb]['orders'][buy_id] -= filled
        self._book['Sell'][bs]['orders'][sell_id] -= filled
        if self._book['Buy'][bb]['orders'][buy_id] == 0:
            self._book['Buy'][bb]['ids'] = self._book['Buy'][bb]['ids'][1:]
            del self._book['Buy'][bb]['orders'][buy_id]
        if self._book['Sell'][bs]['orders'][sell_id] == 0:
            self._book['Sell'][bs]['ids'] = self._book['Sell'][bs]['ids'][1:]
            del self._book['Sell'][bs]['orders'][sell_id]
        self._reconcile_orders()

    def _set_audit(self):
        if len(self._prices['Buy']) == 0 or len(self._prices['Sell']) == 0:
            middle = (self._audit['bid'][-1] + self._audit['ask'][-1]) / 2.0
            self._audit['bid'] += [middle]
            self._audit['ask'] += [middle]
        else:
            self._audit['bid'] += [-self._prices['Buy'][0]]
            self._audit['ask'] += [self._prices['Sell'][0]]
