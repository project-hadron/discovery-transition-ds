from __future__ import absolute_import

import unittest
from datetime import datetime

from ds_behavioral.simulator.orderbook_manager import OrderbookManager


class OrderbookManagerTest(unittest.TestCase):

    def setup(self):
        pass

    def tearDown(self):
        pass

    def test_runs(self):
        """Basic smoke test"""
        OrderbookManager()

    def test_create_book(self):
        obm = OrderbookManager()
        self.assertTrue(len(obm.orderbooks) == 0)
        ob = obm.create_orderbook('createTest01', 99)
        self.assertEqual(['createTest01'], obm.orderbooks)
        self.assertTrue(len(obm.orderbooks) == 1)
        ob = obm.create_orderbook('createTest02', 102)
        self.assertEqual(['createTest01', 'createTest02'], obm.orderbooks)
        self.assertTrue(len(obm.orderbooks) == 2)
        for book in ['createTest01', 'createTest02']:
            self.assertEqual({}, obm.get_order_agent(book))
            self.assertEqual([], obm.get_fulfilled(book))

    def test_add_order(self):
        name = 'addOrderTest'
        obm = OrderbookManager()
        obm.create_orderbook(name, 99)
        obm.add_limit_order(name, '1001', 'S', 98, 20, 1000001)
        control = {'Sell': {98.0: {'ids': [0], 'orders': {0: 20}, 'total': 20}}, 'Buy': {}}
        self.assertEqual(control, obm.get_raw_order_book(name))
        obm.add_limit_order(name, '1001', 'S', 98, 10, 1000002)
        obm.add_limit_order(name, '1002', 'b', 97, 20, 1000003)
        control = {'Sell': {98.0: {'ids': [0, 1], 'orders': {0: 20, 1: 10}, 'total': 30}},
                   'Buy': {97.0: {'ids': [2], 'orders': {2: 20}, 'total': 20}}}
        self.assertEqual(control, obm.get_raw_order_book(name))
        # test ask bid
        self.assertEqual(98, obm.get_bid_ask(name).get('ask')[-1])
        self.assertEqual(97, obm.get_bid_ask(name).get('bid')[-1])
        #test agent orders
        control = {0: {'agent': '1001', 'action': {1000001: {'status': 'limit'}}},
                   1: {'agent': '1001', 'action': {1000002: {'status': 'limit'}}},
                   2: {'agent': '1002', 'action': {1000003: {'status': 'limit'}}}}
        self.assertEqual(control, obm.get_order_agent(name))

    def test_cancel_order_sell(self):
        name = 'cancelOrderTest'
        obm = OrderbookManager()
        obm.create_orderbook(name, 99)
        obm.add_limit_order(name, '1001', 'S', 98, 20, 1000001)
        control = {'Sell': {98.0: {'ids': [0], 'orders': {0: 20}, 'total': 20}}, 'Buy': {}}
        self.assertEqual(control, obm.get_raw_order_book(name))
        obm.cancel_order(name, 0, 1000003)
        control = {'Buy': {}, 'Sell': {}}
        self.assertEqual(control, obm.get_raw_order_book(name))
        control = {0: {'agent': '1001', 'action': {1000001: {'status': 'limit'},
                                                   1000003: {'status': 'cancel'}}}}
        self.assertEqual(control, obm.get_order_agent(name))

    def test_cancel_order_buy(self):
        name = 'cancelOrderTest'
        obm = OrderbookManager()
        obm.create_orderbook(name, 99)
        obm.add_limit_order(name, '1001', 'B', 98, 20, 1000001)
        control = {'Sell': {}, 'Buy': {98.0: {'ids': [0], 'orders': {0: 20}, 'total': 20}}}
        self.assertEqual(control, obm.get_raw_order_book(name))
        obm.cancel_order(name, 0, 1000003)
        control = {'Buy': {}, 'Sell': {}}
        self.assertEqual(control, obm.get_raw_order_book(name))
        control = {0: {'agent': '1001', 'action': {1000001: {'status': 'limit'},
                                                   1000003: {'status': 'cancel'}}}}
        self.assertEqual(control, obm.get_order_agent(name))

    def test_fulfilled(self):
        name = 'fulfillOrderTest'
        obm = OrderbookManager()
        obm.create_orderbook(name, 99)
        obm.add_limit_order(name, '1001', 'S', 98, 20, 1000001)
        control = []
        self.assertEqual(control, obm.get_fulfilled(name))
        obm.add_limit_order(name, '1001', 'B', 98, 10, 1000002)
        control = [{'Sell': {0: [[10, 98.0]]},
                    'Buy': {1: [[10, 98.0]]},
                    'timestamp': 1000002}]
        self.assertEqual(control, obm.get_fulfilled(name))
        obm.add_limit_order(name, '1001', 'B', 98, 20, 1000004)
        control = [{'Sell': {0: [[10, 98.0]]},
                    'Buy': {1: [[10, 98.0]]},
                    'timestamp': 1000002},
                   {'Sell': {0: [[10, 98.0]]},
                    'Buy': {2: [[10, 98.0]]},
                    'timestamp': 1000004}]
        self.assertEqual(control, obm.get_fulfilled(name))
        control = {'Sell': {},
                   'Buy': {98.0: {'ids': [2], 'orders': {2: 10}, 'total': 10}}}
        self.assertEqual(control, obm.get_raw_order_book(name))

    def test_market_order_sell(self):
        name = 'marketOrderSellTest'
        obm = OrderbookManager()
        obm.create_orderbook(name, 99)
        obm.add_limit_order(name, '1001', 'B', 98, 10, 1000001)
        obm.add_market_order(name, '1002', 'S', 20, 1000002)
        control = {'Sell': {1.0: {'ids': [1], 'orders': {1: 10}, 'total': 10}}, 'Buy': {}}
        self.assertEqual(control, obm.get_raw_order_book(name))
        control = [{'Sell': {1: [[10, 98.0]]},
                    'Buy': {0: [[10, 98.0]]},
                    'timestamp': 1000002}]
        self.assertEqual(control, obm.get_fulfilled(name))
        obm.add_limit_order(name, '1001', 'B', 92, 20, 1000003)
        control = {'Sell': {}, 'Buy': {92.0: {'ids': [2], 'orders': {2: 10}, 'total': 10}}}
        self.assertEqual(control, obm.get_raw_order_book(name))
        control = [{'Sell': {1: [[10, 98.0]]},
                    'Buy': {0: [[10, 98.0]]}, 'timestamp': 1000002},
                   {'Sell': {1: [[10, 92.0]]},
                    'Buy': {2: [[10, 92.0]]}, 'timestamp': 1000003}]
        self.assertEqual(control, obm.get_fulfilled(name))

    def test_market_order_buy_sell(self):
        name = 'marketOrderBuySellTest'
        obm = OrderbookManager()
        obm.create_orderbook(name, 99)
        obm.add_market_order(name, '1001', 'B', 20, 1000001)
        obm.add_market_order(name, '1002', 'S', 20, 1000002)
        control = {'Sell': {}, 'Buy': {}}
        self.assertEqual(control, obm.get_raw_order_book(name))
        control = [{'Sell': {1: [[20, 99.0]]},
                    'Buy': {0: [[20, 99.0]]}, 'timestamp': 1000002}]
        self.assertEqual(control, obm.get_fulfilled(name))
        obm.add_market_order(name, '1002', 'S', 20, 1000003)
        obm.add_market_order(name, '1001', 'B', 20, 1000004)
        control = {'Sell': {}, 'Buy': {}}
        self.assertEqual(control, obm.get_raw_order_book(name))
        control = [{'Sell': {1: [[20, 99.0]]},
                    'Buy': {0: [[20, 99.0]]}, 'timestamp': 1000002},
                   {'Sell': {2: [[20, 99.0]]},
                    'Buy': {3: [[20, 99.0]]}, 'timestamp': 1000004}]
        self.assertEqual(control, obm.get_fulfilled(name))
        # print(obm.get_order_book(name))
        # print(obm.get_fulfilled(name))

    def test_get_current_bid_ask(self):
        name = 'currentBidAsk'
        obm = OrderbookManager()
        obm.create_orderbook(name, 99)
        # obm.add_limit_order(name, '1001', 'S', 98, 20, 1000001)
        # obm.add_limit_order(name, '1002', 'B', 95, 20, 1000002)

        obm.add_market_order(name, '1001', 'B', 20, 1000001)
        obm.add_market_order(name, '1002', 'S', 20, 1000002)
        obm.add_limit_order(name, '1001', 'B', 96, 10, 1000003)
        obm.add_limit_order(name, '1002', 'S', 98, 10, 1000004)

        self.assertEqual(96.0, obm.get_current_bid(name))
        self.assertEqual(98.0, obm.get_current_ask(name))
        # print("\nbid: {}\nask: {}".format(obm.current_bid(name), obm.current_ask(name)))
        # print(obm.get_order_book(name))
        # print(obm.get_fulfilled(name))

    def test_exchange(self):
        exchange = 'LSEG'
        obm = OrderbookManager()
        self.assertEqual('Default', obm.exchange)
        obm = OrderbookManager(exchange=exchange)
        self.assertEqual(exchange, obm.exchange)
        obm = OrderbookManager(exchange='')
        self.assertEqual('Default', obm.exchange)

    def test_date_ref(self):
        exchange = 'LSEG'
        now_ref = datetime.now().strftime("%Y-%m-%d")
        day_ref = "2018-01-01"
        obm = OrderbookManager()
        self.assertEqual(now_ref, obm.day_reference)
        obm = OrderbookManager(day_reference=day_ref)
        self.assertEqual(day_ref, obm.day_reference)
        obm = OrderbookManager(day_reference='')
        self.assertEqual(now_ref, obm.day_reference)

    def test_build_orderbooks(self):
        exchange = 'LSEG'
        orderbooks = {'OAH7.L': 29.575, 'OB65.L': 17.835, 'OCXC.L': 14.9225, 'ODHJ.L': 9.19}
        obm = OrderbookManager(exchange=exchange, orderbooks=orderbooks)
        self.assertEqual(['OAH7.L', 'OB65.L', 'OCXC.L', 'ODHJ.L', ], obm.orderbooks)
        self.assertEqual(29.575, obm.get_current_ask('OAH7.L'))
        self.assertEqual(obm.get_current_bid('OAH7.L'), obm.get_current_ask('OAH7.L'))
        self.assertEqual(14.9225, obm.get_current_ask('OCXC.L'))
        self.assertEqual(obm.get_current_bid('OCXC.L'), obm.get_current_ask('OCXC.L'))

    def test_get_volume(self):
        name = 'volumeTest'
        obm = OrderbookManager()
        obm.create_orderbook(name, 99)
        self.assertEqual(0, obm.get_current_volume(name))
        obm.add_limit_order(name, '1001', 'S', 98, 40, 1000001)
        self.assertEqual(0, obm.get_current_volume(name))
        obm.add_limit_order(name, '1001', 'B', 98, 10, 1000002)
        obm.add_limit_order(name, '1001', 'B', 98, 20, 1000003)
        obm.add_limit_order(name, '1002', 'B', 98, 5, 1000004)
        self.assertEqual(35, obm.get_current_volume(name))

    def test_get_formatted_orderbook(self):
        name = 'volumeTest'
        obm = OrderbookManager()
        obm.create_orderbook(name, 99)
        obm.add_limit_order(name, '1001', 'S', 98, 40, 1000001)
        obm.add_limit_order(name, '1001', 'S', 99, 15, 1000001)
        obm.add_limit_order(name, '1001', 'B', 98, 10, 1000002)
        obm.add_limit_order(name, '1001', 'B', 95, 10, 1000003)
        control = {'buy': [{'price': 95.0, 'volume': 10}],
                   'buy_volume': 10,
                   'sell': [{'price': 98.0, 'volume': 30},
                            {'price': 99.0, 'volume': 15}],
                   'sell_volume': 45}
        self.assertEqual(control, obm.get_order_book(name))

    def test_scratch(self):
        pass


if __name__ == '__main__':
    unittest.main()
