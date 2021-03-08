from __future__ import absolute_import

import unittest
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from ds_behavioral.simulator.orderbook import Orderbook


class MyTestCase(unittest.TestCase):

    def setup(self):
        pass

    def tearDown(self):
        pass

    def test_runs(self):
        """Basic smoke test"""
        Orderbook('smoke', 100)

    def test_add_order(self):
        ob = Orderbook('testOrder', 100)
        ob.create_order(ob.SELL, 99, 5)
        self.assertEqual([*ob.book[ob.SELL]], [99])
        self.assertEqual(ob.book[ob.SELL][99]['orders'], {0:5})

        ob.create_order(ob.BUY, 99, 2)
        self.assertEqual([*ob.book[ob.SELL]], [99])
        self.assertEqual(ob.book[ob.SELL][99]['orders'], {0:3})

        self.assertEqual(ob.fulfilled[ob.BUY], {1: [[2, 99.0]]})
        print(ob.fulfilled)

    def test_demo(self):
        ob = Orderbook('testbook', 100)
        for _ in range(300):
            side = 'B'
            price = np.random.normal(99, 5)
            if np.random.random() < 0.5:
                side = 'S'
                price = np.random.normal(100, 5)
            price = int(price * 10) / 10.0
            size = int(200 * np.random.random())
            ob.create_order(side, price, size)
            print("Buy Fulfilled: {}\nSell Fulfilled: {}".format(ob.fulfilled['Buy'], ob.fulfilled['Sell']))

        print("Buy Orderbook\n{}\n\nSell Orderbook\n{}".format(ob.book['Buy'], ob.book['Sell']))

        Bids = ob.bid_ask['bid']
        Asks = ob.bid_ask['ask']
        plt.plot(Bids, label="bid price")
        plt.plot(Asks, label="ask price")
        plt.title("bid/ask prices times series")
        plt.legend(loc="upper right")
        plt.show()

    def test_delete(self):
        ob = Orderbook('testdel', 100)
        ob.create_order(ob.SELL, 96, 5)
        ob.create_order(ob.SELL, 97, 1)
        id = ob.create_order(ob.BUY, 90, 5)
        # print("\n\nBuy Orderbook\n{}\n\nSell Orderbook\n{}".format(ob.book['Buy'], ob.book['Sell']))
        # print("Bid/Ask: {}".format(ob.bid_ask))
        # print("prices: {}".format(ob.prices))
        self.assertEqual({96.0: {'ids': [0], 'orders': {0: 5}, 'total': 5}, 97.0: {'ids': [1], 'orders': {1: 1}, 'total': 1}}, ob.book['Sell'])
        self.assertEqual({'Sell': [96.0, 97.0], 'Buy': [-90.0]}, ob.prices)

        ob.delete_order(id)

        self.assertEqual({96.0: {'ids': [0], 'orders': {0: 5}, 'total': 5}, 97.0: {'ids': [1], 'orders': {1: 1}, 'total': 1}}, ob.book['Sell'])
        self.assertEqual({'Sell': [96, 97.0], 'Buy': []}, ob.prices)
        # print("\n\nBuy Orderbook\n{}\n\nSell Orderbook\n{}".format(ob.book['Buy'], ob.book['Sell']))
        # print("Bid/Ask: {}".format(ob.bid_ask))
        # print("prices: {}".format(ob.prices))

    def test_limit_buy(self):
        ob = Orderbook('testbook', 100)
        ob.create_order(ob.SELL, 99, 4)
        ob.create_order(ob.SELL, 97, 6)
        ob.create_order(ob.BUY, 100, 9)
        result = ob.fulfilled
        control = {'Sell': {1: [[6, 97.0]], 0: [[3, 99.0]]}, 'Buy': {2: [[6, 97.0], [3, 99.0]]}}
        self.assertEqual(control, result)
        result = ob.book
        control = {'Sell': {99.0: {'ids': [0], 'orders': {0: 1}, 'total': 1}}, 'Buy': {}}
        self.assertEqual(control, result)

    def test_market_buy(self):
        ob = Orderbook('testbook', 100)
        ob.create_order(ob.SELL, 99, 4)
        ob.create_order(ob.SELL, 97, 6)
        ob.create_order(ob.BUY, float("inf"), 15)
        print("\nfulfilled = {}".format(ob.fulfilled))
        print("\nbook = {}".format(ob.book))
        ob.create_order(ob.BUY, 103, 15)
        ob.create_order(ob.SELL, 102, 10)
        print("\nfulfilled = {}".format(ob.fulfilled))
        print("\nbook = {}".format(ob.book))

    def test_scratch(self):
        pass



if __name__ == '__main__':
    unittest.main()
