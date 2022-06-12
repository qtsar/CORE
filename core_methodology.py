import numpy as np
import math
import gurobipy as grb

delta_T = 1 / 360

"""
portfolio = [
    {
        'type': 'stock', # акция
        'underlying': 'stock1', # тикер базового актива
        'ds': True, # ежедневная переоценка
        'quantity': 10, # кол-во
        'daily_limit': 10000, # дневной лимит
        'execution_lag': 1, # на какой день от дефолта начать ликвидацию
        'days_to_maturity': 15 # дней до исполнения
    },
    {
        'type': 'forward', # форвард/фьючерс
        'underlying': 'stock1',
        'ds': True,
        'quantity': -100,
        'daily_limit': 10000,
        'execution_lag': 10,
        'strike': 90, # цена исполнения: только для фьючерсов и опционов
        'days_to_maturity': 360 # дней до исполнения
    },
]
"""

# основной класс
class CORE:

    def __init__(self, portfolio, price_paths=None, T=None):
        self._portfolio = portfolio

        self._quantities = np.array([asset['quantity'] for asset in self._portfolio]).reshape(len(self._portfolio), 1)
        self._ds = np.array([asset['ds'] for asset in self._portfolio]).reshape(len(self._portfolio), 1)
        self._abs_quantities = np.abs(self._quantities)

        if price_paths is not None:
            self._price_paths = price_paths
        else:
            # generate by gbm
            pass

        self._psi = self.get_portfolio_psi()

        if T is not None:
            self._T = T
        else:
            self._T = self.get_liquidation_period()

        self.validate_portfolio()

        self._strategy = None
        self._pl_matrix = None
        self._c_value = None

    def get_liquidation_period(self):
        check = [asset['execution_lag'] + math.ceil(abs(asset['quantity']) / asset['daily_limit'])
                 for asset in self._portfolio]
        return max(check) - 1

    def get_portfolio_psi(self):
        assert self._price_paths is not None

        mid = self._price_paths[:, :, 1:] - self._ds * self._price_paths[:, :, :-1]
        return mid * self._quantities

    def optimized_strategy(self, param="WL"):
        set_T = range(self._T)
        set_I = range(len(self._portfolio))

        opt_model = grb.Model()

        # переменная ---> оптимальная стратегия
        opt_strategy = [
            [opt_model.addVar(vtype=grb.GRB.INTEGER,
                              lb=0,  # ограничение 2
                              ub=self._abs_quantities[i],  # ограничение 3
                              name=f"Q_{i + 1}{t + 1}") for t in set_T]
            for i in set_I
        ]

        # ограничение 1
        opt_model.addConstrs(grb.quicksum(opt_strategy[i][t] for t in set_T) ==
                             self._abs_quantities[i] for i in set_I)

        # ограничение 5
        opt_model.addConstrs(opt_strategy[i][t] == 0
                             for i in set_I
                             for t in range(self._portfolio[i]["execution_lag"] - 1))

        # ограничение 6
        opt_model.addConstrs(opt_strategy[i][t] == 0
                             for i in set_I
                             for t in range(self._portfolio[i]["days_to_maturity"], self._T) if
                             self._portfolio[i]["days_to_maturity"] < self._T)

        # ограничение 7
        opt_model.addConstrs(opt_strategy[i][t] <= self._portfolio[i]["daily_limit"]
                             for t in set_T
                             for i in set_I)

        # то ---> что оптимизируем
        L_var = opt_model.addVar(vtype=grb.GRB.CONTINUOUS,
                                 lb=-grb.GRB.INFINITY)

        step1 = self._ds * (self._abs_quantities - np.cumsum(opt_strategy, axis=1)) + opt_strategy
        step2 = self._psi * step1 / self._abs_quantities
        step3 = np.sum(step2, axis=1).cumsum(axis=1)  # матрица L(r_k, h)

        first = step3[:, -1]  # L(r_k, H)

        # ограничение 8
        if param.upper() == "WL":
            opt_model.addConstrs(L_var <= step3[pp][t] + first[pp]
                                 for t in set_T
                                 for pp in range(len(step3)))

        elif param.upper() == "PL":
            opt_model.addConstrs(L_var <= first[pp]
                                 for pp in range(len(step3)))

        elif param.upper() == "TL":
            opt_model.addConstrs(L_var <= step3[pp][t]
                                 for t in set_T
                                 for pp in range(len(step3)))

        # L_var maximization
        opt_model.ModelSense = grb.GRB.MAXIMIZE
        opt_model.setObjective(L_var)
        opt_model.optimize()

        # optimal strategy
        self._strategy = np.array([[int(str(opt_strategy[i][t]).split()[-1][:-4]) for t in set_T] for i in set_I])
        self.profit_loss_matrix()
        self.calc_c_value()

        return self._strategy

    def profit_loss_matrix(self):
        assert self._strategy is not None

        step1 = self._ds * (self._abs_quantities - np.cumsum(self._strategy, axis=1)) + self._strategy
        step2 = self._psi * step1 / self._abs_quantities
        self._pl_matrix = step2

    def calc_c_value(self):
        assert self._strategy is not None

        step2 = self._pl_matrix
        permanent_loss = np.minimum(0, np.sum(step2, axis=1).sum(axis=1))

        mid = np.minimum(0, np.sum(step2, axis=1).cumsum(axis=1).min(axis=1))
        transient_loss = np.minimum(0, mid - permanent_loss)

        self._c_value = -np.min(transient_loss + permanent_loss)

    def get_strategy(self):
        return self._strategy

    def get_pl_matrix(self):
        return self._pl_matrix

    def get_c_value(self):
        return self._c_value

    def naive_strategy(self):
        set_T = range(self._T)
        set_I = range(len(self._portfolio))

        strategy = [[0 for _ in set_T] for _ in set_I]

        for i in set_I:
            hold = self._abs_quantities.squeeze()[i]

            for t in set_T:
                if (t < self._portfolio[i]['execution_lag'] - 1) or (t >= self._portfolio[i]["days_to_maturity"]):
                    strategy[i][t] = 0
                else:
                    strategy[i][t] = min(hold, self._portfolio[i]['daily_limit'])
                    hold -= strategy[i][t]

        self._strategy = np.array(strategy)
        self.profit_loss_matrix()
        self.calc_c_value()

        return self._strategy

    def set_strategy(self, new_strategy):
        self._strategy = new_strategy
        self.profit_loss_matrix()
        self.calc_c_value()

    def validate_portfolio(self):
        for asset in self._portfolio:
            if 'days_to_maturity' not in asset.keys():
                asset['days_to_maturity'] = self._T + 5

            if 'execution_lag' not in asset.keys():
                asset['execution_lag'] = 1

            if 'daily_limit' not in asset.keys():
                asset['daily_limit'] = 10000



