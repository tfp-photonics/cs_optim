import torch
import numpy as np
from torch.autograd.functional import jacobian
from models.losses import *
from models.model import LinearNetwork
import sqlite3


class Objective:  ## one objective for single with _n_shells
    def __init__(
        self,
        model: LinearNetwork,
        X: torch.Tensor,
        Y: torch.Tensor,
        fix_indices: bool = False,
        n_shells: int = 4,
        enhancement: list = None,
        width: int = 0,
    ):

        """

        Args:
            model (LinearNetwork): Surrogate model
            X (torch.Tensor): initial parameters
            Y (torch.Tensor): target
            fix_indices (bool, optional): If set to True, fixes index values of X and exlcudes them from jacobians. Defaults to False.
            n_shells (int, optional): Number of shells considered. Defaults to 4.
            enhancement (list, optional): Wavelength(s) to enhance. Defaults to None.
            width (int, optional): wl width around enhancement values. Defaults to 0.

        """

        self._model = model
        self.X = X
        self.Y = Y
        self._n_shells = n_shells
        self._wlens = np.linspace(400, 800, 200)
        self._wd = int(width // (400 / 201))
        self._enhancement = None
        self.fix_indices = fix_indices

        @property
        def X(self):
            return self.X

        @X.setter
        def X(self, val: torch.Tensor):
            self.X = val

        @property
        def Y(self):
            return self.Y

        @Y.setter
        def Y(self, val: torch.Tensor):
            self.Y = val

        @property
        def fix_indices(self):
            return self.fix_indices

        @fix_indices.setter
        def fix_indices(self, val: bool):
            self.fix_indices = val

        ## calculate indices that belong to enhancement region
        if enhancement != None:
            inside = np.zeros_like(self._wlens, dtype=bool)
            for enh in enhancement:
                assert (
                    400 < enh < 800
                ), f"enhancement wavelengths have to be inside 400nm to 800nm but is {enh}"
                ind0 = np.argmin(np.abs(self._wlens - enh))
                inside[ind0 - self._wd : ind0 + self._wd + 1] = True
            self._enhancement = inside

    def forward(self, x):  ## forward function, returns loss

        for p in self._model.parameters():
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()
        if type(x) is not torch.Tensor:
            x = torch.from_numpy(x).float()

        thickness = torch.zeros_like(
            x
        )  ### convert thickness to radii in a way that autograd likes
        thickness[: self._n_shells + 1] = torch.cumsum(x[: self._n_shells + 1], 0)
        x = x + thickness

        spec = self._model(x)

        if type(self._enhancement) is np.ndarray:
            loss_inside = (1 / spec[self._enhancement] ** 2).mean()
            loss_outside = (spec[~self._enhancement]).mean()
            # loss = 10*loss_inside+loss_outside
            loss = loss_inside * loss_outside
        else:
            loss = MSEloss_fn(spec, self.Y)

        return loss

    def jacs(self, x):  ### compute jacobians
        if type(x) is not torch.Tensor:
            x = torch.from_numpy(x).float()
        self.jac = jacobian(self.forward, x)

        if self.fix_indices:
            self.jac[self._n_shells + 1 :] = 0.0  ### not ideal..

        self.jac = self.jac.detach().numpy().astype("f8")
        return self.jac

    def spec(self, x, round=False):

        if type(x) is not torch.Tensor:
            x = torch.from_numpy(x).float()
        if round:
            x[self._n_shells + 1 :] = torch.round(x[self._n_shells + 1 :])

        thickness = torch.zeros_like(
            x
        )  ### convert thickness to radii in a way that autograd likes
        thickness[: self._n_shells + 1] = torch.cumsum(x[: self._n_shells + 1], 0)
        x = x + thickness
        y = self._model(x)
        return y.detach().numpy()


class Results:  ## store results
    def __init__(self):
        self.ids = -1
        self.specs = []
        self.MAEs = []
        self.params = []
        self.targets = []

    def __call__(self, num):
        return self.params[num], self.specs[num], self.MAEs[num]

    def add_result(self, params, spec, MAE, target):

        self.ids += 1
        self.specs.append(spec)
        self.MAEs.append(MAE)
        self.params.append(params)
        self.targets.append(target)

    def best_result(self):
        best = np.argmin(np.array(self.MAEs))
        return self.__call__(best)

    ###  /!\ under construction /!\
    ### still needs spectrum in and output? maybe also seeds... mhm not happy
    """
        ich glaube man könnte hier richtig coole relationslogik einbauen. target spectrum mit verschiedenen ids verknüpfen die darauf optimiert sind
        dann bräuchte man aber sowas wie target id und design id. 
    """

    # def save_to_db(self, name="db", ids=None):
    #     raise NotImplementedError

    #     con = sqlite3.connect(f"{name}.sqlite")
    #     cur = con.cursor()
    #     cur.execute(
    #         f"CREATE TABLE IF NOT EXISTS designs (design id INT, core material FLOAT, s1 material FLOAT, s2 material FLOAT, s3 material FLOAT,  s4 material FLOAT,",
    #         f"core radius FLOAT, s1 thickness FLOAT, s2 thickness FLOAT, s3 thickness FLOAT, s4 thickness FLOAT)",
    #     )
    #     cur.execute(
    #         f"CREATE TABLE IF NOT EXISTS target (target id INT)"
    #     )  ### target values
    #     cur.execute(f"CREATE TABLE IF NOT EXISTS spectrum (design id INT)")
    #     cur.execute(
    #         f"CREATE TAVLE IF NOT EXISTS optimization (design id INT, mae FLOAT)"
    #     )
    #     con.commit()
    #     if not ids:
    #         ids = range(self.num_sols)
    #     for i in ids:
    #         exec_string = ""
    #         for k in range(5):
    #             if k < len(self.params[i]) // 2:
    #                 exec_string += (
    #                     ", "
    #                     + str(self.params[i][k])
    #                     + ", "
    #                     + str(self.params[i][k + len(self.params[i] // 2)])
    #                 )
    #             else:
    #                 exec_string += ", NULL, NULL"
    #         cur.execute(f"INSERT INTO designs VALUES ({i}, {exec_string[2:]})")
    #         cur.execute(f"INSERT INTO optimization VALUES ({i}, {self.MAES[i]})")
    #         ### missing target and spectrum.
    #     con.close()
