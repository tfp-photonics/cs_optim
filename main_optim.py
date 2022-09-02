from models.model import *
from objective import *
from models.losses import *
import torch

from scipy.optimize import minimize
import numpy as np
import os

import matplotlib.pyplot as plt
import matplotlib as mpl
from objective import Objective
import PyMieScatt as pms

import argparse

### refractive index dictionary for classes
refractive_indices = {
    1: 1.4649,  ## SiO2
    2: 1.7196,  ## MgO
    3: 1.9447,  ## ZnO
    4: 2.0745,  ## ZrO2
    5: 2.4317,  ## TiO2
}

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-n_init", type=int, default=100, help="max number of initializations"
    )
    parser.add_argument(
        "-max_iter", type=int, default=150, help="max interations of L-BFGS-B"
    )
    parser.add_argument(
        "-cond", type=float, default=0.1, help="success criterion in MAE for stopping"
    )
    parser.add_argument(
        "-max_suggested",
        type=int,
        default=5,
        help="maximum number of suggested layers to consider. In case of enhancement, should be 5",
    )
    parser.add_argument(
        "-enhancement", type=list, default=None, help="enhancement wavelenths"
    )
    parser.add_argument(
        "-width", type=int, default=0, help="width in nm for enhancement optimization"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="seed for torch and numpy random generators",
    )

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    n_points = 200
    w_min = 400
    w_max = 800
    wave_range = np.linspace(w_min, w_max, n_points)
    n_shells_max = 4

    ## load models for design
    ## c0s
    hidden_dim_c0s = np.array([279, 494, 100, 157, 295, 331])
    nlaf_c0s = np.array(
        [
            nn.Tanh(),
            nn.Tanhshrink(),
            nn.LeakyReLU(),
            nn.LeakyReLU(),
            nn.Tanhshrink(),
            nn.Tanh(),
        ]
    )
    ## c1s
    hidden_dim_c1s = np.array([576, 192, 400, 172, 184])
    nlaf_c1s = np.array(
        [nn.Tanh(), nn.Tanhshrink(), nn.LeakyReLU(), nn.Tanhshrink(), nn.Tanhshrink()]
    )
    ## c2s
    hidden_dim_c2s = np.array([309, 313, 183])
    nlaf_c2s = np.array([nn.Tanh(), nn.LeakyReLU(), nn.Tanhshrink()])
    ## c3s
    hidden_dim_c3s = np.array([132, 296, 124, 478, 211, 292, 110])
    nlaf_c3s = np.array(
        [
            nn.LeakyReLU(),
            nn.Tanhshrink(),
            nn.Tanhshrink(),
            nn.LeakyReLU(),
            nn.LeakyReLU(),
            nn.LeakyReLU(),
            nn.LeakyReLU(),
        ]
    )
    ## c4s
    hidden_dim_c4s = np.array([111, 188, 200, 287, 117])
    nlaf_c4s = np.array(
        [
            nn.LeakyReLU(),
            nn.Tanhshrink(),
            nn.LeakyReLU(),
            nn.LeakyReLU(),
            nn.LeakyReLU(),
        ]
    )

    hiddens = [
        hidden_dim_c0s,
        hidden_dim_c1s,
        hidden_dim_c2s,
        hidden_dim_c3s,
        hidden_dim_c4s,
    ]
    nlafs = [nlaf_c0s, nlaf_c1s, nlaf_c2s, nlaf_c3s, nlaf_c4s]

    save_model_path = "models/saved_models/"
    input_dim = [(i + 1) * 2 for i in range(n_shells_max + 1)]
    output_dim = n_points
    models = [
        LinearNetwork(input_dim[i], hiddens[i], output_dim, nlafs[i])
        for i in range(n_shells_max + 1)
    ]

    names = [f"c{i}s.pt" for i in range(5)]
    for i in range(n_shells_max + 1):
        checkpoint = torch.load(save_model_path + names[i])
        models[i].load_state_dict(checkpoint["model_state_dict"])
        models[i].eval()

    ## load classifier:
    hidden_dim = np.array([559, 234, 426, 330, 204, 454])
    nlaf = np.array(
        [
            nn.Tanh(),
            nn.LeakyReLU(),
            nn.LeakyReLU(),
            nn.LeakyReLU(),
            nn.Tanh(),
            nn.Tanh(),
        ]
    )
    classifier_name = "cns_class.pt"

    classifier = LinearNetwork(n_points, hidden_dim, n_shells_max + 1, nlaf)
    # classifier = ClassifierNetwork(n_points, hidden_dim, n_shells_max+1, nlaf)
    checkpoint = torch.load(save_model_path + classifier_name)
    classifier.load_state_dict(checkpoint["model_state_dict"])
    classifier.eval()

    bounds = [[[50, 100], [1, 5]]]
    for i in range(n_shells_max):
        a = [[50, 100]]
        a.extend([[30, 50] for k in range(i + 1)])
        a.extend([[1, 5] for k in range(i + 2)])
        bounds.append(a)

    ## bare core and core-shell example target spectra
    spec_des = np.array(
        [pms.MieQ(m=2, wavelength=wl, diameter=400)[1] for wl in wave_range]
    )
    # spec_des = np.array([pms.MieQCoreShell(mCore = 3, mShell = 2, wavelength = wl, dCore = 50, dShell = 250)[1] for wl in wave_range])
    Y = torch.from_numpy(spec_des).float()

    suggest_layeres = classifier(Y).detach().numpy()
    suggest_layeres = np.flip(np.argsort(suggest_layeres))

    res = Results()
    limit = 100
    for i, n in enumerate(list(suggest_layeres[: args.max_suggested])):
        limit = 100
        for k in range(args.n_init):
            params_init = np.array(
                [np.random.uniform(b[0], b[1]) for b in bounds[n][: n + 1]]
            )
            params_init = np.append(params_init, np.random.randint(1, 6, (n + 1)))
            X = torch.from_numpy(params_init).float()

            model = Objective(
                models[n],
                X,
                Y,
                n_shells=n,
                enhancement=args.enhancement,
                width=args.width,
            )

            opt = minimize(
                model.forward,
                model.X,
                method="L-BFGS-B",
                jac=model.jacs,
                bounds=bounds[n],
                options={"gtol": 1e-6, "disp": False, "maxiter": args.max_iter},
            )

            spec_opt = model.spec(opt.x, round=True)
            MAE_opt = MAEloss_fn(spec_des, spec_opt)
            if MAE_opt <= limit:
                limit = MAE_opt
                param_opt = opt.x
                if MAE_opt <= args.cond:
                    break

        params_init_FI = np.copy(param_opt)
        params_init_FI[n + 1 :] = np.round(params_init_FI[n + 1 :])

        X = torch.from_numpy(params_init_FI).float()
        model.X = X
        model.fix_indices = True

        opt = minimize(
            model.forward,
            model.X,
            method="L-BFGS-B",
            jac=model.jacs,
            bounds=bounds[n],
            options={"gtol": 1e-6, "disp": False, "maxiter": args.max_iter},
        )

        param_opt = opt.x
        spec_opt = model.spec(param_opt, round=True)
        MAE_opt = MAEloss_fn(spec_des, spec_opt)
        print("Result using ", n, " layers: ", MAE_opt)
        res.add_result(param_opt, spec_opt, MAE_opt)

    ## best results:
    params_opt, spec_opt, MAE_opt = res.best_result()
    print("best result: ", params_opt, "\nbest MAE: ", MAE_opt)
