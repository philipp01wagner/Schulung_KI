import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def return_cmap():
    from matplotlib import cm
    import seaborn as sbn
    N = 10
    vals = np.ones((N, 4))
    vals[:, 0] = np.array([225, 240, 250, 250, 245, 230, 215, 200, 180, 165]) / 256
    vals[:, 1] = np.array([150, 180, 200, 220, 235, 235, 230, 215, 200, 175]) / 256
    vals[:, 2] = np.array([150, 170, 190, 210, 230, 245, 250, 255, 250, 230]) / 256
    vals[:, 3] = np.array([0.9 for _ in range(N)])
    colormap = ListedColormap(vals)
    newcmp = ListedColormap(colormap.colors[::-1])
    #return "Reds"
    return  sbn.diverging_palette(h_neg=245, h_pos=20, l=75, as_cmap=True)
    #return plt.cm.viridis(0.5)
    virdis = cm.get_cmap('viridis', 12)
    return virdis(0.2)

def single_plot_epochs(path):

    plt.rcParams.update({'font.size': 26})

    fig, ax = plt.subplots(figsize=(27, 10))
    data = np.atleast_2d(np.loadtxt(os.path.join(path, "results.csv"), delimiter=',', skiprows=1)).T

    epochs = np.arange(1, data.shape[1]+1)
    ax1, ax2 = ax, ax.twinx()
    ax1.set_xlabel("Number of Epochs", fontsize=28)
    ax1.set_ylabel("RMSE", fontsize=28)
    ax2.set_ylabel("NLL", fontsize=28)
    ax1.errorbar(epochs, data[0], yerr=data[1], fmt='-x', color='r', linewidth=3, elinewidth=2.5, label='RMSE', capsize=4)
    ax2.errorbar(epochs, data[2], yerr=data[3], fmt='-o', color='b', linewidth=3, elinewidth=2.5, label='NLL', capsize=4)
    fig.legend()
    

    plt.savefig(os.path.join(path, "epochs.pdf"), dpi=900, bbox_inches="tight")


def plot_binary_classification(model, x, y, i, fig_path, feature_range, modeltype):
    plt.rcParams.update({'font.size': 18})
    # for slides
    import seaborn as sbn
    sbn.set()
    min, max = feature_range
    _saved = {'x': x, 'y': y, 'index': i}

    fig, ax = plt.subplots(nrows=2, figsize=(9, 14))
    cticks = [[0, 0.5, 1], [0, 0.1, 0.2]]
    if modeltype == "MCD":
        cticks[1] = [0, 0.3, 0.5]
    elif modeltype == "DE":
        cticks[1] = [0, 0.05, 0.11]
    for n in [0, 1]:
        ax[n].set_xlabel(r'$x_1$', fontsize=22)
        ax[n].set_ylabel(r'$x_2$', fontsize=22)
        ax[n].set_xlim([min, max])
        ax[n].set_ylim([min, max])
        ax[n].xaxis.set_ticks(np.arange(min, max+0.01, 1.0))
        ax[n].yaxis.set_ticks(np.arange(min, max+0.01, 1.0))

        # for slides
        ax[n].set_axis_off()


        xx, yy, zz = create_points(model, feature_range, n)
        _saved["grid_x"] = xx
        _saved["grid_y"] = yy
        if n == 0:
            bon_1 = list(np.arange(0, 1.2, 0.2))
            cp_pred = ax[n].contourf(xx, yy, zz, bon_1, cmap=return_cmap())
            _saved["grid_pred"] = zz
        else:
            bon_2 = list(np.arange(0, 0.251, 0.025))
            if modeltype == "MCD":
                bon_2 = list(np.arange(0, 0.6, 0.05))
                zz = np.clip(zz, 0, 0.5)
            elif modeltype == "DE":
                bon_2 = list(np.arange(0, 0.12, 0.01))
                zz = np.clip(zz, 0, 0.1)
            cp_pred = ax[n].contourf(xx, yy, zz, bon_2, cmap=return_cmap())

            _saved["grid_pred_cov"] = zz
        fig.colorbar(cp_pred, ax=ax[n], pad=0.015, ticks=cticks[n])

    for j in range(i):
        ax[0].scatter(x[j, 0], x[j, 1], color=['#5874ac' if not y[j] else '#b95755'], s=10, alpha=0.6)
        ax[1].scatter(x[j, 0], x[j, 1], color=['#5874ac' if not y[j] else '#b95755'], s=10, alpha=0.6)

    plt.savefig(fig_path, bbox_inches="tight")



def plot_synthetic_plot(bnn_model, nn_model, X, y, fun, dir, x_scaler=None, y_scaler=None):

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 7))

    min1, max1 = [-6, 6]
    min2, max2 = [-99, 99]

    x_grid = np.arange(min1 - 0.1, max1 + 0.1, 0.1)
    if x_scaler is not None:
        x_grid = x_scaler.transform(x_grid.reshape(-1, 1))

    x_grid = x_grid.reshape(-1, 1)
    y_pred, y_cov = bnn_model.predict(torch.from_numpy(x_grid).float())
    nn_pred = nn_model.predict(x_grid)

    if x_scaler is not None:
        x_grid = x_scaler.inverse_transform(x_grid)
        X = x_scaler.inverse_transform(X)

    if y_scaler is not None:
        y = y * y_scaler[1] + y_scaler[0]
        nn_pred = nn_pred * y_scaler[1] + y_scaler[0]

    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_xlim([min1, max1])
    ax.set_ylim([min2, max2])

    l = r"$y = x^3$"

    ax.scatter(X, y, color='red', s=8, alpha=0.7)
    ax.plot(x_grid.squeeze(), fun(x_grid.squeeze()), color='black', alpha=0.8, label="ground truth", linewidth=3.0)
    ax.plot(x_grid, nn_pred.squeeze(), color='orange', label="MLP", linewidth=3.0)
    ax.plot(x_grid, y_pred.squeeze(), color='blue', label="KBNN", linewidth=3.0)
    
    int1 = y_pred.squeeze() - 3 * np.sqrt(y_cov).squeeze()
    int2 = y_pred.squeeze() + 3 * np.sqrt(y_cov).squeeze()
    ax.fill_between(x_grid.squeeze(), int1, int2, color='b', alpha=0.2, label=r'$\pm 3 \sigma $ Confidence')
    ax.legend(loc="upper center", ncol=4)

    plt.savefig(os.path.join(dir, "regression.pdf"), dpi=300, bbox_inches="tight")


def plot_synthetic_plot_with_all(models, X, y, fun, dir, metrics, ds, hidden):
    import seaborn as sbn
    import matplotlib.pyplot
    import pyro
    sbn.set()
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'text.usetex': True,
        'pgf.rcfonts': False,
        'font.size': 11,
    })
    n_models = len(models.keys())

    fig, axes = plt.subplots(nrows=2, ncols=n_models, figsize=(22, 3.5), sharex='col')
    #plt.suptitle("Compare " + ", ".join(list(models.keys())) + ": " + ds + "\nhidden="+str(hidden[0]))
    if ds == "linearv2":
        min1, max1 = [0, 10]
        min2, max2 = [0, 120]
    else:
        min1, max1 = [-6, 6]
        min2, max2 = [-99, 99]

    x_grid = np.arange(min1 - 0.1, max1 + 0.1, 0.1)
    x_grid = x_grid.reshape(-1, 1)
    axes = axes.flat
    ll = []
    for i in range(n_models):
        #pyro.get_param_store().clear()
        #pyro.clear_param_store()
        ax = [axes[i], axes[i+n_models]]
        model = list(models.values())[i]


        if list(models.keys())[i] == "PBP":
            m, v, v_noise = model.predict(x_grid)
            y_pred = m
            y_cov = v+v_noise
        else:
            y_pred, y_cov = model.predict(torch.from_numpy(x_grid).float())

        #ax[0].set_xlabel(r'$x$')
        if i == 0:
            #ax[0].set_ylabel(r'$\mu_y$')
            #ax[1].set_ylabel(r'$\sigma^2_y$')
            pass
        ax[0].set_xlim([min1, max1])
        ax[0].set_ylim([min2, max2])
        l5 = ax[0].scatter(X, y, color='#b95755', s=8, alpha=0.7,label="s")
        l1, = ax[0].plot(x_grid.squeeze(), fun(x_grid.squeeze()), color='#333333', alpha=0.8, label="ground truth", linewidth=2.0)
        l2, = ax[0].plot(x_grid, y_pred.squeeze(), "#5874ac", label=list(models.keys())[i], linewidth=2.0)

        int1 = y_pred.squeeze() - 3 * np.sqrt(y_cov).squeeze()
        int2 = y_pred.squeeze() + 3 * np.sqrt(y_cov).squeeze()
        #ax[0].set_title("RMSE={:.3f}".format(metrics[i]))
        ax[0].set_title(list(models.keys())[i] + "\nRMSE={:.3f}".format(metrics[i]))
        l3 = ax[0].fill_between(x_grid.squeeze(), int1, int2, color='#5874ac', alpha=0.2, label=r'$\pm 3 \sigma $ Confidence')
        #ax[0].legend(loc="upper left",  prop={'size': 7})

        l4, = ax[1].plot(x_grid, y_cov, "#67a66b", alpha=0.8, linewidth=2.0, label="a")
        ax[1].set_xlabel(r'$x$')
        if i == 0:
            ll = [l1, l2, l3, l4, l5]
            frame = fig.legend(
                ll,
                labels=["ground-truth", r"$\mu_y$", "training data", r'$\pm 3 \sigma $ confidence', r"$\sigma^2_y$"],
                loc="upper center",
                ncol=5,
                borderaxespad=-0.4,
                facecolor='none',
                edgecolor="none",
                fontsize=10
            )


    plt.subplots_adjust(hspace=0.05, wspace=0.3)
    plt.savefig(os.path.join(dir, f"regression_hidden={hidden[0]}_m={list(models.keys())}_rmse={list(metrics)}.pdf"), dpi=300, bbox_inches="tight")

def plot_deep_ensembles(model, X, y, fun, metrics, n=1):
    import seaborn as sbn
    import matplotlib.pyplot
    import matplotlib.pylab as pl

    sbn.set()
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'text.usetex': False,
        'pgf.rcfonts': False,
        'font.size': 11,
    })

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 8.5), sharex='col')
    min1, max1 = [-10, 10]
    min2, max2 = [-150, 150]
    ax = axes.flat
    x_grid = np.arange(min1 - 0.1, max1 + 0.1, 0.1)
    x_grid = x_grid.reshape(-1, 1)
    ll = []
    y_pred, y_cov, raw = model.predict(torch.from_numpy(x_grid).float())
    
    ax[0].set_xlim([min1, max1])
    ax[0].set_ylim([min2, max2])
    ax[0].plot(x_grid.squeeze(), fun(x_grid.squeeze()), color='#333333', alpha=0.8, label="Zielfunktion (Ground Truth)", linewidth=2.0)
    if len(raw)>0:
    
        colors = plt.cm.Blues(np.linspace(0.3, 1, n))
        for i in range(n):
               ax[0].plot(x_grid, raw[i].squeeze(),color=colors[i],linestyle="--", linewidth=2.0, label=f"Modell {i}")
    

        preds = raw[:n]
        y_pred = np.mean(preds, axis=0)
        y_cov = np.var(preds, axis=0)
    
        
    
    
        ax[0].plot(x_grid, y_pred.squeeze(), "#179c80", linewidth=2.0, label=f"Mittelwert")
        
        int1 = y_pred.squeeze() - 3 * np.sqrt(y_cov).squeeze()
        int2 = y_pred.squeeze() + 3 * np.sqrt(y_cov).squeeze()
        l3 = ax[0].fill_between(x_grid.squeeze(), int1, int2, color='#179c80', alpha=0.2, label=r'Unsicherheit')
        
        l4, = ax[1].plot(x_grid, y_cov, "#67a66b", alpha=0.8, linewidth=2.0, label="Varianz der Ensembles")
        ax[1].legend(loc="upper left")
    ax[1].set_xlabel(r'$x$')
    ax[0].legend()
    ax[0].scatter(X, y, color='#db8642', s=8, alpha=0.7,label="Daten")
    plt.subplots_adjust(hspace=0.05, wspace=0.3)
    plt.show()

def create_points(model, feature_range, mode):
    min1, max1 = feature_range
    min2, max2 = feature_range

    # define the x and y scale
    x1grid = np.arange(min1 - 0.1, max1 + 0.1, 0.1)
    x2grid = np.arange(min2 - 0.1, max2 + 0.1, 0.1)
    # create all of the lines and rows of the grid
    xx, yy = np.meshgrid(x1grid, x2grid)

    r1, r2 = xx.flatten(), yy.flatten()
    r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
    grid = np.hstack((r1, r2))

    grid = torch.from_numpy(grid.astype("Float32"))
    y, y_cov = model.predict(grid)
    if mode == 0:
        zz = np.reshape(y, xx.shape)
    else:
        zz = np.reshape(y_cov, xx.shape)

    return xx, yy, zz





