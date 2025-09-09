import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from predictor import predict


def plot_probability(
    model,
    data_loader,
    device,
    threshold=0.07,
    out_path=None,
    bw_method='silverman',
    figsize=(4, 3.5),
    tick_labelsize=13):
    
    preds, labels, _ = predict(
        model, data_loader, device,
        return_embeddings=False
    )
    preds = preds.ravel()
    labels = labels.ravel().astype(bool)
    pos = preds[labels]
    neg = preds[~labels]

    kde_pos = gaussian_kde(pos, bw_method=bw_method)
    kde_neg = gaussian_kde(neg, bw_method=bw_method)

    lower, upper = -0.25, 1.25
    xs = np.linspace(lower, upper, 600)
    y_pos = kde_pos(xs)
    y_neg = kde_neg(xs)

    fig, ax = plt.subplots(figsize=figsize)

    ax.fill_between(
        xs, y_neg,
        color="#A1C3CE", alpha=0.7,
        label="Inactives"
    )
    ax.fill_between(
        xs, y_pos,
        color="#2AAD0F", alpha=0.5,
        label="Actives"
    )
    ax.axvline(
        threshold,
        color="gray", linestyle="--", lw=2,
        label=f"Threshold = {threshold}"
    )
    ax.set_xlim(lower, upper)
    ax.set_xticks([0, 0.5, 1.0])
    peak = max(y_neg.max(), y_pos.max())
    ymax = int(np.ceil(peak))
    ax.set_ylim(0, ymax)
    ax.set_yticks(np.arange(0, ymax+1, 1))

    ax.set_xlabel(
        "Predicted probability", fontsize=15
    )
    ax.set_ylabel("Density", fontsize=15)
    ax.tick_params(
        axis="both", which="major",
        labelsize=tick_labelsize,
        direction="out", length=6, width=1
    )
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)
        spine.set_color("black")

    ax.grid(False)
    ax.legend(
        loc="upper left",
        frameon=False, fontsize=9
    )

    if out_path:
        os.makedirs(
            os.path.dirname(out_path) or ".",
            exist_ok=True
        )
        fig.savefig(
            out_path,
            dpi=300,
            bbox_inches="tight"
        )
    plt.tight_layout()
    plt.show()



import torch
import seaborn as sns
from sklearn.metrics import confusion_matrix
import json

def plot_confusion_matrix_and_histograms(
    model,
    dataloader_list,
    device,
    out_path=None,
    set_names=["Train", "Validation", "Test"],
    thresholds_json=None    
):
    model.eval()
    results = []

    # 1. Inferência e preparo dos dados para cada split
    for loader in dataloader_list:
        probs = []
        targets = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                y = batch.y
                output = model(batch)
                prob = torch.sigmoid(output).view(-1).cpu().numpy()
                target = y.view(-1).cpu().numpy()
                probs.append(prob)
                targets.append(target)
        probs = np.concatenate(probs)
        targets = np.concatenate(targets)
        valid_mask = ~np.isnan(targets)
        probs = probs[valid_mask]
        targets = targets[valid_mask]
        results.append((probs, targets))

    # 2. Determinação dos thresholds a partir do JSON, SE existir
    thresholds = [0.5] * len(dataloader_list)  # Default
    using_calibrated = False
    if thresholds_json:
        try:
            if os.path.isfile(thresholds_json):
                with open(thresholds_json, "r") as f:
                    th_dict = json.load(f)
                thresholds = [
                    float(np.mean(th_dict['train'])),
                    float(np.mean(th_dict['val'])),
                    float(np.mean(th_dict['test']))
                ]
                using_calibrated = True
                print("Thresholds usados (do JSON):", thresholds)
            else:
                print(f"Arquivo {thresholds_json} não encontrado, usando thresholds padrão (0.5).")
        except Exception as e:
            print(f"Erro ao ler o arquivo {thresholds_json}: {e}")
            print("Usando thresholds padrão (0.5).")

    # 3. Layout horizontal: 2 linhas, 3 colunas
    fig = plt.figure(figsize=(18, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 1], hspace=0.28, wspace=0.20)

    axes = []
    for row in range(2):
        for col in range(3):
            ax = fig.add_subplot(gs[row, col])
            axes.append(ax)
    axes = np.array(axes).reshape(2, 3)

    for i, (set_name, (probs, targets), threshold) in enumerate(zip(set_names, results, thresholds)):
        preds = (probs >= threshold).astype(int)
        cm = confusion_matrix(targets, preds)
        labels = ['Inativo', 'Ativo']

        print("Conjunto:", set_name)
        print("Threshold:", threshold)
        print("Targets únicos:", np.unique(targets))
        print("Preds únicos:", np.unique(preds))
        print("Matriz de confusão:\n", cm)

        # Linha 1: Matriz de Confusão
        ax1 = axes[0, i]
        sns.heatmap(
            cm,
            annot=True,
            fmt='.0f',
            cmap='GnBu',
            cbar=False,
            square=True,
            linewidths=2,
            linecolor='white',
            ax=ax1,
            annot_kws={'fontsize':16, 'weight':'bold'}
        )
        ax1.set_xlabel('Predito', fontsize=16, weight='bold')
        ax1.set_ylabel('Experimental', fontsize=16, weight='bold')
        ax1.set_xticklabels(labels, fontsize=14)
        ax1.set_yticklabels(labels, fontsize=14, rotation=90)
        ax1.set_title(f"{set_name}", fontsize=17, weight='bold')
        for spine in ax1.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2)
            spine.set_edgecolor('white')


        # Linha 2: Histograma das probabilidades
        ax2 = axes[1, i]
        bins = np.linspace(0, 1, 40)
        # ax2.hist(probs[targets == 0], bins=bins, color='green', alpha=0.7, label='Inativo')
        # ax2.hist(probs[targets == 1], bins=bins, color='orange', alpha=0.8, label='Ativo')
        # ax2.set_yscale('log')  # escala logarítima para exxergar melhor ativos, porém distorce
        ax2.hist(probs[targets == 0], bins=bins, color='green', alpha=0.7, label=f'Inativo (n={np.sum(targets==0)})', density=True)
        ax2.hist(probs[targets == 1], bins=bins, color='orange', alpha=0.8, label=f'Ativo (n={np.sum(targets==1)})', density=True)
        ax2.axvline(threshold, color='red', linestyle='--', linewidth=2.5, zorder=5, 
                    label=f'Threshold = {threshold:.2f}')
        ax2.set_xlabel('Probabilidade de Predição', fontsize=15, weight='bold')
        ax2.set_ylabel('Densidade', fontsize=15, weight='bold')
        # ax2.set_ylabel('Frequência', fontsize=15, weight='bold') 
        ax2.set_xlim(0, 1)
        handles, labels_ = ax2.get_legend_handles_labels()
        ax2.legend(handles, labels_, loc='best', fontsize=10, frameon=False)
        # ax2.set_title(f"{set_name}", fontsize=15, weight='bold')
        # for spine in ax2.spines.values():
        #     spine.set_visible(True)
        #     spine.set_linewidth(1.3)
        #     spine.set_edgecolor('black')

    plt.tight_layout()
    
    # Gera nome do arquivo baseado no tipo de threshold usado
    if using_calibrated:
        filename = "calibrated_confusion_matrix_and_histograms.png"
    else:
        filename = "confusion_matrix_and_histograms.png"
    
    file_path = os.path.join(out_path, filename)
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.show()
    
