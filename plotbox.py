import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 16
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = '#000000'
plt.rcParams['axes.linewidth'] = 1.2

science_palette = ["#1f77b4", "#ff7f0e", "#2ca02c",
                   "#d62728", "#9467bd", "#8c564b", "#e377c2"]

input_dim = 4
distribution = 'chi2'

dist_index = {'norm': 0, 't': 1, 'chi2': 2}
dim_index = {1: 0, 3: 1, 4: 2}
base_number = 2 + dim_index[input_dim] * 3 + dist_index[distribution]

train_data = [
    np.load(f'ANN/data/{input_dim}0.2train_{distribution}.npy'),
    np.load(f'ANN/data/{input_dim}0.8train_{distribution}.npy'),
    np.load(f'ERNN/data/{input_dim}0.2train_{distribution}.npy'),
    np.load(f'ERNN/data/{input_dim}0.8train_{distribution}.npy'),
    np.load(f'CERNN/data/{input_dim}3train_{distribution}.npy'),
    np.load(f'CERNN/data/{input_dim}5train_{distribution}.npy'),
    np.load(f'CERNN/data/{input_dim}9train_{distribution}.npy')
]
test_data = [
    np.load(f'ANN/data/{input_dim}0.2test_{distribution}.npy'),
    np.load(f'ANN/data/{input_dim}0.8test_{distribution}.npy'),
    np.load(f'ERNN/data/{input_dim}0.2test_{distribution}.npy'),
    np.load(f'ERNN/data/{input_dim}0.8test_{distribution}.npy'),
    np.load(f'CERNN/data/{input_dim}3test_{distribution}.npy'),
    np.load(f'CERNN/data/{input_dim}5test_{distribution}.npy'),
    np.load(f'CERNN/data/{input_dim}9test_{distribution}.npy')
]

labels = [
    "ANN\n(τ = 0.2)",
    "ANN\n(τ = 0.8)",
    "ERNN\n(τ = 0.2)",
    "ERNN\n(τ = 0.8)",
    "CERNN$_3$",
    "CERNN$_5$",
    "CERNN$_9$"
]

palette_dict = {labels[i]: science_palette[i] for i in range(len(labels))}

df_train = pd.DataFrame({
    'Model': np.repeat(labels, [len(d) for d in train_data]),
    'MAE': np.concatenate([d[:, 0] for d in train_data]),
    'RMSE': np.concatenate([d[:, 1] for d in train_data])
})
df_test = pd.DataFrame({
    'Model': np.repeat(labels, [len(d) for d in test_data]),
    'MAE': np.concatenate([d[:, 0] for d in test_data]),
    'RMSE': np.concatenate([d[:, 1] for d in test_data])
})

def plot_and_save(df, metric, fig_number):

    plt.figure(figsize=(10, 7))

    sns.boxplot(
        x='Model', y=metric, hue='Model',
        data=df, palette=palette_dict, order=labels,
        boxprops=dict(linewidth=1.2),
        medianprops=dict(color='black', linewidth=1.5)
    )

    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")

    plt.tight_layout()

    save_path = f"论文（投稿）/{fig_number}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {save_path}")

plot_and_save(df_train, "MAE",  f"{base_number}-1")
plot_and_save(df_train, "RMSE", f"{base_number}-2")
plot_and_save(df_test,  "MAE",  f"{base_number}-3")
plot_and_save(df_test,  "RMSE", f"{base_number}-4")
