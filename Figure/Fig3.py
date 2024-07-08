import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
# 9596
# 7663
# 9470
# 7103
# 9279
# 6325
column_names = ['methods', '1:7', '1:6', '1:5']
data = [
    ['AnomalGRN',       0.9279, 0.02, 0.01],
    ['GNNLink',         0.75, 0.05, 0.02],
    ['GENELink',        0.70, 0.05, 0.05],
    ['GNE',             0.68, 0.03, 0.03],
    ['CNNC',            0.62, 0.04, 0.06],
    ['DeepDRIM',        0.59, 0.04, 0.07],
    ['GRN-Transformer', 0.50, 0.02, 0.06],
    ['GENIE3',          0.52, 0.07, 0.03]
]
data_auc = pd.DataFrame(data, columns=column_names)
data = [
    ['AnomalGRN',       0.6325, 0.08, 0.05],
    ['GNNLink',         0.48, 0.01, 0.03],
    ['GENELink',        0.42, 0.04, 0.01],
    ['GNE',             0.21, 0.05, 0.04],
    ['CNNC',            0.29, 0.02, 0.07],
    ['DeepDRIM',        0.30, 0.05, 0.02],
    ['GRN-Transformer', 0.23, 0.03, 0.01],
    ['GENIE3',          0.22, 0.03, 0.02]
]
data_aupr = pd.DataFrame(data, columns=column_names)

data_auc['methods'] = data_auc['methods'].str.replace('\t', '', regex=False)
data_auc.set_index('methods', inplace=True)
data_aupr['methods'] = data_aupr['methods'].str.replace('\t', '', regex=False)
data_aupr.set_index('methods', inplace=True)

custom_colors = ['#71b3e7', '#e0dbf0', '#b3d4f1']  # 示例颜色

fig, ax = plt.subplots(2, 2, figsize=(20, 16))

data_auc.plot(kind='bar', stacked=True, ax=ax[0][0], color=custom_colors, edgecolor='black', linewidth=1.5)
ax[0][0].set_xlabel('')
ax[0][0].set_xticklabels(data_auc.index, size=14, rotation=20)
ax[0][0].set_ylabel('AUC score', size=16)
ax[0][0].tick_params(axis='y', labelsize=16)
ax[0][0].set_ylim(0.47, 1)  # 调整y轴范围
ax[0][0].legend(title='Ratio', fontsize=16)

data_aupr.plot(kind='bar', stacked=True, ax=ax[1][0], color=custom_colors, edgecolor='black', linewidth=1.5)
ax[1][0].set_xlabel('')
ax[1][0].set_xticklabels(data_aupr.index, size=14, rotation=20)
ax[1][0].set_ylabel('AUPR score', size=16)
ax[1][0].tick_params(axis='y', labelsize=16)
ax[1][0].set_ylim(0.2, 1)  # 调整y轴范围
ax[1][0].legend(title='Ratio', fontsize=16)

sns.set(font_scale=1.6)
sns.set_style("white")

methods = ['AnomalGRN', 'GNNLink', 'GENELink', 'GCN', 'GAT', 'GraphSAGE']
ratios = ['1:10', '1:20', '1:30', '1:40', '1:50']
auc_values = np.array([
    [0.8435, 0.6332, 0.6195, 0.6218, 0.6139],
    [0.4793, 0.4794, 0.4740, 0.4745, 0.4811],
    [0.6026, 0.5455, 0.4740, 0.3395, 0.4659],
    [0.5831, 0.5546, 0.5432, 0.5428, 0.5313],
    [0.5323, 0.5139, 0.5030, 0.4924, 0.4917],
    [0.5345, 0.5242, 0.4827, 0.4825, 0.4711],
]).T
aupr_values = np.array([
    [0.3945, 0.5238, 0.5161, 0.5122, 0.5087],
    [0.0813, 0.0423, 0.0282, 0.0213, 0.0174],
    [0.1020, 0.0607, 0.0299, 0.0179, 0.0181],
    [0.1443, 0.1117, 0.0135, 0.1134, 0.0912],
    [0.1224, 0.1263, 0.0354, 0.1154, 0.0843],
    [0.1719, 0.1161, 0.0953, 0.1745, 0.0554],
]).T

# Create a custom colormap
colors = ["#81c7c1", "#ffffff", "#fc7956"]
n_bins = 100
cmap_name = "custom_gradient"
cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

# AUC heatmap
ax1 = sns.heatmap(auc_values, ax=ax[0][1], annot=True, fmt=".4f", cmap=cm,
                  yticklabels=ratios, xticklabels=methods, linewidths=2, linecolor='black')
ax[0][1].set_title('AUC score', size=16)
ax[0][1].set_ylabel('Ratios', fontsize=16)
ax[0][1].tick_params(axis='x', labelsize=16)
ax[0][1].tick_params(axis='y', labelsize=16)
# Adding border
for _, spine in ax1.spines.items():
    spine.set_visible(True)
    spine.set_linewidth(1.5)

# AUPR heatmap
ax2 = sns.heatmap(aupr_values, ax=ax[1][1], annot=True, fmt=".4f", cmap=cm,
                  yticklabels=ratios, xticklabels=methods, linewidths=2, linecolor='black')
ax[1][1].set_title('AUPR score', size=16)
ax[1][1].set_ylabel('Ratios', fontsize=16)
ax[1][1].tick_params(axis='x', labelsize=16)
ax[1][1].tick_params(axis='y', labelsize=16)
# Adding border
for _, spine in ax2.spines.items():
    spine.set_visible(True)
    spine.set_linewidth(1.5)

plt.tight_layout()
plt.show()
