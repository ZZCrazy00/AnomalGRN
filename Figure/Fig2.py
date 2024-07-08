import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
fig, ax = plt.subplots(3, 4, figsize=(18, 12))

column_names1 = ["STRING", "AnomalGRN", "GNNLink", "GENELink", "GNE", "CNNC", "DeepDRIM", "GRN-Transformer", "GENIE3"]
# AUC
data1 = [
    ["hESC",    0.90, 0.92, 0.88, 0.73, 0.54, 0.61, 0.62, 0.64],
    ["hHEP",    0.86, 0.92, 0.86, 0.73, 0.53, 0.52, 0.64, 0.63],
    ["mDC",     0.92, 0.92, 0.86, 0.76, 0.51, 0.57, 0.61, 0.61],
    ["mESC",    0.92, 0.92, 0.87, 0.76, 0.50, 0.58, 0.59, 0.61],
    ["mHSC-E",  0.67, 0.91, 0.86, 0.63, 0.56, 0.50, 0.63, 0.70],
    ["mHSC-GM", 0.93, 0.89, 0.81, 0.69, 0.64, 0.58, 0.64, 0.75],
    ["mHSC-L",  0.77, 0.85, 0.81, 0.72, 0.50, 0.64, 0.60, 0.70]
]
data1 = pd.DataFrame(data1, columns=column_names1)
data1.set_index('STRING', inplace=True)
ax1 = sns.heatmap(data1, ax=ax[0][0], cmap="YlGnBu", annot=True, fmt=".2f", cbar=False,
                  xticklabels=False, yticklabels=True, annot_kws={"size": 14})
ax1.set_ylabel('STRING', fontsize=16)
ax1.set_title('AUC', size=16)
# AUPR
data2 = [
    [0.88,	0.20,	0.10,	0.05,	0.02,	0.04,	0.03,	0.04],
    [0.85,	0.22,	0.18,	0.05,	0.06,	0.05,	0.06,	0.04],
    [0.90,	0.29,	0.28,	0.09,	0.06,	0.04,	0.04,	0.05],
    [0.82,	0.18,	0.16,	0.05,	0.04,	0.05,	0.04,	0.05],
    [0.73,	0.22,	0.19,	0.04,	0.12,	0.09,	0.12,	0.12],
    [0.91,	0.35,	0.30,	0.07,	0.29,	0.26,	0.27,	0.27],
    [0.77,	0.32,	0.29,	0.09,	0.14,	0.32,	0.29,	0.33]
]
data2 = pd.DataFrame(data2)
ax2 = sns.heatmap(data2, ax=ax[0][1], cmap="YlGnBu", annot=True, fmt=".2f", cbar=False,
                  xticklabels=False, yticklabels=False, annot_kws={"size": 14})
ax2.set_title('AUPR', size=16)
# AUC
data3 = [
    [0.93,	0.91,	0.87,	0.75,	0.50,	0.58,	0.61,	0.64],
    [0.90,	0.89,	0.88,	0.75,	0.55,	0.50,	0.58,	0.63],
    [0.94,	0.92,	0.88,	0.78,	0.54,	0.50,	0.50,	0.60],
    [0.87,	0.90,	0.88,	0.79,	0.60,	0.64,	0.56,	0.61],
    [0.70,	0.89,	0.88,	0.65,	0.58,	0.60,	0.61,	0.70],
    [0.97,	0.88,	0.86,	0.69,	0.56,	0.61,	0.58,	0.75],
    [0.88,	0.83,	0.78,	0.71,	0.67,	0.76,	0.72,	0.71]
]
data3 = pd.DataFrame(data3)
ax3 = sns.heatmap(data3, ax=ax[0][2], cmap="YlGnBu", annot=True, fmt=".2f", cbar=False,
                  xticklabels=False, yticklabels=False, annot_kws={"size": 14})
ax3.set_title('AUC', size=16)
# AUPR
data4 = [
    [0.90,	0.19,	0.12,	0.05,	0.02,	0.05,	0.04,	0.03],
    [0.88,	0.18,	0.15,	0.05,	0.04,	0.03,	0.04,	0.02],
    [0.92,	0.27,	0.27,	0.08,	0.07,	0.05,	0.05,	0.06],
    [0.86,	0.11,	0.13,	0.05,	0.04,	0.05,	0.03,	0.02],
    [0.72,	0.23,	0.21,	0.04,	0.15,	0.13,	0.13,	0.12],
    [0.94,	0.31,	0.25,	0.06,	0.23,	0.26,	0.23,	0.25],
    [0.85,	0.31,	0.24,	0.08,	0.26,	0.30,	0.28,	0.32]
]
data4 = pd.DataFrame(data4)
ax4 = sns.heatmap(data4, ax=ax[0][3], cmap="YlGnBu", annot=True, fmt=".2f", cbar=False,
                  xticklabels=False, yticklabels=False, annot_kws={"size": 14})
ax4.set_title('AUPR', size=16)

column_names2 = ["Non-Specific", "AnomalGRN", "GNNLink", "GENELink", "GNE", "CNNC", "DeepDRIM", "GRN-Transformer", "GENIE3"]
# AUC
data5 = [
    ["hESC",    0.93,	0.74,	0.69,	0.63,	0.53,	0.50,	0.50,	0.50],
    ["hHEP",    0.91,	0.75,	0.69,	0.63,	0.50,	0.50,	0.50,	0.50],
    ["mDC",     0.91,	0.82,	0.75,	0.67,	0.62,	0.58,	0.52,	0.55],
    ["mESC",    0.86,	0.76,	0.71,	0.67,	0.61,	0.59,	0.58,	0.56],
    ["mHSC-E",  0.91,	0.76,	0.72,	0.56,	0.60,	0.54,	0.59,	0.64],
    ["mHSC-GM", 0.94,	0.76,	0.71,	0.60,	0.68,	0.75,	0.61,	0.71],
    ["mHSC-L",  0.94,	0.67,	0.64,	0.60,	0.56,	0.61,	0.65,	0.65]
]
data5 = pd.DataFrame(data5, columns=column_names2)
data5.set_index('Non-Specific', inplace=True)
ax5 = sns.heatmap(data5, ax=ax[1][0], cmap="YlGnBu", annot=True, fmt=".2f", cbar=False,
                  xticklabels=False, yticklabels=True, annot_kws={"size": 14})
ax5.set_ylabel('Non-Specific', fontsize=16)
# AUPR
data6 = [
    [0.90,	0.06,	0.04,	0.02,	0.02,	0.02,	0.02,	0.02],
    [0.88,	0.05,	0.03,	0.02,	0.02,	0.02,	0.01,	0.02],
    [0.87,	0.16,	0.13,	0.03,	0.04,	0.02,	0.02,	0.02],
    [0.84,	0.09,	0.05,	0.03,	0.02,	0.02,	0.02,	0.02],
    [0.89,	0.12,	0.12,	0.04,	0.06,	0.05,	0.05,	0.06],
    [0.93,	0.22,	0.19,	0.04,	0.11,	0.12,	0.12,	0.12],
    [0.90,	0.15,	0.12,	0.06,	0.14,	0.09,	0.11,	0.13]
]
data6 = pd.DataFrame(data6)
ax6 = sns.heatmap(data6, ax=ax[1][1], cmap="YlGnBu", annot=True, fmt=".2f", cbar=False,
                  xticklabels=False, yticklabels=False, annot_kws={"size": 14})
# AUC
data7 = [
    [0.94,	0.91,	0.87,	0.75,	0.50,	0.58,	0.61,	0.64],
    [0.93,	0.89,	0.88,	0.75,	0.55,	0.50,	0.58,	0.63],
    [0.94,	0.92,	0.88,	0.78,	0.54,	0.50,	0.50,	0.60],
    [0.88,	0.90,	0.88,	0.79,	0.60,	0.64,	0.56,	0.61],
    [0.93,	0.89,	0.88,	0.65,	0.58,	0.60,	0.61,	0.70],
    [0.95,	0.88,	0.86,	0.69,	0.56,	0.61,	0.58,	0.75],
    [0.93,	0.83,	0.78,	0.71,	0.67,	0.76,	0.72,	0.71]
]
data7 = pd.DataFrame(data7)
ax7 = sns.heatmap(data7, ax=ax[1][2], cmap="YlGnBu", annot=True, fmt=".2f", cbar=False,
                  xticklabels=False, yticklabels=False, annot_kws={"size": 14})
# AUPR
data8 = [
    [0.92,	0.04,	0.03,	0.02,	0.03,	0.02,	0.02,	0.01],
    [0.91,	0.05,	0.04,	0.02,	0.02,	0.02,	0.01,	0.01],
    [0.91,	0.15,	0.10,	0.02,	0.03,	0.03,	0.02,	0.02],
    [0.86,	0.08,	0.05,	0.02,	0.02,	0.02,	0.02,	0.02],
    [0.91,	0.16,	0.10,	0.02,	0.02,	0.06,	0.05,	0.05],
    [0.94,	0.23,	0.21,	0.05,	0.10,	0.15,	0.12,	0.12],
    [0.95,	0.12,	0.10,	0.05,	0.13,	0.13,	0.14,	0.12]
]
data8 = pd.DataFrame(data8)
ax8 = sns.heatmap(data8, ax=ax[1][3], cmap="YlGnBu", annot=True, fmt=".2f", cbar=False,
                  xticklabels=False, yticklabels=False, annot_kws={"size": 14})

column_names3 = ["Specific", "AnomalGRN", "GNNLink", "GENELink", "GNE", "CNNC", "DeepDRIM", "GRN-Transformer", "GENIE3"]
# AUC
data9 = [
    ["hESC",    0.99,	0.85,	0.79,	0.67,	0.68,	0.50,	0.51,	0.5],
    ["hHEP",    0.99,	0.82,	0.83,	0.80,	0.64,	0.52,	0.50,	0.54],
    ["mDC",     0.98,	0.70,	0.67,	0.52,	0.54,	0.50,	0.50,	0.5],
    ["mESC",    0.96,	0.84,	0.80,	0.81,	0.73,	0.51,	0.53,	0.5],
    ["mHSC-E",  0.98,	0.83,	0.79,	0.82,	0.67,	0.56,	0.64,	0.52],
    ["mHSC-GM", 0.98,	0.89,	0.84,	0.83,	0.69,	0.64,	0.50,	0.53],
    ["mHSC-L",  0.98,	0.84,	0.81,	0.77,	0.67,	0.58,	0.64,	0.52]
]
data9 = pd.DataFrame(data9, columns=column_names3)
data9.set_index('Specific', inplace=True)
ax9 = sns.heatmap(data9, ax=ax[2][0], cmap="YlGnBu", annot=True, fmt=".2f", cbar=False,
                  xticklabels=False, yticklabels=True, annot_kws={"size": 14})
ax9.set_ylabel('Specific', fontsize=16)
# AUPR
data10 = [
    [0.98,	0.52,	0.48,	0.34,	0.25,	0.13,	0.15,	0.15],
    [0.98,	0.75,	0.69,	0.65,	0.46,	0.39,	0.35,	0.33],
    [0.98,	0.25,	0.11,	0.06,	0.06,	0.06,	0.06,	0.05],
    [0.93,	0.76,	0.75,	0.64,	0.48,	0.46,	0.49,	0.31],
    [0.98,	0.88,	0.88,	0.80,	0.74,	0.76,	0.71,	0.56],
    [0.99,	0.89,	0.88,	0.78,	0.68,	0.64,	0.66,	0.53],
    [0.98,	0.85,	0.81,	0.70,	0.64,	0.59,	0.64,	0.5]
]
data10 = pd.DataFrame(data10)
ax10 = sns.heatmap(data10, ax=ax[2][1], cmap="YlGnBu", annot=True, fmt=".2f", cbar=False,
                   xticklabels=False, yticklabels=False, annot_kws={"size": 14})
# AUC
data11 = [
    [0.99,	0.80,	0.78,	0.68,	0.72,	0.56,	0.67,	0.5],
    [0.99,	0.84,	0.85,	0.81,	0.66,	0.63,	0.57,	0.54],
    [0.99,	0.78,	0.73,	0.52,	0.56,	0.50,	0.50,	0.52],
    [0.93,	0.84,	0.82,	0.82,	0.73,	0.62,	0.59,	0.5],
    [0.99,	0.87,	0.84,	0.84,	0.72,	0.50,	0.53,	0.5],
    [0.99,	0.92,	0.87,	0.84,	0.69,	0.66,	0.58,	0.51],
    [0.99,	0.86,	0.81,	0.77,	0.62,	0.57,	0.50,	0.52]
]
data11 = pd.DataFrame(data11)
ax11 = sns.heatmap(data11, ax=ax[2][2], cmap="YlGnBu", annot=True, fmt=".2f", cbar=False,
                   xticklabels=False, yticklabels=False, annot_kws={"size": 14})
# AUPR
data12 = [
    [0.99,	0.51,	0.50,	0.34,	0.27,	0.19,	0.16,	0.15],
    [0.98,	0.78,	0.70,	0.66,	0.49,	0.46,	0.53,	0.38],
    [0.98,	0.21,	0.12,	0.05,	0.05,	0.06,	0.05,	0.05],
    [0.90,	0.78,	0.75,	0.65,	0.50,	0.46,	0.51,	0.31],
    [0.99,	0.93,	0.89,	0.81,	0.77,	0.73,	0.69,	0.54],
    [0.98,	0.93,	0.91,	0.81,	0.73,	0.64,	0.61,	0.53],
    [0.99,	0.86,	0.82,	0.68,	0.56,	0.48,	0.52,	0.48]
]
data12 = pd.DataFrame(data12)
ax12 = sns.heatmap(data12, ax=ax[2][3], cmap="YlGnBu", annot=True, fmt=".2f", cbar=False,
                   xticklabels=False, yticklabels=False, annot_kws={"size": 14})

plt.tight_layout()
plt.show()

# fig, ax = plt.subplots(1, 4, figsize=(18, 2))
#
# column_names4 = ["Lofgof", "AnomalGRN", "GNNLink", "GENELink", "GNE", "CNNC", "DeepDRIM", "GRN-Transformer", "GENIE3"]
# # AUC
# data13 = [["mESC",    0.98,	0.79,	0.74,	0.71,	0.54,	0.59,	0.52,	0.59]]
# data13 = pd.DataFrame(data13, columns=column_names4)
# data13.set_index('Lofgof', inplace=True)
# ax13 = sns.heatmap(data13, ax=ax[0], cmap="YlGnBu", annot=True, fmt=".2f", cbar=False,
#                    xticklabels=True, yticklabels=True, annot_kws={"size": 14})
# ax13.set_ylabel('Lofgof', fontsize=16)
# ax13.set_yticklabels(labels=['mESC 500'], rotation=0)
# # AUPR
# data14 = [[0.97,	0.50,	0.45,	0.27,	0.28,	0.34,	0.35,	0.31]]
# data14 = pd.DataFrame(data14, columns=["AnomalGRN", "GNNLink", "GENELink", "GNE", "CNNC", "DeepDRIM", "GRN-Transformer", "GENIE3"])
# ax14 = sns.heatmap(data14, ax=ax[1], cmap="YlGnBu", annot=True, fmt=".2f", cbar=False,
#                    xticklabels=True, yticklabels=False, annot_kws={"size": 14})
# # AUC
# data15 = [[0.98,	0.81,	0.78,	0.72,	0.67,	0.64,	0.57,	0.59]]
# data15 = pd.DataFrame(data15, columns=["AnomalGRN", "GNNLink", "GENELink", "GNE", "CNNC", "DeepDRIM", "GRN-Transformer", "GENIE3"])
# ax15 = sns.heatmap(data15, ax=ax[2], cmap="YlGnBu", annot=True, fmt=".2f", cbar=False,
#                    xticklabels=True, yticklabels=False, annot_kws={"size": 14})
# # AUPR
# data16 = [[0.98,	0.49,	0.45,	0.27,	0.34,	0.35,	0.31,	0.3]]
# data16 = pd.DataFrame(data16, columns=["AnomalGRN", "GNNLink", "GENELink", "GNE", "CNNC", "DeepDRIM", "GRN-Transformer", "GENIE3"])
# ax16 = sns.heatmap(data16, ax=ax[3], cmap="YlGnBu", annot=True, fmt=".2f", cbar=False,
#                    xticklabels=True, yticklabels=False, annot_kws={"size": 14})
#
# plt.tight_layout()
# plt.show()
