


# import matplotlib
# # Use a non-interactive backend to avoid GUI issues on headless systems.
# matplotlib.use("Agg")

# import matplotlib.pyplot as plt
# import numpy as np
# import os

# def plot_mde_comparison():
#     # ================= 数据准备 =================
#     timesteps = ['6h', '12h', '18h', '24h']
#     baseline_mde = np.array([34.65, 48.11, 91.26, 159.45])
#     ours_mde = np.array([27.82, 36.63, 76.23, 140.22]) 
#     reduction_rates = (baseline_mde - ours_mde) / baseline_mde * 100

#     # ================= 绘图设置 =================
#     plt.rcParams['font.family'] = 'serif'
#     plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
#     plt.rcParams['font.size'] = 14
    
#     fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
#     x = np.arange(len(timesteps))
#     width = 0.35
    
#     rects1 = ax.bar(x - width/2, baseline_mde, width, label='TC-Diffuser', color='#A9A9A9', edgecolor='black', alpha=0.8)
#     rects2 = ax.bar(x + width/2, ours_mde, width, label='TYPCD (Ours)', color='#4169E1', edgecolor='black', alpha=0.9)
    
#     ax.set_xlabel('Forecast Lead Time', fontsize=16, fontweight='bold')
#     ax.set_ylabel('Mean Distance Error (km)', fontsize=16, fontweight='bold')
#     ax.set_xticks(x)
#     ax.set_xticklabels(timesteps, fontsize=14)
#     ax.legend(fontsize=12, frameon=False)
    
#     ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.5)
#     ax.set_axisbelow(True)
    
#     # 稍微调高Y轴上限，给上方标注留出更多空间
#     ax.set_ylim(0, max(baseline_mde) * 1.25)
    
#     # ================= 标注设置 =================
    
#     # 1. 柱子顶部数值：改为保留两位小数 (.2f)
#     def autolabel_value(rects):
#         for rect in rects:
#             height = rect.get_height()
#             ax.annotate(f'{height:.2f}', # <--- 修改此处：保留两位小数
#                         xy=(rect.get_x() + rect.get_width() / 2, height),
#                         xytext=(0, 3), 
#                         textcoords="offset points",
#                         ha='center', va='bottom', fontsize=14, color='black')
    
#     autolabel_value(rects1)
#     autolabel_value(rects2)

#     # 2. 降低百分比：位置上移 (xytext=18) 以避免重叠
#     for i in range(len(x)):
#         h_ours = ours_mde[i]
#         rate = reduction_rates[i]
        
#         ax.annotate(f'↓{rate:.2f}%',
#                     xy=(x[i] + width/2, h_ours), 
#                     xytext=(5, 20), # 保持较高的位置
#                     textcoords="offset points",
#                     ha='center', va='bottom',
#                     fontsize=14, fontweight='bold', color='#D2222D') 

#     # ================= 保存图片 =================
#     output_dir = 'fig/response_plots'
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
        
#     save_path = os.path.join(output_dir, 'mde_reduction_final.png')
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     plt.close()
    
#     print(f"Chart saved to: {save_path}")

# if __name__ == "__main__":
#     plot_mde_comparison()