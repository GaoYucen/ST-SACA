import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
import time
import numpy as np

# 导入模型模块
try:
    from ablation_wo_route import train_ablation_wo_route
    import SACA
    import SACA_baseline
    import ablation_SACA_ORR
except ImportError as e:
    print(f"Import Error: {e}")

# ==========================================
# 1. 全局配置
# ==========================================
LOG_DIR = "log"
MAX_EPISODES = 80  # 分析截断点

# 定义实验配置列表
# 每个字典代表一个实验任务
EXPERIMENTS = [
    {
        "name": "ST-SACA",
        "module": SACA,
        "train_func_name": "train_saca",
        "pattern": "saca_training_log_*.csv"
    },
    {
        "name": "SACA",
        "module": SACA_baseline,
        "train_func_name": "train_saca",
        "pattern": "mlp_training_log_*.csv"
    },
    {
        "name": "SACA w/o Route",
        "module": None, # 特殊处理，直接调用函数
        "train_func": train_ablation_wo_route,
        "pattern": "ablation_log_wo_AttnRoute_*.csv"
    },
    {
        "name": "SACA w/o ORR",
        "module": ablation_SACA_ORR,
        "train_func_name": "train_saca",
        "pattern": "ablation_wo_ORR_*.csv"
    }
]

# ==========================================
# 2. 辅助函数
# ==========================================
def get_latest_file(pattern):
    """获取目录下符合模式的最新文件"""
    search_path = os.path.join(LOG_DIR, pattern)
    files = glob.glob(search_path)
    if not files:
        return None
    return max(files, key=os.path.getmtime)

# ==========================================
# 3. 实验执行模块
# ==========================================
def run_experiments():
    """
    依次执行所有消融实验。
    由于每个实验脚本内部已经定义了不同的日志文件名，这里不需要重命名。
    """
    print("Starting Ablation Experiments Sequence...")
    print("=" * 60)
    
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    for exp in EXPERIMENTS:
        print(f"\n>>> Running Experiment: {exp['name']}")
        
        try:
            if exp['name'] == "SACA w/o Route":
                # 直接调用导入的函数
                exp['train_func']()
            else:
                # 获取模块和配置类
                module = exp['module']
                config = module.Config()
                
                # 获取训练函数并执行
                train_func = getattr(module, exp['train_func_name'])
                train_func(config)
            
            # 简单的延时，确保文件系统时间戳差异
            time.sleep(2)
                
        except Exception as e:
            print(f"  [Error] Experiment failed: {e}")
            import traceback
            traceback.print_exc()
            
    print("\nAll experiments completed.")

# ==========================================
# 4. 数据分析模块
# ==========================================
def analyze_results():
    """
    读取日志文件，计算指标并输出表格。
    """
    print("\n" + "="*80)
    print("Ablation Study Analysis (Average over last 10 episodes)")
    print("="*80)
    
    metrics = ["total_reward", "revenue", "cost", "orr", "total_distance"]
    results = {}
    
    for exp in EXPERIMENTS:
        # 查找该实验对应的最新文件
        pattern = exp['pattern']
        latest_file = get_latest_file(pattern)
        
        if not latest_file:
            print(f"Warning: No data found for {exp['name']} (Pattern: {pattern})")
            continue
            
        try:
            df = pd.read_csv(latest_file)
            
            # 数据截断
            if len(df) > MAX_EPISODES:
                df = df.iloc[:MAX_EPISODES]
            
            if len(df) == 0:
                print(f"Warning: Empty data for {exp['name']}")
                continue

            # 计算最后10个episode的平均值
            last_10 = df.tail(10)
            exp_res = {}
            for m in metrics:
                if m in df.columns:
                    exp_res[m] = last_10[m].mean()
                else:
                    exp_res[m] = float('nan')
            
            results[exp['name']] = exp_res
            print(f"Loaded data for {exp['name']} from {os.path.basename(latest_file)}")
            
        except Exception as e:
            print(f"Error analyzing {latest_file}: {e}")

    # 生成表格
    if not results:
        print("No results to display.")
        return

    # 转置 DataFrame: 行=模型, 列=指标
    df_summary = pd.DataFrame(results).T
    
    # 确保列存在
    available_metrics = [m for m in metrics if m in df_summary.columns]
    df_summary = df_summary[available_metrics]
    
    # 格式化输出 (保留4位小数)
    pd.set_option('display.float_format', lambda x: '%.4f' % x)
    print("\nSummary Table:")
    print(df_summary)
    
    # 保存结果 (暂时注释)
    output_path = os.path.join(LOG_DIR, "ablation_study_summary.csv")
    # df_summary.to_csv(output_path)
    # print(f"\nSummary table saved to {output_path}")

# ==========================================
# 主入口
# ==========================================
if __name__ == "__main__":
    # 1. 运行实验 (如果需要重新跑数据，请取消注释)
    # run_experiments()
    
    # 2. 分析结果
    analyze_results()

