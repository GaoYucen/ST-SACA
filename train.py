import SACA
import GRC_ELG
import SACA_baseline
import JDRL_POMO
import matplotlib.pyplot as plt

def main(A=10, w=1):
    # 禁用 plt.show() 以免阻塞训练过程中的弹窗
    plt.show = lambda: None

    SACA_config = SACA.Config()
    GRC_ELG_config = GRC_ELG.Config()
    SACA_baseline_config = SACA_baseline.Config()
    JDRL_POMO_config = JDRL_POMO.Config()

    SACA_config.demand_fluctuation, SACA_config.demand_frequency = A, w
    GRC_ELG_config.demand_fluctuation, GRC_ELG_config.demand_frequency = A, w
    SACA_baseline_config.demand_fluctuation, SACA_baseline_config.demand_frequency = A, w
    JDRL_POMO_config.demand_fluctuation, JDRL_POMO_config.demand_frequency = A, w

    # JDRL_POMO 训练
    JDRL_POMO.train_jdrl(JDRL_POMO_config)

    # SACA 训练
    SACA.train_saca(SACA_config)

    # GRC_ELG 训练
    GRC_ELG.train_grc_elg(GRC_ELG_config)

    # SACA_baseline 训练
    SACA_baseline.train_saca(SACA_baseline_config)

if __name__ == "__main__":
    main()