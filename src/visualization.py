import matplotlib.pyplot as plt
from config import FIGURES

def save_plot(name):
    plt.tight_layout()
    plt.savefig(FIGURES / f"{name}.png", dpi=300)
    plt.close()