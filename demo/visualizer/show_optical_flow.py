import numpy as np
import matplotlib.pyplot as plt
from sakuramoti.visualizer.optical_flow import make_colorwheel


def show_colorwheel():
    colorwheel = make_colorwheel() / 255.0

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})

    theta = np.linspace(0, 2 * np.pi, 55, endpoint=False)

    r = np.ones_like(theta)

    ax.scatter(theta, r, c=colorwheel.numpy(), s=300)

    ax.set_yticklabels([])
    ax.set_xticklabels([])

    plt.title("Color Wheel Visualization")
    plt.show()
