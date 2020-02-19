import matplotlib.pyplot as plt
import seaborn as sns


def plot_time_series(times, X, X_labels, noisy_X, noisy_X_labels, ax):
    plt.sca(ax)
    plt.plot(times, X)
    plt.plot(times, noisy_X, ".")
    plt.title("Time series of concentrations")
    return

def plot_derivatives(times, X_dot, X_dot_labels, splined_X_dot, splined_X_dot_labels, ax):
    plt.sca(ax)
    plt.plot(times, X_dot)
    plt.plot(times, splined_X_dot, "--")
    plt.title("Time series of derivatives")
    return



def spline_ground(X_dot_eq, splined_X_dot, ax):
    plt.sca(ax)
    plt.scatter(X_dot_eq[:, 0], splined_X_dot[:, 0] - X_dot_eq[:, 0], 1)
    plt.scatter(X_dot_eq[:, 1], splined_X_dot[:, 1] - X_dot_eq[:, 1], 1)
    plt.xlabel("X_dot", fontdict={"size": 20})
    plt.ylabel("splined_X_dot - X_dot", fontdict={"size": 20})
    plt.title("Spline vs ground-truth", fontdict={"size": 25})
    ax.axis('equal')
    return

""""
def arrow_plot(X, X_dot, ax):
    for specie_n in range(clean_X.shape[1]):
        color = ["red", "blue"][specie_n]
        col = ["pink", "yellow"][specie_n]
        for i in range(0, len(X_dot_eq[:, specie_n]), 5
                       ):
            ax.arrow(clean_times[i], clean_X[i, specie_n], 1, X_dot_eq[i, specie_n], width=0.125, color=color)
            ax.arrow(clean_times[i], splined_X[i, specie_n], 1, splined_X_dot[i, specie_n], width=0.125, color=col)
            # ax.annotate("", xy=(1, X_dot_eq[int(t / 0.001), int(specie_n)]), xytext=(t, clean_X[int(t / 0.001), int(specie_n)]),arrowprops=dict(arrowstyle="->"))
            # print(X_dot_eq[t, specie_n])

            # , clean_times[:-1], X_dot_eq[:])
    # plt.savefig("test.eps")"""