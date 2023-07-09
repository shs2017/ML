import matplotlib.pyplot as plt
import numpy as np

min_player, max_player = (0, 22)
min_dealer, max_dealer = (0, 12)

def plot_blackjack_axis(policy: np.ndarray, axis: plt.axis, title: str):
    axis.set_title(title)
    axis.set_xlabel('Visible Dealer Hand')
    axis.set_ylabel('Player Hand')
    axis.set_xticks(range(min_dealer, max_dealer))
    axis.set_yticks(range(min_player, max_player))
    axis.imshow(policy, cmap='summer', origin='lower')

def plot_blackjack_policy(policy: np.ndarray):
    no_usable_ace_policy = policy[min_player : max_player,
                                  min_dealer : max_dealer,
                                  0]
    usable_ace_policy = policy[min_player : max_player,
                               min_dealer : max_dealer,
                               1]

    fig, axes = plt.subplots(ncols=2)
    ax1, ax2 = axes
    plot_blackjack_axis(policy=no_usable_ace_policy,
                        axis=ax1,
                        title='No Usable Ace Policy')
    plot_blackjack_axis(policy=usable_ace_policy,
                        axis=ax2,
                        title='No Usable Ace Policy')
    plt.show()
