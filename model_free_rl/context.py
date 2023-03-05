class Context:
    """State context:

       S: state t
       A: action t
       R: reward
       S: state t + 1
       A: action t + 1
    """

    def __init__(self, action, state, next_state, reward, next_action=None):
        self.action = action
        self.state = state

        self.next_action = next_action
        self.next_state = next_state

        self.reward = reward

    def __str__(self):
        return ('(' + \
                f'action={self.action}, ' + \
                f'state={self.state}, ' + \
                f'next_state={self.next_state}, ' + \
                f'reward={self.reward}' + \
                ')')
