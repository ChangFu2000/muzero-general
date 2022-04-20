import numpy as np

class Defender(object):
    def __init__(self, config):
        self.config = config

    def act(self, state):
        # find minimal status
        min_status = 1e5
        candidates = list()
        for idx, status in enumerate(state):
            if status < 0:
                pass
            elif status < min_status:
                min_status = status
                candidates = [idx]
            elif status == min_status:
                candidates.append(idx)
            else:
                pass

        action = np.random.choice(candidates)
        return action

if __name__ == '__main__':
    defender = Defender(None)
    state = np.array([-1, -1, 2, 1, 2, 3, 1, 2, 3])
    print(defender.act(state))