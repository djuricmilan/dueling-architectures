class TargetDqnUpdater:
    def __init__(self, main_dqn, target_dqn):
        self.main_dqn = main_dqn
        self.target_dqn = target_dqn

    def update_target_network(self):
        self.target_dqn.set_weights(self.main_dqn.get_weights())