import torch


class Reinforce(torch.nn.Module):
    def __init__(self, num_inputs, nun_actions, hidden_size):
        super(Reinforce, self).__init__()
        self.linear1 = torch.nn.Linear(num_inputs, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, nun_actions)
        self.rewards, self.log_probs = [], []

    def forward(self, state):
        outputs = torch.nn.ReLU()(self.linear1(state))
        outputs = torch.nn.Softmax()(self.linear2(outputs))
        return outputs

    def get_action(self, inputs):
        """
        This function will be used to evaluate your policy.
        """
        action_probs = self.forward(inputs.float())
        dist = torch.distributions.Categorical(action_probs)  # same as multinomial
        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        return action.item(), action_log_prob


class ActorCritic(torch.nn.Module):
    def __init__(self, num_inputs, nun_actions, hidden_size):
        super(ActorCritic, self).__init__()
        self.linear1 = torch.nn.Linear(num_inputs, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, hidden_size*2)
        # actor's layer
        self.action_head = torch.nn.Linear(hidden_size*2, nun_actions)
        # critic's layer
        self.value_head = torch.nn.Linear(hidden_size*2, 2)

        self.rewards, self.log_probs = [], []

    def forward(self, state):
        outputs = torch.nn.ReLU()(self.linear1(state))
        outputs = torch.nn.ReLU()(self.linear2(outputs))

        # actor: choose action to take from state input
        action_prob = torch.nn.Softmax()(self.action_head(outputs))

        # critic: evaluates being in the state input
        state_value = self.value_head(outputs)
        return action_prob, state_value

    def get_action(self, inputs):
        """
        This function will be used to evaluate your policy.
        """
        action_prob, state_value = self.forward(inputs.float())

        # sample action distribution
        dist = torch.distributions.Categorical(action_prob)  # same as multinomial
        action = dist.sample()
        action_log_prob = dist.log_prob(action)

        return action.item(), action_log_prob, state_value
