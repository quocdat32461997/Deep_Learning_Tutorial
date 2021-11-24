import torch


class Reinforce(torch.nn.Module):
    def __init__(self, num_inputs, nun_actions, hidden_size):
        super(Reinforce, self).__init__()
        self.linear1 = torch.nn.Linear(num_inputs, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, nun_actions)

    def forward(self, state):
        outputs = torch.nn.ReLU(self.linear1(state))
        outputs = torch.nn.Softmax(self.linear2(outputs))
        return outputs

    def get_action(self, inputs):
        """
        This function will be used to evaluate your policy.
        """
        action_probs = self.forward(inputs)
        dist = torch.distributions.Categorical(action_probs)  # same as multinomial
        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        return action.item(), action_log_prob


class ActorCritic(torch.nn.Module):
    def __init__(self, episode):
        super(ActorCritic, self).__init__()
        pass

    def forward(self):
        pass