import torch
import torch.optim as optim
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Model:
    def __init__(self, dqn, discount):
        self.dqn = dqn
        self.optimizer = optim.Adam(dqn.parameters())
        self.discount = discount
    
    def predict(self, state, action=None):
        if action is None:
            return self.dqn(state)
        return self.dqn(state).gather(1, action)
    
    def train(self, target, buffer, batch_size):
        if len(buffer) < batch_size:
            return
        exp = buffer.sample(batch_size)
        batch = Transition(*zip(*exp))
        
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        non_terminal_mask = torch.tensor([s is not None for s in batch.next_state], device=device)
        next_state_batch = torch.tensor([s for s in batch.next_state if s is not None], device=device)
        
        next_state_values = torch.zeros(batch_size, device=device)
        next_state_values[non_terminal_mask] = target.predict(next_state_batch, self.predict(next_state_batch).argmax(1))
        target_values = (next_state_values * self.d) + reward_batch
        
        state_action_values = self.predict(state_batch).gather(1, action_batch)
        loss = F.MSELoss(state_action_values, target_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.dqn.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
    def clone(self):
        dqn = type(self.dqn)()
        dqn.load_state_dict(self.dqn.state_dict())
        return Model(dqn, discount)
        
    def update(self, target):
        self.dqn.load_state_dict(target.state_dict())
        
    def save(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        dqn.save(path)