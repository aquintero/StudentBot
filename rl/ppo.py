from keras.models import Model
from keras.layers import Input, Dense
from keras import backend as K
from keras.optimizers import Adam
from collections import namedtuple

Transition = namedtuple('Transition', ['state', 'policy', 'value', 'reward', 'next_state'])

class PPO:
    def __init__(self, model, horizon=256, discount=0.99, epochs=10):
        self.model = model
        self.horizon = horizon
        self.discount = discount
        self.epochs = epochs
        
    def init(self, state):
        self.state = state
        self.step = 0
        self.batch = []
        self.policy, self.value = self.model.predict(state)
        return self.policy.argmax()
        
    def next(reward, next_state):
        self.batch.append(Transition(self.state, self.policy, self.value, reward, next_state))
        
        if (self.step + 1) % self.horizon == 0:
            self.train()
            self.batch = []
        
        self.policy, self.value = self.model.predict(next_state)
        self.state = next_state
        self.step += 1
        
        return self.policy.argmax()
        
    def train():
        batch = Transition(*zip(*self.batch))
        for i in range(self.epochs):
            targets = np.zeros(self.horizon)
            target = 0
            for i in range(self.horizon - 1, -1, -1):
                target *= self.discount
                target += batch.value[i]
                targets[i] = ret
            advantages = targets - batch.values
            self.model.fit(batch.state, targets, advantages, batch.policy)
        
class PPOModel:
    def __init__(self, state_dim, action_dim, hidden_dim=256, hidden_layers=2, lr=0.0005, clip = 0.2, value_scale=0.1, entropy_scale=0.01):
        state = Input(shape=(state_dim,))
        advantage = Input(shape=(1,))
        
        x = state
        for i in range(hidden_layers):
            x = Dense(hidden_dim, activation='relu')(x)
        policy = Dense(action_dim, activation='softmax')(x)
        value = Dense(1)(x)
        
        self.model_train = Model(inputs=[state, advantage, old_policy], outputs=[policy, value])
        self.model_pred = Model(inputs=[state], outputs=[policy, value])
        
        def ppo_loss(advantage):
            def loss(y_true, y_pred):
                r = y_pred/(y_true + 1e-10)
                return -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1-clip, max_value=1+clip) * advantage)) + entropy_scale * (prob * K.log(prob + 1e-10))
            return loss
        
        self.model_train.compile(optimizer=Adam(lr=lr), loss=[ppo_loss(advantage=advantage), 'mse'] loss_weights=[1, value_scale])
        
    def predict(state):
        policy, value = self.model_pred.predict(state)
        return policy, value
    
    def fit(state, target, advantage, old_policy):
        self.model_train.train_on_batch([state, advantage, old_policy], [target])
        