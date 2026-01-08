import numpy as np
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
from itertools import count
from env import Env
import config


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

env = Env()
torch.manual_seed(0)
num_state = env.n_observations
num_action = env.node_num

Transition = namedtuple(
    'Transition',
    ['state', 'action', 'reward', 'a_log_prob', 'next_state', 'done', 'hidden', 'next_hidden']
)


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)


class NodeCNN(nn.Module):
    def __init__(self, node_feat_dim, out_dim):
        super(NodeCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32 * node_feat_dim, out_dim)

    def forward(self, x):
        # x: [batch, node_feat_dim]
        x = x.unsqueeze(1)  # [batch, 1, node_feat_dim]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Actor(nn.Module):
    def __init__(self, node_feat_dim, image_feat_dim, task_feat_dim):
        super(Actor, self).__init__()
        self.node_cnn = NodeCNN(node_feat_dim, 32)
        self.fc_cat = nn.Linear(32 + image_feat_dim + task_feat_dim, 128)
        self.gru = nn.GRU(128, 128, 2, batch_first=True)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 16)
        self.action_head = nn.Linear(16, num_action)

    def forward(self, node_feat, image_feat, task_feat, hidden):
        node_cnn_out = self.node_cnn(node_feat)
        x = torch.cat([node_cnn_out, image_feat, task_feat], dim=1)
        x = F.leaky_relu(self.fc_cat(x))
        x = x.unsqueeze(1)
        self.gru.flatten_parameters()
        x, h = self.gru(x, hidden)
        x = x.squeeze(1)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.action_head(x)
        action_prob = F.softmax(x, dim=1, dtype=torch.float)
        return action_prob, h


class Critic(nn.Module):
    def __init__(self, node_feat_dim, image_feat_dim, task_feat_dim):
        super(Critic, self).__init__()
        self.node_cnn = NodeCNN(node_feat_dim, 32)
        self.fc_cat = nn.Linear(32 + image_feat_dim + task_feat_dim, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 16)
        self.state_value = nn.Linear(16, num_action)

    def forward(self, node_feat, image_feat, task_feat):
        node_cnn_out = self.node_cnn(node_feat)
        x = torch.cat([node_cnn_out, image_feat, task_feat], dim=1)
        x = F.leaky_relu(self.fc_cat(x))
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        value = self.state_value(x)
        return value


class SAC():
    clip_param = 0.2
    max_grad_norm = 0.5
    buffer_capacity = 30000
    minimal_size = 1500
    batch_size = 3000

    def __init__(self, node_feat_dim, image_feat_dim, task_feat_dim):
        super(SAC, self).__init__()
        self.buffer = []
        self.counter = 0
        self.actor = Actor(node_feat_dim, image_feat_dim, task_feat_dim).to(device)
        self.critic_1 = Critic(node_feat_dim, image_feat_dim, task_feat_dim).to(device)
        self.critic_2 = Critic(node_feat_dim, image_feat_dim, task_feat_dim).to(device)
        self.target_critic_1 = Critic(node_feat_dim, image_feat_dim, task_feat_dim).to(device)
        self.target_critic_2 = Critic(node_feat_dim, image_feat_dim, task_feat_dim).to(device)
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=3e-4)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=3e-4)
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float, requires_grad=True)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=1e-4)
        self.target_entropy = -1
        self.gamma = 0.98
        self.tau = 0.005
        self.target_update = 100
        self.count = 0

    def get_initial_states(self):
        h_0 = torch.zeros((self.actor.gru.num_layers, 1, self.actor.gru.hidden_size), dtype=torch.float)
        h_0 = h_0.to(device)
        return h_0

    def calc_target(self, rewards, next_node_feat, next_image_feat, next_task_feat, dones, next_hiddens):
        next_probs, _ = self.actor(next_node_feat, next_image_feat, next_task_feat, next_hiddens)
        next_log_probs = torch.log(next_probs + 1e-8)
        entropy = -torch.sum(next_probs * next_log_probs, dim=1, keepdim=True)
        q1_value = self.target_critic_1(next_node_feat, next_image_feat, next_task_feat)
        q2_value = self.target_critic_2(next_node_feat, next_image_feat, next_task_feat)
        min_qvalue = torch.sum(next_probs * torch.min(q1_value, q2_value), dim=1, keepdim=True)
        next_value = min_qvalue + self.log_alpha.exp() * entropy
        td_target = rewards + self.gamma * next_value * (1 - dones)
        return td_target

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def select_action(self, env, task, node_feat, image_feat, task_feat, hidden):
        node_feat = torch.from_numpy(node_feat).float().unsqueeze(0).to(device)
        image_feat = torch.from_numpy(image_feat).float().unsqueeze(0).to(device)
        task_feat = torch.from_numpy(task_feat).float().unsqueeze(0).to(device)
        action_mask = torch.tensor([1] * env.node_num, dtype=torch.float).to(device)
        for idx, n in enumerate(env.node):
            flag = env.image[task.image_id].image_size > n.disk and n.image_list[task.image_id] == 0
            if n.type == "CPU":
                if flag or task.mem > n.mem or n.cpu_freq <= 0:
                    action_mask[idx] = 0
                    continue
            if n.type == "GPU":
                if flag or task.type == "CPU" or task.gpu_mem > n.gpu_mem:
                    action_mask[idx] = 0
                    continue
        if torch.all(action_mask == 0):
            return -1, 0, None
        with torch.no_grad():
            action_prob, h = self.actor(node_feat, image_feat, task_feat, hidden)
            action_prob = torch.mul(action_prob, action_mask)
        action_dist = torch.distributions.Categorical(action_prob)
        action = action_dist.sample().item()
        return action, 0, h

    def store_transition(self, transition):
        if len(self.buffer) < self.buffer_capacity:
            self.buffer.append(transition)
        else:
            self.buffer.pop(0)
            self.buffer.append(transition)
        return len(self.buffer) % self.minimal_size == 0

    def update(self):
        tiny_batch = random.sample(self.buffer, self.batch_size)
        node_feats = torch.tensor([t.state[0] for t in tiny_batch], dtype=torch.float).to(device)
        image_feats = torch.tensor([t.state[1] for t in tiny_batch], dtype=torch.float).to(device)
        task_feats = torch.tensor([t.state[2] for t in tiny_batch], dtype=torch.float).to(device)
        actions = torch.tensor([t.action for t in tiny_batch]).view(-1, 1).to(device)
        rewards = torch.tensor([t.reward for t in tiny_batch], dtype=torch.float).view(-1, 1).to(device)
        hiddens = torch.cat([t.hidden for t in tiny_batch], dim=1).to(device)
        next_hiddens = torch.cat([t.next_hidden for t in tiny_batch], dim=1).to(device)
        next_node_feats = torch.tensor([t.next_state[0] for t in tiny_batch], dtype=torch.float).to(device)
        next_image_feats = torch.tensor([t.next_state[1] for t in tiny_batch], dtype=torch.float).to(device)
        next_task_feats = torch.tensor([t.next_state[2] for t in tiny_batch], dtype=torch.float).to(device)
        dones = torch.tensor([t.done for t in tiny_batch], dtype=torch.float).view(-1, 1).to(device)
        td_target = self.calc_target(rewards, next_node_feats, next_image_feats, next_task_feats, dones, next_hiddens)
        critic_1_q_values = self.critic_1(node_feats, image_feats, task_feats).gather(1, actions)
        critic_1_loss = torch.mean(F.mse_loss(critic_1_q_values, td_target.detach()))
        critic_2_q_values = self.critic_2(node_feats, image_feats, task_feats).gather(1, actions)
        critic_2_loss = torch.mean(F.mse_loss(critic_2_q_values, td_target.detach()))
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()
        probs, _ = self.actor(node_feats, image_feats, task_feats, hiddens)
        log_probs = torch.log(probs + 1e-8)
        entropy = -torch.sum(probs * log_probs, dim=1, keepdim=True)
        q1_value = self.critic_1(node_feats, image_feats, task_feats)
        q2_value = self.critic_2(node_feats, image_feats, task_feats)
        min_qvalue = torch.sum(probs * torch.min(q1_value, q2_value), dim=1, keepdim=True)
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - min_qvalue)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        alpha_loss = torch.mean((entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()
        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)


def main():
    env = Env()
    env.reset()
    node_feat, image_feat, task_feat = env.get_obs(env.task[0], divide=True)
    node_feat_dim = len(node_feat)
    image_feat_dim = len(image_feat)
    task_feat_dim = len(task_feat)
    agent = SAC(node_feat_dim, image_feat_dim, task_feat_dim)

    total_times = []
    total_download_time = []
    record = []

    i_epoch = 0
    while i_epoch < config.epoch:
        ep_reward = []
        env.reset()
        cnt = 0
        for t in count():
            done, _, idx = env.env_up()
            if done:
                i_epoch += 1
                total_times.append(env.total_time)
                total_download_time.append(env.download_time)
                num_on_time = env.num_on_time
                total_task = env.total_task
                complete_ratio = num_on_time / total_task
                
                msg = 'Episode: {}, reward: {}, total_time: {}, complet_ratio: {}, donwload time: {}'.format(
                    i_epoch, round(np.mean(ep_reward), 3), env.total_time, complete_ratio, env.download_time)
                print(msg)
                with open('log_node_20/sac_cnn_gru.txt', 'a', encoding='utf-8') as f:
                    f.write(msg + '\n')

                record.append(
                    [i_epoch, round(np.mean(ep_reward), 3), env.total_time, complete_ratio, env.download_time])
                break

            temp = 0
            actions = []
            states = []
            action_probs = []
            t_on_n = [0] * env.node_num
            tasks = []
            hiddens = []
            next_hiddens = []
            hidden = agent.get_initial_states()
            while env.task and env.task[0].start_time == env.time:
                curr_task = env.task.pop(0)
                node_feat, image_feat, task_feat = env.get_obs(curr_task, divide=True)
                action, action_prob, next_hidden = agent.select_action(env, curr_task, node_feat, image_feat, task_feat,
                                                                       hidden)
                if action == -1:
                    curr_task.start_time += 1
                    continue
                temp += 1
                states.append((node_feat, image_feat, task_feat))
                tasks.append(curr_task)
                hiddens.append(hidden)
                next_hiddens.append(next_hidden)
                hidden = next_hidden
                actions.append(action)
                action_probs.append(action_prob)
            if not actions:
                continue
            for n_id in actions:
                t_on_n[n_id] += 1
            next_states, rewards, done, download_finish_time = env.step(tasks, actions, t_on_n)
            for i in range(0, temp):
                cnt += 1
                reward = rewards[i]
                ep_reward.append(reward)
                next_node_feat, next_image_feat, next_task_feat = env.get_obs(tasks[i], divide=True)
                trans = Transition(states[i], actions[i], reward, action_probs[i],
                                   (next_node_feat, next_image_feat, next_task_feat), done, hiddens[i], next_hiddens[i])
                agent.store_transition(trans)
                if cnt > 3000:
                    if cnt % 1000 == 0:
                        agent.update()