import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import time
from zmqRemoteApi import RemoteAPIClient

class DeepQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DeepQNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim, learning_rate=0.001):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.memory = deque(maxlen=2000)
        self.gamma = 0.9    
        self.epsilon = 1.0   
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
    
        self.model = DeepQNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_dim)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()
    
    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return 0 

        minibatch = random.sample(self.memory, batch_size)
        
        states = torch.FloatTensor([m[0] for m in minibatch])
        actions = torch.LongTensor([m[1] for m in minibatch])
        rewards = torch.FloatTensor([m[2] for m in minibatch])
        next_states = torch.FloatTensor([m[3] for m in minibatch])
        dones = torch.FloatTensor([m[4] for m in minibatch])
    
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.model(next_states).max(1)[0]
   
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = self.loss_fn(current_q_values, target_q_values.detach())
 
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item() 

    
    def save_model(self, filepath):
        torch.save(self.model.state_dict(), filepath)
    
    def load_model(self, filepath):
        self.model.load_state_dict(torch.load(filepath))

def get_state(positions):
    def categorize_grid(grid_positions):
        blue_count = sum(1 for idx in grid_positions if idx < 9)
        total_count = len(grid_positions)
        
        if total_count == 0:
            return 0
        
        blue_ratio = blue_count / total_count
        
        if 0.4 <= blue_ratio <= 0.6:
            return 2
        elif 0 < blue_ratio < 1:
            return 1
        return 0
    grid_groups = [[] for _ in range(4)]
    for idx, grid in enumerate(positions):
        grid_groups[grid].append(idx)
    
    return tuple(categorize_grid(grid) for grid in grid_groups)

def calculate_mixing_reward(prev_positions, current_positions, step_no):
    prev_state = get_state(prev_positions)
    current_state = get_state(current_positions)
    goal_state = (2, 2, 2, 2)
    
    reward_components = {
        'goal_completion': 0,
        'time_bonus': 0,
        'progression': 0,
        'regression': 0,
        'state_bonus': 0,
        'step_penalty': 0
    }
    
    if current_state == goal_state:
        reward_components['goal_completion'] = 10
        reward_components['time_bonus'] = max(5 - step_no // 10, 0)
    
    progression_count = sum(cur > prev for cur, prev in zip(current_state, prev_state))
    regression_count = sum(cur < prev for cur, prev in zip(current_state, prev_state))
    
    reward_components['progression'] = progression_count * 1
    reward_components['regression'] = -regression_count * 2

    for state_value in current_state:
        if state_value == 2:
            reward_components['state_bonus'] += 1
        elif state_value == 1:
            reward_components['state_bonus'] += 0.5

    reward_components['step_penalty'] = -step_no * 0.1

    total_reward = sum(reward_components.values())
    
    return max(total_reward, 0)

def train_dqn(simulation_class, episodes=100, max_steps=50):
    
    MOVEMENT_DIRECTIONS = ['Up', 'Down', 'Left', 'Right']
    STATE_DIMENSIONS = 4
    ACTION_DIMENSIONS = 4
    LOG_FILE_PATH = "dqn_rewards_log.txt"
    MODEL_SAVE_PATH = 'dqn_mixing_model.pth'

    agent = DQNAgent(STATE_DIMENSIONS, ACTION_DIMENSIONS)
    episode_rewards = []
    with open(LOG_FILE_PATH, "w") as log_file:
        log_file.write("episode_no\treward\tTemporal Difference Error\n")
        for episode_index in range(episodes):
            env = simulation_class()
            
            try:
                initial_positions = env.getObjectsPositions()
                current_state = get_state(initial_positions)
                cumulative_reward = 0
                episode_complete = False
                temporal_difference_loss = 0
                for step in range(max_steps):
                    action = agent.act(list(current_state))
                    selected_direction = MOVEMENT_DIRECTIONS[action]

                    env.action(direction=selected_direction)
 
                    new_positions = env.getObjectsPositions()
                    next_state = get_state(new_positions)

                    step_reward = calculate_mixing_reward(
                        initial_positions, 
                        new_positions, 
                        step
                    )
                    cumulative_reward += step_reward
  
                    agent.remember(
                        list(current_state), 
                        action, 
                        step_reward, 
                        list(next_state), 
                        episode_complete
                    )
                    temporal_difference_loss = agent.replay()

                    current_state = next_state
                    initial_positions = new_positions
     
                    if next_state == (2, 2, 2, 2):
                        break

                episode_rewards.append(cumulative_reward)
                log_file.write(
                    f"{episode_index + 1}\t"
                    f"{cumulative_reward:.2f}\t"
                    f"{temporal_difference_loss:.6f}\n"
                )
                print(
                    f"Episode {episode_index+1}: "
                    f"Total Reward = {cumulative_reward}, "
                    f"Final State = {current_state}"
                )
            
            finally:
                env.stopSim()

    agent.save_model(MODEL_SAVE_PATH)
    
    return agent, episode_rewards



class Simulation():
    def __init__(self, sim_port=23000):
        self.sim_port = sim_port
        self.directions = ['Up', 'Down', 'Left', 'Right']
        self.initializeSim()

    def initializeSim(self):
        self.client = RemoteAPIClient('localhost', port=self.sim_port)
        self.client.setStepping(True)
        self.sim = self.client.getObject('sim')
        self.defaultIdleFps = self.sim.getInt32Param(self.sim.intparam_idle_fps)
        self.sim.setInt32Param(self.sim.intparam_idle_fps, 0)

        self.getObjectHandles()
        self.sim.startSimulation()
        self.dropObjects()
        self.getObjectsInBoxHandles()

    def getObjectHandles(self):
        self.tableHandle = self.sim.getObject('/Table')
        self.boxHandle = self.sim.getObject('/Table/Box')

    def dropObjects(self):
        self.blocks = 18
        frictionCube = 0.06
        frictionCup = 0.8
        blockLength = 0.016
        massOfBlock = 14.375e-03

        self.scriptHandle = self.sim.getScript(self.sim.scripttype_childscript, self.tableHandle)
        self.client.step()
        retInts, retFloats, retStrings = self.sim.callScriptFunction('setNumberOfBlocks', self.scriptHandle, [self.blocks], [massOfBlock, blockLength, frictionCube, frictionCup], ['cylinder'])

        print('Wait until blocks finish dropping')
        while True:
            self.client.step()
            signalValue = self.sim.getFloatSignal('toPython')
            if signalValue == 99:
                loop = 20
                while loop > 0:
                    self.client.step()
                    loop -= 1
                break

    def getObjectsInBoxHandles(self):
        self.object_shapes_handles = []
        self.obj_type = "Cylinder"
        for obj_idx in range(self.blocks):
            obj_handle = self.sim.getObjectHandle(f'{self.obj_type}{obj_idx}')
            self.object_shapes_handles.append(obj_handle)

    def getObjectsPositions(self):
        pos_step = []
        box_position = self.sim.getObjectPosition(self.boxHandle, self.sim.handle_world)
        box_position_x, box_position_y = box_position[:2]
        for obj_handle in self.object_shapes_handles:
            
            obj_position = self.sim.getObjectPosition(obj_handle, self.sim.handle_world)
            obj_position_x, obj_position_y = obj_position[:2]
            if obj_position_x >= box_position_x and obj_position_y >= box_position_y:
                pos_step.append(0)
            elif obj_position_x < box_position_x and obj_position_y >= box_position_y:
                pos_step.append(1)
            elif obj_position_x < box_position_x and obj_position_y < box_position_y:
                pos_step.append(2)
            elif obj_position_x >= box_position_x and obj_position_y < box_position_y:
                pos_step.append(3)
        return pos_step

    def action(self, direction=None):
        if direction not in self.directions:
            print(f'Direction: {direction} invalid, please choose one from {self.directions}')
            return
        box_position = self.sim.getObjectPosition(self.boxHandle, self.sim.handle_world)
        _box_position = box_position
        span = 0.02
        steps = 5
        if direction == 'Up':
            idx = 1
            dirs = [1, -1]
        elif direction == 'Down':
            idx = 1
            dirs = [-1, 1]
        elif direction == 'Right':
            idx = 0
            dirs = [1, -1]
        elif direction == 'Left':
            idx = 0
            dirs = [-1, 1]

        for _dir in dirs:
            for _ in range(steps):
                _box_position[idx] += _dir * span / steps
                self.sim.setObjectPosition(self.boxHandle, self.sim.handle_world, _box_position)
                self.stepSim()

    def stepSim(self):
        self.client.step()

    def stopSim(self):
        self.sim.stopSimulation()
        time.sleep(1)



def main():
   train_dqn(Simulation)
   

if __name__ == '__main__':
    main()