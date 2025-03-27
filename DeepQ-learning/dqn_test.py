import sys
import time
import torch
import numpy as np
from zmqRemoteApi import RemoteAPIClient
from exec_environment import Simulation
import json

class DeepQNetwork(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DeepQNetwork, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, output_dim)
        )

    def forward(self, x):
        return self.network(x)

class DQNTestCase:
    def __init__(self, simulation_class, model_path='dqn_mixing_model.pth'):
        """
        Initialize test case with simulation class and model path
        """
        self.simulation_class = simulation_class
        self.model_path = model_path
        self.directions = ['Up', 'Down', 'Left', 'Right']
        self.load_model()

    def load_model(self):
        """
        Load the trained DQN model
        """
        state_dim = 4  
        action_dim = 4  

        self.model = DeepQNetwork(state_dim, action_dim)
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()  

    def get_state(self, positions):
        """
        Create a more explicit state representation that distinguishes object mixing.
        """
        blue = [0] * 4
        red = [0] * 4

        for index, grid in enumerate(positions):
            if index < 9:
                blue[grid] += 1
            else:
                red[grid] += 1
        
        state = []
        for grid_no in range(4):
            total_objects = blue[grid_no] + red[grid_no]
            blue_ratio = blue[grid_no] / total_objects if total_objects > 0 else 0
            
            # State encoding:
            # 0: Empty or not mixed
            # 1: Partially mixed (blue and red present)
            # 2: Fully mixed (near equal blue and red)
            if total_objects == 0:
                state.append(0)
            elif 0.4 <= blue_ratio <= 0.6:
                state.append(2)
            elif 0 < blue_ratio < 1:
                state.append(1)
            else:
                state.append(0)
        
        return state

    def run_test(self, num_episodes=100):
        """
        Run comprehensive test and log results
        """
        successful_mixes = 0
        total_times = []
        detailed_log = []
        test_start_time = time.time()

        for episode in range(num_episodes):
            success, duration, balanced_regions = self.run_episode()
            if success:
                successful_mixes += 1

            total_times.append(duration)
            status = "Goal Achieved" if success else "Goal Not Achieved"
            print(
                f"Episode {episode + 1}/{num_episodes}: {status}, "
                f"Balanced Regions: {balanced_regions}/4, Duration: {duration:.2f} seconds"
            )

            detailed_log.append({
                'episode': episode + 1,
                'success': success,
                'balanced_regions': balanced_regions,
                'duration': duration
            })
        total_test_duration = time.time() - test_start_time
        self.generate_report(successful_mixes, total_times, total_test_duration,num_episodes)

        return successful_mixes, total_times

    def run_episode(self, max_steps=50):
        env = self.simulation_class()

        try:
            start_time = time.time()
            prev_positions = env.getObjectsPositions()
            curr_state = self.get_state(prev_positions)

            for step in range(max_steps):
                state_tensor = torch.FloatTensor(curr_state).unsqueeze(0)
                with torch.no_grad():
                    q_values = self.model(state_tensor)
                action = torch.argmax(q_values).item()
                direction = self.directions[action]
                env.action(direction=direction)
                current_positions = env.getObjectsPositions()
                next_state = self.get_state(current_positions)
                if next_state == [2, 2, 2, 2]:
                    duration = time.time() - start_time
                    env.stopSim()
                    return True, duration, 4  
                curr_state = next_state

            balanced_regions = sum(1 for region in curr_state if region >= 1)
            duration = time.time() - start_time
            env.stopSim()
            return False, duration, balanced_regions

        except Exception as e:
            print(f"Error in test episode: {e}")
            env.stopSim()
            return False, 0, 0

    def generate_report(self, successful_mixes, total_times, total_test_duration,num_episodes):
        total_episodes = len(total_times)
        success_rate = (successful_mixes / total_episodes) * 100

        report = f"""
        DQN Test Report
        ---------------------
        Total Episodes: {total_episodes}
        Objects were mixed successfully: {successful_mixes} out of {num_episodes}
        Total Completion Time: {total_test_duration:.2f} seconds

        """

        with open('dqn_test_report.txt', 'w') as f:
            f.write(report)
        print(report)

def main():
    test_case = DQNTestCase(Simulation)
    test_case.run_test()

if __name__ == "__main__":
    main()
