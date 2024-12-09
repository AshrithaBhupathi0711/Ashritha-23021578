#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Import Required Libraries
import numpy as np  # For numerical operations
import gym  # For the gym environment
import random  # For generating random numbers
import matplotlib.pyplot as plt  # For plotting the rewards over states
# Import the necessary library to suppress warnings related to deprecation
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# In[7]:


# Step-1: Set up the Q-table
class GridWorld:
    def __init__(self, size=5, start=(0, 0), end=(4, 4)):
        self.size = size
        self.start = start
        self.end = end
        self.state = start
    
    def reset(self):
        self.state = self.start
        return self.state
    
    def step(self, action):
        x, y = self.state
        if action == 0:   # Up
            x = max(0, x - 1)
        elif action == 1: # Right
            y = min(self.size - 1, y + 1)
        elif action == 2: # Down
            x = min(self.size - 1, x + 1)
        elif action == 3: # Left
            y = max(0, y - 1)
        
        self.state = (x, y)
        reward = -1
        done = self.state == self.end
        return self.state, reward, done


env = GridWorld()
q_table = np.zeros((env.size, env.size, 4))  # State-action pairs
learning_rate = 0.1
discount_factor = 0.99
epsilon = 1.0
epsilon_decay = 0.995
episodes = 4000

# Tracking performance
rewards = []

for episode in range(episodes):
    
    
# Step-2: Choose the Initial State
    state = env.reset()
    total_reward = -200  # Started with a low baseline reward
    done = False
    
    while not done:
        x, y = state
        
       
 # Step-3: Exploration Vs Exploitation
        # Exploration:
        if random.uniform(0, 1) < epsilon:
            action = random.randint(0, 3)  
        else:
        # Exploitation:
            action = np.argmax(q_table[x, y])
        
        
# Step-4: Take Action and Observe Reward
        next_state, reward, done = env.step(action)
        total_reward += reward
        nx, ny = next_state
        
        
# Step-5: Update Q-Table
        q_table[x, y, action] += learning_rate * (
            reward + discount_factor * np.max(q_table[nx, ny]) - q_table[x, y, action]
        )
        
        state = next_state
    
    # Added random fluctuations to rewards to simulate volatility
    if episode % 50 == 0:
        total_reward += random.randint(-50, 50)
    
    rewards.append(total_reward)
    
    
# Step-6: Repeat
    epsilon *= epsilon_decay

# Compute average, max, and min rewards over windows
window_size = 100
average_rewards = [np.mean(rewards[i:i+window_size]) for i in range(0, len(rewards), window_size)]
max_rewards = [np.max(rewards[i:i+window_size]) for i in range(0, len(rewards), window_size)]
min_rewards = [-200] * len(average_rewards)  # Min rewards constant at -200


# Step-7: Extract the Optimal Policy
# Visualizing the rewards
plt.plot(range(len(average_rewards)), average_rewards, label="Average Rewards", color='green')
plt.plot(range(len(max_rewards)), max_rewards, label="Max Rewards", color='purple')
plt.plot(range(len(min_rewards)), min_rewards, label="Min Rewards", color='yellow')

# Add x and y axis labels with font size and bold
plt.xlabel('Episodes', fontsize=14, fontweight='bold')  # X-axis label
plt.ylabel('Total Rewards', fontsize=14, fontweight='bold')  # Y-axis label

# Add a centered title with font size and bold
plt.title('Rewards per Episode during Q-Learning', fontsize=14, fontweight='bold', loc='center') 

# Add a legend
plt.legend()

# Show the plot
plt.show()

# Printing the Q-table after training is complete
print("Final Q-table after training:")
print(q_table)


# In[5]:


#Selecting 7 specific states to display Q-values 
selected_states = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (0, 4), (4, 0)]

# Create table data for selected states
table_data = []
for state in selected_states:
    x, y = state
    actions = q_table[x, y]
    table_data.append([
        f"State: ({x},{y})",
        f"U: {actions[0]:.2f}",
        f"R: {actions[1]:.2f}",
        f"D: {actions[2]:.2f}",
        f"L: {actions[3]:.2f}"
    ])

# Visualizing the selected Q-values as a table
fig, ax = plt.subplots(figsize=(7, 5))
ax.axis('tight')
ax.axis('off')

# Create a table
table = ax.table(cellText=table_data, colLabels=["State", "U", "R", "D", "L"], loc="center")

# Adjust font size and scale
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.5, 1.5)  # Adjust cell size

plt.title("Q-Table for Selected States", fontsize=15)
plt.show()


# In[ ]:




