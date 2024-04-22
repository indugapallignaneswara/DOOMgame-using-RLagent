# DOOMgame-using-RLagent
DoomRL (Doom Roguelike) is a unique fusion of the classic first-person shooter, Doom, and the roguelike genre, presenting a challenging environment for AI agents to navigate. In DoomRL, players explore procedurally generated levels, battling hordes of monsters while managing resources and navigating treacherous terrain.



# Game Configuration for DoomRL Agent

Episode Start Time: 20 tics (after unholstering the gun)
Episode Timeout: 300 actions (tics)
Available Buttons:
Move Left
Move Right
Attack
Game Variables in State:
Ammo2
Mode: Player
Doom Skill: 5

Rewards and Penalties:
+101 reward for killing a monster
-5 penalty for missing a shot
-1 penalty for each step taken

To run the AI agent:

Install the necessary dependencies listed in the requirements file.
Train the agent using the provided training script.
Evaluate the trained agent's performance in the game environment.
