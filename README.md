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

Rewards and Penalties in doom basic config. :
+101 reward for killing a monster
-5 penalty for missing a shot
-1 penalty for each step taken
Rewards and Penalties in doom defend config. :
+1 reward for killing a monster
-1 penalty for each step taken


---> !cd github & git clone https://github.com/mwydmuch/ViZDoom
 TO DOWNLOAD THE ENVIRONMENT CONFIGURATION FILES

Train the agent using the provided training script.
Evaluate the trained agent's performance in the game environment.


## License

This repository is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
