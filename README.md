# Starcraft II AI bots in pure zig

The goal of this project is not to be purely competitive in Starcraft II bot leagues. If that happens, cool.
I like Starcraft II and reinforcement learning, and I want to learn zig. I also am using this as an opportunity to learn how reinforcement learning
concepts work at a low level (outside of frameworks like `gymnasium` in python), so I will resist the urge to pull in external ML libraries.
Fortunately, at the moment, it seems that AI development in zig is in its very early stages, so there's not a lot to pull in even if I get
frustrated and want to.

## Approach

- Use the Starcraft II protobuf client to launch/interact with games.
- Allow for two scenarios of bot develompent
  - Bot vs Bot - need to launch two clients, then have them join the same game
  - Bot vs builtin AI - need to launch one client, and create/join game with correct settings
- Build AI structs allowing for the creation of neural network architectures
- Build loss functions and backward propagation tools
- Build optimizer - always wanted to build my own, so likely won't be following the standard literature of some form of stochastic gradient descent. I think that's dumb, but I'm sure reality will humble me at some point.
- Build example deep policy network and reward model
- Train bot
- Persist training results and artifacts locally - maybe I'll upload model artifacts to some cloud repo later.
- See if I can beat my creation 😛

## Contributing

If someone else happens to look at this repo, 👋, hello! Happy to collaborate on concepts once the broader framework is built, so open a PR and I'll try to get to it at some point.
