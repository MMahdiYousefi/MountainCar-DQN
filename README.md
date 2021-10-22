# Problem Description

The agent (Car) which is positioned between the hills has to reach the flag up the hill by choosing the actions described down bellow.
Now the agent can't just keep going forward until it gets to the flag, Because of the gravitaional pull (which is 0.0025 
in this case). So it has to keep going from left to right in order to build up the momentum to be able to reach the flag up the hill.

        Observation(state):
            0 --> Car position (min: -1.2    , max: 0.6)
            1 --> Car velocity (min: -0.07   , max: 0.07)

        Actions:
            0 --> Accelerate to the Left
            1 --> Don't accelerate
            2 --> Accelerate to the Right

        Rewards:
            0 if agent reaches the flag (position = 0.5)
           -1 if agent's position is less than 0.5
