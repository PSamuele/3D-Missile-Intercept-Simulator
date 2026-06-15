# [cite_start]3D Missile Intercept Simulator Report [cite: 1]

## [cite_start]1.0 Executive Summary [cite: 2]
[cite_start]This document outlines the architecture of a custom 3D Intercept Simulator developed in Python[cite: 3]. [cite_start]The main purpose of this tool is to test missile guidance algorithms[cite: 4]. [cite_start]Instead of using basic, hardcoded trajectories, the simulation forces the target to evade using realistic flight mechanics and randomized maneuvers[cite: 5]. [cite_start]This setup allows for a practical validation of True Proportional Navigation (TPN) against a dynamically reacting target[cite: 6].

---

## [cite_start]2.0 Target Kinematics and Structural Limits [cite: 7]
[cite_start]For the simulation to be valid, the target aircraft cannot perform maneuvers that violate physics[cite: 8]. [cite_start]The target is modeled as a point-mass, but its turning capability is strictly linked to its structural load factor ($nz_{max}$)[cite: 9]. [cite_start]In a coordinated turn, the maximum turn rate ($\omega_{max}$) is constrained by how much G-force the airframe can handle before structural failure[cite: 10]. [cite_start]The simulator calculates this limit dynamically using the following relationship[cite: 11]:

[cite_start]$$\omega_{max} = \min \left( \text{heading\_rate\_limit}, \frac{(nz_{max} - 1) \cdot g}{V} \right)$$ [cite: 12]

[cite_start]It is important to note that this equation is a conscious, first-order approximation[cite: 13]. [cite_start]It was chosen to keep computational costs low while still providing a highly realistic constraint: as the target flies faster (higher V), its ability to turn sharply decreases[cite: 14]. [cite_start]This effectively models a realistic maneuverability envelope[cite: 15].

---

## [cite_start]3.0 Interceptor Dynamics and TPN Guidance [cite: 16]
[cite_start]The interceptor uses True Proportional Navigation (TPN) to reach the target[cite: 17]. [cite_start]TPN works by predicting where the target is going, rather than pointing at where it is right now[cite: 18]. [cite_start]It does this by reacting to the rotation rate of the Line-Of-Sight (LOS) vector[cite: 19]:

[cite_start]$$\vec{a}_{cmd} = N' \cdot V_c \cdot (\vec{\omega}_{LOS} \times \hat{r})$$ [cite: 20]

[cite_start]In practical terms, N' (the Navigation Ratio) acts as an aggressiveness multiplier[cite: 21]. [cite_start]If the target moves and the LOS shifts by 1 degree, a missile with N'=4 will command a turn 4 times as sharp to "pull lead" and establish a collision course[cite: 22]. [cite_start]If N' is too low, the missile just chases the target's tail[cite: 23]. [cite_start]If N' is too high, it overreacts to every minor movement and wastes its kinetic energy[cite: 24]. 

[cite_start]Additionally, during the unpowered coast phase, the code adds a continuous [0, 0, +g] upward acceleration command[cite: 25]. [cite_start]This gravitational compensation prevents the missile from dropping under its own weight, which would otherwise cause a false rotation in the LOS calculation and waste steering energy[cite: 26].

---

## [cite_start]4.0 Random Evasion and Integration [cite: 27]
[cite_start]To thoroughly test the guidance law, the target's maneuvers are triggered by a Poisson process[cite: 28]. [cite_start]In concrete terms, this means evasive maneuvers happen at random, unpredictable intervals rather than on a fixed timer[cite: 29]. [cite_start]The code sets an average wait time (e.g., 20 seconds), but the actual time between turns fluctuates constantly[cite: 30]. [cite_start]This accurately simulates a human pilot making erratic, sudden decisions under pressure[cite: 31].

[cite_start]Every time a maneuver is triggered, the target selects a new heading and climb rate[cite: 32]. [cite_start]The simulation then updates the positions of both the target and the missile 20 times per second (dt = 0.05s) using Euler integration, calculating the engagement geometry step-by-step until intercept or miss[cite: 33].

---

## [cite_start]5.0 Workflow and AI Assistance [cite: 34]
[cite_start]The core flight mechanics, the TPN vector mathematics, and the integration logic were developed and programmed manually[cite: 35]. [cite_start]However, to speed up the workflow, Google Gemini was used within VS Code strictly as a programming assistant[cite: 36]. [cite_start]Specifically, the AI was used to write the boilerplate code for the visual outputs[cite: 37]. [cite_start]It helped structure the complex Matplotlib 3D projections and align the time-series data for the diagnostic dashboard[cite: 38]. [cite_start]Using AI to handle the UI and plotting syntax allowed the primary engineering focus to remain on getting the physics and the math right[cite: 39].

---

## [cite_start]6.0 Current Limitations and Next Steps [cite: 40]
[cite_start]This simulator provides a solid baseline, but there are clear steps for future improvement[cite: 41]:

* [cite_start]**Aerodynamic Drag:** The current coast phase assumes constant velocity[cite: 42]. [cite_start]Adding Mach-dependent drag coefficients (Cd) will allow the simulator to calculate the missile's actual kinematic range[cite: 43].
* [cite_start]**6-DOF Rigid Body:** Upgrading from a 3-DOF point-mass model to 6 Degrees of Freedom[cite: 44]. [cite_start]This will allow for the simulation of roll rates, angle of attack, and actuator delay[cite: 45].
* [cite_start]**Advanced Guidance (APN):** Implementing Augmented Proportional Navigation (APN)[cite: 46]. [cite_start]APN estimates the target's acceleration (often using a Kalman filter) to intercept highly maneuverable threats much more efficiently than standard TPN[cite: 46].

---

## Visuals

![Visual Output](Visual_output.png)

![Terminal Output](Terminal_output.png)