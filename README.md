# RayTransferMatrix
This repo is a simple implementation of the Ray Transfer Matrix (RTM) theory for the case of a Compound Refractive Lens (CRL). This repo represents a collection of script that is aimed at understanding the theory behind CRL focusing lengths, back-focal plane transformations, as well as the downstream imaging plane. This code represents a set of casual scripts used for personal learing in the field of DFXM by Axel Henningsson, as is self evident - this is not a general library for X-ray otics.

# Simulation
To run a simulation the script crl_simulation.py is useful :=)

![image](https://github.com/user-attachments/assets/bc515268-68bf-47e2-a6e2-bdcb9dcce676)

# Theory Outline
This repo relates to the following two iUCR publcations:

[https://doi.org/10.1107/S1600576717011037](https://doi.org/10.1107/S1600576717011037)

[https://doi.org/10.1107/S1600576717011037](https://doi.org/10.1107/S160057751602049X)

the main idea is that a single thin lens in the CRL can be described as the matrix M,

![image](https://github.com/user-attachments/assets/cc01498a-b60b-4c8f-a418-3b904be49e0b)

acting on a ray [y, alpha] where y is the ofset from the optical axis and alpha is the angle.

