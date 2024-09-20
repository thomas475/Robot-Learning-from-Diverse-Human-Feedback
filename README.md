# Robot Learning from Diverse Human Feedback

This repository contains the code for my project **"Robot Learning from Diverse Human Feedback"**, developed as part of the **Interactive Learning Seminar and Practical Course** at the **Intuitive Robots Lab (IRL), Karlsruhe Institute of Technology (KIT)** during the summer semester of 2024.

I extended the Uni-RLHF system by implementing learning from **attribute, evaluative, and keypoint feedback**. Additionally, I integrated support for **learning from multiple feedback types simultaneously**. I also added a challenging robot task with the **Franka kitchen domain**, along with a **diffusion-based offline reinforcement learning algorithm** for benchmarking purposes.

## Installation

As the Uni-RLHF system is originally split into two repositories, we have also put our forks of the annotation platform and offline RL benchmarking framework in separate repositories.

To fully utilize this system, you need to install both repositories:

1. First, follow the installation instructions for the **Uni-RLHF Platform**.
2. Then, proceed with the **Clean-Offline-RLHF** installation.

## Slurm Scripts

Our experiments were conducted on the **BwUniCluster**, which uses the **Slurm Workload Manager** for job scheduling. Our Slurm scripts are provided in this repository. If you also use this cluster or another system that employs Slurm, these scripts can be quite helpful.