# Classical and Modern Approaches for Solving the Steiner Tree Problem (IDP)

Implemented:

1. Repetitive shortest path heuristic: https://core.ac.uk/download/pdf/82609861.pdf
2. Primal-dual algorithm: https://faculty.cc.gatech.edu/~vigoda/6550/Notes/Lec14.pdf + early stopping mechanism
3. Mehlhorn algorithm: https://people.mpi-inf.mpg.de/~mehlhorn/ftp/SteinerTrees.pdf
4. Reinforcement learning algorithm based on the paper https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9516291

Evaluation:
1. Run all algorithms on PACE instances
2. Compute running time for the first 3 algorithms on PACE instances
3. Early stopping mechanism for the primal-dual algorithm - compare results with the results computed by the primal-dual algorithm without early stopping
4. Cherrypick: 

    1. Run the algorithm for the same instance multiple times and pick the best solution
    2. Copy weights: run Cherrypick on PACE instances with random initial weights, get the final Q values and set them as the Q values in the next algorithm run (no training in the latter run, compute the final solution using the trained weights from the former run)
    3. Sequential runs: run Cherrypick on PACE instances with random initial weights, get the final Q values and set them as the initial weights for the subsequent algorithm run - repeat for a predefined number of steps
    4. Transfer learning - subgraph: use the trained weights of a larger graph on a subgraph of it 

Evaluation results: https://docs.google.com/spreadsheets/d/14AepsbIOwTj-dg9KPeTvKIgYVx1qr4jRyKGqT_nWQF4/edit?usp=sharing
