import torch
from camera import Lie
lie = Lie()

while True:
    SE3 = lie.se3_to_SE3(torch.rand(6))
    if (SE3.T @ SE3).det() == 0: continue
    U = (SE3.T @ SE3).inverse() @ SE3.T
    rank = (torch.linalg.matrix_rank(U)).item()
    if rank != 3: print(U, rank)