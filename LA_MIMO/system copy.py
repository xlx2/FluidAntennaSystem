import cvxpy as cp
import numpy as np


class MIMO_system:
    def __init__(self, **parameters):
        for key, value in parameters.items():
            setattr(self, key, value)

    def run(self, verbose=False):
        """
        Run the traditional MIMO beamforming optimization process.
        Parameters:
            verbose (bool): Whether to print the optimization process. Default is False.
        Returns:
            float: The CRB value.
        """
        lambda_old = np.zeros((self.N, 1))
        start_idx = (self.N - self.Nt) // 2
        end_idx = start_idx + self.Nt
        lambda_old[start_idx:end_idx] = 1

        # Define CVXPY Variables
        Rx = cp.Variable((self.N, self.N), hermitian=True)
        W = [cp.Variable((self.N, self.N), hermitian=True) for _ in range(self.K)]
        t = cp.Variable()

        # Constraints
        constraints = []

        # Objective Function Constraints for Sensing
        for m in range(self.M):
            constraints.append(cp.real(self.a[:, m:m+1].T.conj() @ np.diag(lambda_old[:, 0]) @ Rx @ np.diag(lambda_old[:, 0]) @ self.a[:, m:m+1]) >= t)

        # Power Constraint
        constraints.append(cp.real(cp.trace(np.diag(lambda_old[:, 0]) @ Rx @ np.diag(lambda_old[:, 0]))) <= self.P)

        # Qos Constraints
        for k in range(self.K):
            lhs = (1 + 1 / self.Gamma) * cp.real(self.H[:, k:k+1].T.conj() @ np.diag(lambda_old[:, 0]) @ W[k] @ np.diag(lambda_old[:, 0]) @ self.H[:, k:k+1])
            rhs = cp.real(self.H[:, k:k+1].T.conj() @ np.diag(lambda_old[:, 0]) @ Rx @ np.diag(lambda_old[:, 0]) @ self.H[:, k:k+1]) + self.sigmaC2
            constraints.append(rhs - lhs <= 0)

        # SDR Constraints
        constraints.append((Rx - sum(W)) >> 0)
        constraints.append(Rx >> 0)
        for k in range(self.K):
            constraints.append(W[k] >> 0)

        # Define the objective function
        objective = cp.Maximize(t)

        # Solve the problem
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.MOSEK, verbose=verbose)
        assert problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE], "Optimization failed.\n"

