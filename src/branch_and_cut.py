""" A Bender decomposition based Branch-and-Cut algorithm """

from gurobipy import *


class BranchAndCut(object):
    def __init__(self, param1, param2):
        self.param1 = param1
        self.param2 = param2

    def branch_and_cut(self):
        pass

    def solveMasterProblem(self, relaxed=True):
        """ X_s is the binary decision variable to open facility or not """
        model = Model('Bender Decomposition MP')
        model.Params.OutputFlag = 0


    def solveSubploblem(self, duals):
        model = Model('Bender Decomposition Sub-problem')
        model.Params.OutputFlag = 0
        var_dict = {}

if __name__ == '__main__':
    pass