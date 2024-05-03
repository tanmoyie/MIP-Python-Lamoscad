""" A Bender decomposition based Branch-and-Cut algorithm """

from gurobipy import *


class BranchAndCut(object):
    def __init__(self, roll_length, orders):
        self.roll_length = roll_length
        self.orders = orders

    def branch_and_cut(self):
        pass

    def solveMasterProblem(self, relaxed=True):
        """ X_s is the binary decision variable to open facility or not """
        model = Model('Column Generation MP')
        model.Params.OutputFlag = 0


    def solveSubploblem(self, duals):
        model = Model('Column Generation Sub-problem')
        model.Params.OutputFlag = 0
        var_dict = {}

if __name__ == '__main__':
    pass # ++