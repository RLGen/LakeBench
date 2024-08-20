import math
import logging

min_read_cost = 1000000.0
read_set_cost_slope = 1253.19054300781
read_set_cost_intercept = -9423326.99507381
read_list_cost_slope = 1661.93366983753
read_list_cost_intercept = 1007857.48225696


def readListCost(length):
    f = read_list_cost_slope * length + read_list_cost_intercept
    if f < min_read_cost:
        f = min_read_cost
    return f / 1000000.0


def readSetCost(size):
    f = read_set_cost_slope * size + read_set_cost_intercept
    if f < min_read_cost:
        f = min_read_cost
    return f / 1000000.0


def readSetSostSeduction(size, truncation):
    return readSetCost(size) - readSetCost(size - truncation)

