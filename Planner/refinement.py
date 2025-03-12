import torch
import theseus as th
import matplotlib.pyplot as plt
from common_utils import *
# NOTE 轨迹微调模块没有参与训练！！！只是在推理的时候用

def speed_constraint(optim_vars, aux_vars):
    ds = optim_vars[0].tensor
    speed_error = ds * (ds < 0)

    return speed_error

def acceleration(optim_vars, aux_vars):
    ds = optim_vars[0].tensor
    current_speed = aux_vars[0].tensor[:, 3]
    speed = torch.cat([current_speed[:, None], ds], dim=-1)
    acc = torch.diff(speed) / DT
    
    return acc

def speed_target(optim_vars, aux_vars):
    ds = optim_vars[0].tensor
    speed_limit = aux_vars[0].tensor
    s = aux_vars[1].tensor
    s = (s * 10).long().clip(0, speed_limit.shape[1]-1)
    speed_limit = speed_limit[torch.arange(s.shape[0])[:, None], s]
    speed_error = ds - speed_limit

    return speed_error

def acceleration_constraint(optim_vars, aux_vars):
    ds = optim_vars[0].tensor
    current_speed = aux_vars[0].tensor[:, 3]
    speed = torch.cat([current_speed[:, None], ds], dim=-1)
    acc = torch.diff(speed) / DT
    acc = acc * torch.logical_or(acc < -4.0, acc > 2.4)
    
    return acc

def jerk(optim_vars, aux_vars):
    ds = optim_vars[0].tensor
    current_speed = aux_vars[0].tensor[:, 3]
    current_acc = aux_vars[0].tensor[:, 5]
    speed = torch.cat([current_speed[:, None], ds], dim=-1)
    acc = torch.diff(speed) / DT
    acc = torch.cat([current_acc[:, None], acc], dim=-1)
    jerk = torch.diff(acc) / DT

    return jerk

def end_condition(optim_vars, aux_vars):
    ds = optim_vars[0].tensor
    s = torch.cumsum(ds * DT, dim=-1)
    end = aux_vars[0].tensor[:, -1:]
    end_condition = (s[:, -1:] - end)
    end_condition = end_condition * (end_condition.abs() > 3.0)

    return end_condition

def safety(optim_vars, aux_vars):
    ds = optim_vars[0].tensor
    s = torch.cumsum(ds * DT, dim=-1)
    occupancy = aux_vars[1].tensor
    safety_cost = []
    grid = torch.arange(0, MAX_LEN).to(occupancy.device)

    for t in [1, 3, 6, 9, 14, 19, 24, 29]:
        o = occupancy[:, t]
        error = ((s[:, t, None] + LENGTH) - grid) * o 
        error = error * ((s[:, t, None] + LENGTH) > grid)
        safety_cost.append(torch.sum(error, dim=-1))

    safety = torch.stack(safety_cost, dim=1)

    return safety


class RefinementPlanner:
    def __init__(self, device):
        self._device = device
        self.N = int(T/DT) # trajectory points (ds/dt)
        self.gains = {
            "speed": 0.3,
            "accel": 0.1,
            "jerk": 1.0,
            "end": 3.0,
            "soft_constraint": 10.0,
            "hard_constraint": 100.0
        }
        self.build_optimizer()

    def build_optimizer(self):
        # 这里的变量是需要优化求解的变量，可以当做是待优化的参数
        control_variables = th.Vector(dof=self.N, name="ds")
        # 定义变量，这里的变量是已知的变量，代入后续求解
        weights = {k: th.ScaleCostWeight(th.Variable(torch.tensor(v), name=f'gain_{k}')) for k, v in self.gains.items()}
        ego_state = th.Variable(torch.empty(1, 7), name="ego_state")
        occupancy = th.Variable(torch.empty(1, 30, MAX_LEN), name="occupancy")
        speed_limit = th.Variable(torch.empty(1, MAX_LEN*10), name="speed_limit")
        ego_pred_plan = th.Variable(torch.empty(1, int(T/DT)), name="s")

        objective = th.Objective()
        self.objective = self.build_cost_function(objective, control_variables, ego_state, ego_pred_plan, 
                                                  speed_limit, occupancy, weights)
        # 定义优化方法
        self.optimizer = th.GaussNewton(objective, th.CholeskyDenseSolver, 
                                        max_iterations=20, 
                                        step_size=0.3,
                                        rel_err_tolerance=1e-3)
        # 构造优化问题
        self.layer = th.TheseusLayer(self.optimizer, vectorize=False)
        self.layer.to(device=self._device)
    # 构造目标函数
    def build_cost_function(self, objective, control_variables, ego_state, ego_pred_plan, speed_limit, occupancy, weights, vectorize=True):
        # speed must be positive
        # 传入的参数分别是优化变量、子目标函数的计算方程、维度、子目标函数的权重、是否使用自动求导、子目标函数的名称
        speed_hard = th.AutoDiffCostFunction([control_variables], speed_constraint, self.N, weights['hard_constraint'], 
                                             autograd_vectorize=vectorize, name="speed_constraint")
        objective.add(speed_hard)

        # minimize acceleration
        acc_cost = th.AutoDiffCostFunction([control_variables], acceleration, self.N, weights['accel'],
                                            aux_vars=[ego_state], autograd_vectorize=vectorize, name="acceleration")
        objective.add(acc_cost)
        
        # close to target speed
        speed_cost = th.AutoDiffCostFunction([control_variables], speed_target, self.N, weights['speed'], 
                                             aux_vars=[speed_limit, ego_pred_plan], autograd_vectorize=vectorize, name="speed")
        objective.add(speed_cost)

        # regularize acceleration within the limits
        acc_hard = th.AutoDiffCostFunction([control_variables], acceleration_constraint, self.N, weights['soft_constraint'],
                                            aux_vars=[ego_state], autograd_vectorize=vectorize, name="acceleration_constraint")
        objective.add(acc_hard)

        # minimize jerk
        jerk_cost = th.AutoDiffCostFunction([control_variables], jerk, self.N, weights['jerk'],
                                            aux_vars=[ego_state], autograd_vectorize=vectorize, name="jerk")
        objective.add(jerk_cost)
        
        # collision avoidance
        safety_cost = th.AutoDiffCostFunction([control_variables], safety, 8, weights['hard_constraint'], 
                                               aux_vars=[ego_state, occupancy], autograd_vectorize=vectorize, name="safety")
        objective.add(safety_cost)

        # minimize distance to the predicted end point
        end_cost = th.AutoDiffCostFunction([control_variables], end_condition, 1, weights['end'],
                                           aux_vars=[ego_pred_plan], autograd_vectorize=vectorize, name="end_condition")
        objective.add(end_cost)
        # 总的优化目标
        return objective

    def plan(self, ego_state, init_plan, pred_plan, occupancy, ref_path):
        # initial plan
        # .clamp(min=0) 是 PyTorch 中的一个方法，它会将 init_plan 中的所有元素限制在 min=0 以上。也就是说，如果 init_plan 中有任何元素小于 0，它们会被替换为 0；如果元素大于或等于 0，则保持不变。
        # 纵向速度 frenet坐标系下
        ds = init_plan.clamp(min=0)
        # ego预测出来的轨迹 在frenet坐标系下（reference path下）纵向轨迹。
        s = pred_plan[:, 1:].clamp(min=0)

        # occupancy grid
        grid = torch.arange(0, MAX_LEN).to(occupancy.device)
        mask = grid > s[:, :, None]
        # 80 120 1
        occupancy = occupancy * mask
        # 只考虑前三十个点
        occupancy = occupancy[:, :30]

        # set speed limit
        speed_limit = ref_path[:, :, 4]

        # update planner inputs
        # NOTE 包含了待优化变量ds,同时也赋予了初始值
        planner_inputs = {
            "ds": ds,
            "s": s,
            "occupancy": occupancy,
            "ego_state": ego_state,
            "speed_limit": speed_limit
        }
        
        # plan
        _, info = self.layer.forward(planner_inputs, optimizer_kwargs={'track_best_solution': True})
        # _, info = self.layer.forward(planner_inputs, optimizer_kwargs={'track_best_solution': True, "verbose":True})

        ds = info.best_solution['ds'].clamp(min=0)
        # 累积行驶距离 也就是frenet坐标系下的纵向距离
        s = torch.cumsum(ds * DT, dim=-1).to(self._device)

        return s, ds
