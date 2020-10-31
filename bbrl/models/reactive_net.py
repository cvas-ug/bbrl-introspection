import torch
import torch.nn as nn

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        torch.nn.init.xavier_normal_(m.weight)

class ReactiveNet(nn.Module):

    LOG_SIG_MAX = 20
    LOG_SIG_MIN = -20
    """
        Implements the reactive network consisting of predefined behaviours.
        The behaviours here are:
            - approach
            - grasp
            - retract
    """
    def __init__(self):

        super(ReactiveNet, self).__init__()
        # approach
        self.approach_x_mean = nn.Linear(128, 1)
        self.approach_x_log_std = nn.Linear(128, 1)
        self.approach_y_mean = nn.Linear(128, 1)
        self.approach_y_log_std = nn.Linear(128, 1)
        self.approach_z_mean = nn.Linear(128, 1)
        self.approach_z_log_std = nn.Linear(128, 1)

        # grasp
        self.grasp_x_mean = nn.Linear(128, 1)
        self.grasp_x_log_std = nn.Linear(128, 1)
        self.grasp_y_mean = nn.Linear(128, 1)
        self.grasp_y_log_std = nn.Linear(128, 1)
        self.grasp_z_mean = nn.Linear(128, 1)
        self.grasp_z_log_std = nn.Linear(128, 1)
        # self.grasp_close_mean = nn.Linear(128, 1)
        # self.grasp_close_log_std = nn.Linear(128, 1)
        self.grasp_rotate_mean = nn.Linear(128, 1)
        self.grasp_rotate_log_std = nn.Linear(128, 1)

        # retract
        self.retract_x_mean = nn.Linear(128, 1)
        self.retract_x_log_std = nn.Linear(128, 1)
        self.retract_y_mean = nn.Linear(128, 1)
        self.retract_y_log_std = nn.Linear(128, 1)
        self.retract_z_mean = nn.Linear(128, 1)
        self.retract_z_log_std = nn.Linear(128, 1)

    def forward(self, x):
        # Feed forward features from feature extraction module
        az = self.approach_z_mean(x)
        log_std_az = self.approach_z_log_std(x)
        log_std_az = torch.clamp(log_std_az, min=self.LOG_SIG_MIN, max=self.LOG_SIG_MAX)
        ay = self.approach_y_mean(x)
        log_std_ay = self.approach_y_log_std(x)
        log_std_ay = torch.clamp(log_std_ay, min=self.LOG_SIG_MIN, max=self.LOG_SIG_MAX)
        ax = self.approach_x_mean(x)
        log_std_ax = self.approach_x_log_std(x)
        log_std_ax = torch.clamp(log_std_ax, min=self.LOG_SIG_MIN, max=self.LOG_SIG_MAX)
        approach_means = torch.cat([az, ay, ax]).type(torch.cuda.FloatTensor)
        approach_log_stds = torch.cat([log_std_az, log_std_ay, log_std_ax]).type(torch.cuda.FloatTensor)
        
        # m0 -> x, m1 -> y, m2 -> z, m3 -> close, m4 -> rotate
        gx = self.grasp_x_mean(x)
        log_std_gx = self.grasp_x_log_std(x)
        log_std_gx = torch.clamp(log_std_gx, min=self.LOG_SIG_MIN, max=self.LOG_SIG_MAX)
        gy = self.grasp_y_mean(x)
        log_std_gy = self.grasp_y_log_std(x)
        log_std_gy = torch.clamp(log_std_gy, min=self.LOG_SIG_MIN, max=self.LOG_SIG_MAX)
        gz = self.grasp_z_mean(x)
        log_std_gz = self.grasp_z_log_std(x)
        log_std_gz = torch.clamp(log_std_gz, min=self.LOG_SIG_MIN, max=self.LOG_SIG_MAX)
        # gc = self.grasp_close_mean(x)
        # log_std_gc = self.grasp_close_log_std(x)
        # log_std_gc = torch.clamp(log_std_gc, min=self.LOG_SIG_MIN, max=self.LOG_SIG_MAX)
        gr = self.grasp_rotate_mean(x)
        log_std_gr = self.grasp_rotate_log_std(x)
        log_std_gr = torch.clamp(log_std_gr, min=self.LOG_SIG_MIN, max=self.LOG_SIG_MAX)
        # grasp_means = torch.cat([gx, gy, gz, gc, gr]).type(torch.cuda.FloatTensor)
        grasp_means = torch.cat([gx, gy, gz, gr]).type(torch.cuda.FloatTensor)
        # grasp_log_stds = torch.cat([log_std_gx, log_std_gy, log_std_gz, log_std_gc, log_std_gr]).type(torch.cuda.FloatTensor)
        grasp_log_stds = torch.cat([log_std_gx, log_std_gy, log_std_gz, log_std_gr]).type(torch.cuda.FloatTensor)
        
        rz = self.retract_z_mean(x)
        log_std_rz = self.retract_z_log_std(x)
        log_std_rz = torch.clamp(log_std_rz, min=self.LOG_SIG_MIN, max=self.LOG_SIG_MAX)
        ry = self.retract_y_mean(x)
        log_std_ry = self.retract_y_log_std(x)
        log_std_ry = torch.clamp(log_std_ry, min=self.LOG_SIG_MIN, max=self.LOG_SIG_MAX)
        rx = self.retract_x_mean(x)
        log_std_rx = self.retract_x_log_std(x)
        log_std_rx = torch.clamp(log_std_rx, min=self.LOG_SIG_MIN, max=self.LOG_SIG_MAX)
        retract_means = torch.cat([rz, ry, rx]).type(torch.cuda.FloatTensor)
        retract_log_stds = torch.cat([log_std_rz, log_std_ry, log_std_rx]).type(torch.cuda.FloatTensor)

        output = {}
        output["approach_means"] = approach_means
        output["approach_log_stds"] = approach_log_stds
        output["retract_means"] = retract_means
        output["retract_log_stds"] = retract_log_stds
        output["grasp_means"] = grasp_means
        output["grasp_log_stds"] = grasp_log_stds
        
        return output

    def get_trainable_layers(self, behaviour):
        # Returns list of layers to not freeze
        if behaviour == "approach":
            return ["approach_x_mean", "approach_x_log_std", "approach_y_mean", "approach_y_log_std",
                    "approach_z_mean", "approach_z_log_std"]
        elif behaviour == "grasp":
            # return ["grasp_x_mean", "grasp_x_log_std", "grasp_y_mean", "grasp_y_log_std",
            #         "grasp_z_mean", "grasp_z_log_std", "grasp_close_mean", "grasp_close_log_std",
            #         "grasp_rotate_mean", "grasp_rotate_log_std"]
            return ["grasp_x_mean", "grasp_x_log_std", "grasp_y_mean", "grasp_y_log_std",
                    "grasp_z_mean", "grasp_z_log_std", "grasp_rotate_mean", "grasp_rotate_log_std"]
        elif behaviour == "retract":
            return ["retract_x_mean", "retract_x_log_std", "retract_y_mean", "retract_y_log_std",
                    "retract_z_mean", "retract_z_log_std"]