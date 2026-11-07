"""
Two-agent active debris removal environment (PettingZoo ParallelEnv) and an
IPPO training loop on Stable-Baselines3.

The "pilot" fires 3 thrusters and 3 reaction wheels; the "arm_operator" drives
the 9 joints of three 3-DOF arms, which only move once hovering is confirmed.
Phases: 1.0 approach to 2.1 m in front of the debris, 1.5 hovering/capture,
2.0 return to the -8 m base with the debris attached, 3.0 done.

Training alternates the two agents each round; the one not being trained acts
with its latest saved policy (random on the first round).

Run: python adr_marl_env.py
"""

import os
import glob
import random
import warnings
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from pettingzoo import ParallelEnv
from pettingzoo.utils import wrappers as pz_wrappers

import supersuit as ss

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from skyfield.api import load
from skyfield.data import hipparcos

warnings.filterwarnings("ignore", category=UserWarning)

# Quaternions are scalar-first [qw, qx, qy, qz] and map body -> ECI.

def q_to_rot_mat(q):
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2*qy**2 - 2*qz**2,  2*qx*qy - 2*qw*qz,  2*qx*qz + 2*qw*qy],
        [2*qx*qy + 2*qw*qz,  1 - 2*qx**2 - 2*qz**2,  2*qy*qz - 2*qw*qx],
        [2*qx*qz - 2*qw*qy,  2*qy*qz + 2*qw*qx,  1 - 2*qx**2 - 2*qy**2]
    ], dtype=np.float64)

def rot_mat_to_q(R):
    # Shepperd's method: branch on the largest diagonal term to avoid dividing by ~0
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2.0
        qw, qx = 0.25*S, (R[2,1]-R[1,2])/S
        qy, qz = (R[0,2]-R[2,0])/S, (R[1,0]-R[0,1])/S
    elif (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
        S = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2.0
        qw, qx = (R[2,1]-R[1,2])/S, 0.25*S
        qy, qz = (R[0,1]+R[1,0])/S, (R[0,2]+R[2,0])/S
    elif R[1,1] > R[2,2]:
        S = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2.0
        qw, qx = (R[0,2]-R[2,0])/S, (R[0,1]+R[1,0])/S
        qy, qz = 0.25*S, (R[1,2]+R[2,1])/S
    else:
        S = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2.0
        qw, qx = (R[1,0]-R[0,1])/S, (R[0,2]+R[2,0])/S
        qy, qz = (R[1,2]+R[2,1])/S, 0.25*S
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    return q / np.linalg.norm(q)

def q_dot(q, omega):
    # Quaternion kinematics: q_dot = 0.5 * Omega(omega) q, omega in body frame
    qw, qx, qy, qz = q
    wx, wy, wz = omega
    w_mat = np.array([
        [  0, -wx, -wy, -wz],
        [ wx,   0,  wz, -wy],
        [ wy, -wz,   0,  wx],
        [ wz,  wy, -wx,   0]
    ], dtype=np.float64)
    return 0.5 * (w_mat @ q)

def rk4_step(mu, r, v, dt):
    # Two-body (Keplerian) dynamics
    def deriv(r_, v_): return v_, -mu / np.linalg.norm(r_)**3 * r_
    k1r, k1v = deriv(r, v)
    k2r, k2v = deriv(r + 0.5*dt*k1r, v + 0.5*dt*k1v)
    k3r, k3v = deriv(r + 0.5*dt*k2r, v + 0.5*dt*k2v)
    k4r, k4v = deriv(r +     dt*k3r, v +     dt*k3v)
    return (r + (dt/6.)*(k1r+2*k2r+2*k3r+k4r),
            v + (dt/6.)*(k1v+2*k2v+2*k3v+k4v))

def circular_state(r_orbit, angle_rad, mu):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    vc = np.sqrt(mu / r_orbit)
    return (r_orbit * np.array([c, s, 0.], dtype=np.float64),
            vc     * np.array([-s, c, 0.], dtype=np.float64))

class StarTrackerModel:
    """Hipparcos bright-star catalog plus Sun direction from DE421, loaded once and shared."""
    def __init__(self):
        print("[StarTracker] Loading JPL DE421 ephemeris + Hipparcos catalog...")
        # Sun direction is frozen at a single epoch; fine for episodes of ~2000 s
        ts = load.timescale()
        t  = ts.utc(2024, 5, 15, 12, 0, 0)
        eph = load('de421.bsp')
        sun_pos = eph['earth'].at(t).observe(eph['sun']).position.km
        self.sun_dir_eci = sun_pos / np.linalg.norm(sun_pos)

        with load.open(hipparcos.URL) as f:
            df = hipparcos.load_dataframe(f)
        bright = df[df['magnitude'] <= 4.5]
        ra  = np.radians(bright['ra_degrees'].values)
        dec = np.radians(bright['dec_degrees'].values)
        self.star_catalog_eci = np.column_stack((
            np.cos(dec)*np.cos(ra),
            np.cos(dec)*np.sin(ra),
            np.sin(dec)
        ))
        self.fov_cos_limit     = np.cos(np.radians(15.0))   # 15 deg half-angle field of view
        self.sun_exclusion_cos = np.cos(np.radians(35.0))   # Sun exclusion half-angle
        print(f"[StarTracker] Ready. {len(self.star_catalog_eci)} bright stars (mag<=4.5).")

    def measure_attitude(self, q_true):
        """
        Noisy TRIAD attitude fix from the two first stars in view. Falls back to
        a much noisier IMU-like estimate when the Sun is in the exclusion cone
        or fewer than two stars are visible. Returns (q_meas, R_eci_to_body).
        """
        R_b2e = q_to_rot_mat(q_true)
        R_e2b = R_b2e.T
        boresight_eci = R_b2e @ np.array([0., 1., 0.])  # body +Y axis = nose / boresight

        sun_blinded = np.dot(self.sun_dir_eci, boresight_eci) > self.sun_exclusion_cos
        dots = np.dot(self.star_catalog_eci, boresight_eci)
        stars_in_fov = self.star_catalog_eci[dots > self.fov_cos_limit]

        if len(stars_in_fov) >= 2 and not sun_blinded:
            # TRIAD: same orthonormal triad built in ECI and in the noisy body
            # frame, then R_e2b = M_body @ M_ref^T
            s1_eci, s2_eci = stars_in_fov[0], stars_in_fov[1]
            noise = 0.002
            s1 = R_e2b @ s1_eci + np.random.normal(0, noise, 3)
            s2 = R_e2b @ s2_eci + np.random.normal(0, noise, 3)
            s1 /= np.linalg.norm(s1); s2 /= np.linalg.norm(s2)

            t1 = s1_eci
            t2 = np.cross(s1_eci, s2_eci); t2 /= np.linalg.norm(t2)
            t3 = np.cross(t1, t2)
            M_ref = np.column_stack((t1, t2, t3))

            s2_o = np.cross(s1, s2)
            n = np.linalg.norm(s2_o)
            s2_o = s2_o / n if n > 1e-6 else np.array([0.,1.,0.])
            s3 = np.cross(s1, s2_o)
            M_body = np.column_stack((s1, s2_o, s3))

            R_e2b_meas = M_body @ M_ref.T
            q_meas = rot_mat_to_q(R_e2b_meas.T)
        else:
            # Stand-in for propagated IMU drift
            q_meas = q_true + np.random.normal(0, 0.05, 4)
            q_meas /= np.linalg.norm(q_meas)
            R_e2b_meas = q_to_rot_mat(q_meas).T

        return q_meas, R_e2b_meas


class ADRParallelEnv(ParallelEnv):
    """
    Two-agent ADR mission. pilot: 6 ternary actions, 18-dim obs.
    arm_operator: 9 ternary actions, 27-dim obs (pilot obs + joint angles).

    Joint commands are ignored outside phase 1.5, but the operator still gets
    observations and rewards every step so it learns when acting is pointless.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "name": "adr_marl_v0"}

    # SI units throughout
    MU          = 3.986004418e14
    RE          = 6_371e3
    DT          = 1.0
    MAX_THRUST  = 50.0
    MAX_TORQUE  = 5.0
    INIT_MASS   = 600.0
    DRY_MASS    = 450.0
    DEBRIS_MASS = 100.0
    ISP         = 220.0
    G0          = 9.81
    RHO         = 1e-13         # rough density at ~700 km [kg/m^3]
    CD          = 2.2           # drag coefficient
    DT_RK4      = 5.0           # step for the scripted Hohmann transfer [s]
    R_PARK      = RE + 600e3    # parking orbit radius
    R_TARGET    = RE + 700e3    # target (debris) orbit radius
    OST_R       = 2.0           # obstacle radius; the obstacle is the LVLH origin
    GOAL_RADIUS = 0.5
    CAP_DIST    = 2.1           # hover distance in front of the debris [m]
    MAX_STEPS   = 2000
    MAX_JOINT_VEL = np.radians(10.0)

    def __init__(self, star_tracker: StarTrackerModel = None, render_mode=None):
        super().__init__()

        self.render_mode = render_mode
        self.star_tracker = star_tracker  # None -> built on first reset

        # Hohmann parking -> target orbit; only the fuel cost is used, in reset
        a_t = 0.5 * (self.R_PARK + self.R_TARGET)
        v1  = np.sqrt(self.MU / self.R_PARK)
        v2  = np.sqrt(self.MU / self.R_TARGET)
        self.dv1 = np.sqrt(self.MU*(2/self.R_PARK - 1/a_t)) - v1
        self.dv2 = v2 - np.sqrt(self.MU*(2/self.R_TARGET - 1/a_t))
        self.T_transfer = np.pi * np.sqrt(a_t**3 / self.MU)

        # Box body, arms on the +Y (nose) face
        self.dims     = np.array([0.8, 1.5, 0.8])
        self.arm_lengths = [0.8, 0.8, 0.4]
        self.arm_bases   = [
            np.array([ 0.0,  self.dims[1]/2,  0.4]),
            np.array([-0.4,  self.dims[1]/2,  0.0]),
            np.array([ 0.4,  self.dims[1]/2,  0.0]),
        ]
        lim = np.pi / 2.0
        self.joint_limits = [(-lim, lim)] * 9
        self.joint_limits[1] = (0.0,  lim)   # top arm shoulder pitch: up only
        self.joint_limits[4] = (-lim, 0.0)   # left arm shoulder pitch: down only
        self.joint_limits[7] = (-lim, 0.0)   # right arm shoulder pitch: down only

        self.cp_offset = np.array([0.0, 0.2, 0.0])  # CoP offset from CoM, lever arm for drag torque

        # Homogeneous box; arm mass is ignored
        Ix = (1/12)*self.INIT_MASS*(self.dims[1]**2 + self.dims[2]**2)
        Iy = (1/12)*self.INIT_MASS*(self.dims[0]**2 + self.dims[2]**2)
        Iz = (1/12)*self.INIT_MASS*(self.dims[0]**2 + self.dims[1]**2)
        self.base_I = np.array([Ix, Iy, Iz])

        # Arms start spread open
        self.initial_joints = np.array([
             0.0,  0.5,  0.0,
             0.5, -0.5,  0.2,
            -0.5, -0.5,  0.2
        ], dtype=np.float64)

        self.possible_agents = ["pilot", "arm_operator"]
        self.agents = list(self.possible_agents)

        # Each channel: 0 = negative, 1 = off, 2 = positive
        self.action_spaces = {
            "pilot":        spaces.MultiDiscrete([3, 3, 3, 3, 3, 3]),
            "arm_operator": spaces.MultiDiscrete([3, 3, 3, 3, 3, 3, 3, 3, 3]),
        }

        pilot_obs_dim    = 18
        operator_obs_dim = 27

        pilot_high    = np.array([1.0]*pilot_obs_dim,    dtype=np.float32)
        operator_high = np.array([1.0]*operator_obs_dim, dtype=np.float32)

        self.observation_spaces = {
            "pilot":        spaces.Box(-pilot_high,    pilot_high,    dtype=np.float32),
            "arm_operator": spaces.Box(-operator_high, operator_high, dtype=np.float32),
        }

        self.traj_out_x, self.traj_out_y, self.traj_out_z = [], [], []
        self.traj_back_x, self.traj_back_y, self.traj_back_z = [], [], []

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def _gravity(self, r):
        return -self.MU / np.linalg.norm(r)**3 * r

    def _lvlh(self, r, v):
        rh = r / np.linalg.norm(r)
        hv = np.cross(r, v); hh = hv / np.linalg.norm(hv)
        vh = np.cross(hh, rh)
        return np.column_stack((rh, vh, hh))   # columns: R-bar, V-bar, H-bar in ECI

    def _disturbances(self, r_eci, q, v_eci):
        """Environmental torques in body frame: gravity gradient + aerodynamic drag."""
        R_b2e = q_to_rot_mat(q)
        R_e2b = R_b2e.T
        r_b   = R_e2b @ r_eci
        I_mat = np.diag(self.I)
        # Gravity gradient: 3 mu / |r|^5 * (r x I r), r in body frame
        tau_gg = (3*self.MU / np.linalg.norm(r_b)**5) * np.cross(r_b, I_mat @ r_b)
        v_b    = R_e2b @ v_eci
        vm     = np.linalg.norm(v_b)
        vh_    = v_b / (vm + 1e-9)
        # Box area projected on the flow direction
        area   = (self.dims[1]*self.dims[2]*abs(vh_[0]) +
                  self.dims[0]*self.dims[2]*abs(vh_[1]) +
                  self.dims[0]*self.dims[1]*abs(vh_[2]))
        f_drag = -vh_ * 0.5*self.RHO*vm**2*self.CD*area
        tau_drag = np.cross(self.cp_offset, f_drag)
        return tau_gg + tau_drag

    def _end_effectors(self):
        """End-effector positions of the three arms, body frame."""
        hands = []
        for i in range(3):
            bp = self.arm_bases[i]
            q1, q2, q3 = self.joints[i*3:(i+1)*3]
            L = self.arm_lengths
            p1 = bp + L[0]*np.array([np.sin(q1)*np.cos(q2),
                                      np.cos(q1)*np.cos(q2),
                                      np.sin(q2)])
            p2 = p1 + L[1]*np.array([np.sin(q1)*np.cos(q2+q3),
                                      np.cos(q1)*np.cos(q2+q3),
                                      np.sin(q2+q3)])
            p3 = p2 + L[2]*np.array([np.sin(q1)*np.cos(q2+q3),
                                      np.cos(q1)*np.cos(q2+q3),
                                      np.sin(q2+q3)])
            hands.append(p3)
        return hands

    def _hohmann_forward(self):
        """Scripted Hohmann transfer. Only its fuel cost reaches the episode; the final state is discarded."""
        r, v = circular_state(self.R_PARK, 0.0, self.MU)
        v += abs(self.dv1) * (v / np.linalg.norm(v))
        for _ in range(int(self.T_transfer / self.DT_RK4) + 1):
            r, v = rk4_step(self.MU, r, v, self.DT_RK4)
        fuel_used = self.mass * (1.0 - np.exp(-(abs(self.dv1)+abs(self.dv2))/(self.ISP*self.G0)))
        self.mass -= fuel_used

    def _sensor_obs(self):
        """Noisy relative-nav, gyro and star-tracker measurements shared by both agents."""
        rel_pos  = self.r_goal - self.r_chaser
        rel_vel  = self.v_goal - self.v_chaser
        rel_ost  = self.r_ost  - self.r_chaser
        dist_t   = np.linalg.norm(rel_pos)
        nf       = 1.0 + dist_t / 50.0   # LiDAR noise grows with range

        m_pos  = rel_pos + np.random.normal(0, 0.02*nf, 3)
        m_vel  = rel_vel + np.random.normal(0, 0.005,   3)
        m_ost  = rel_ost + np.random.normal(0, 0.05*nf, 3)
        m_omega= self.omega + np.random.normal(0, 0.002, 3)

        q_meas, R_e2b_meas = self.star_tracker.measure_attitude(self.q)

        tgt_body = R_e2b_meas @ m_pos
        vel_body = R_e2b_meas @ m_vel
        ost_body = R_e2b_meas @ m_ost
        mf       = self.mass / self.INIT_MASS

        return dict(
            tgt_body=tgt_body,
            vel_body=vel_body,
            ost_body=ost_body,
            m_omega=m_omega,
            q_meas=q_meas,
            mf=mf,
        )

    def _build_observations(self, sensor):
        """
        pilot (18): rel_pos/10, rel_vel/5, obstacle_pos/10, omega/0.5, q, mass
        fraction, phase/3. arm_operator (27): pilot obs + joints/pi. All in body
        frame and clipped to [-1, 1].
        """
        tb = sensor["tgt_body"]
        vb = sensor["vel_body"]
        ob = sensor["ost_body"]
        mo = sensor["m_omega"]
        q  = sensor["q_meas"]
        mf = sensor["mf"]

        pilot_obs = np.array([
            tb[0]/10, tb[1]/10, tb[2]/10,
            vb[0]/5,  vb[1]/5,  vb[2]/5,
            ob[0]/10, ob[1]/10, ob[2]/10,
            mo[0]/0.5, mo[1]/0.5, mo[2]/0.5,
            q[0], q[1], q[2], q[3],
            mf,
            self.phase / 3.0,
        ], dtype=np.float32)

        joint_obs = (self.joints / np.pi).astype(np.float32)
        arm_obs   = np.concatenate([pilot_obs, joint_obs])

        return {
            "pilot":        np.clip(pilot_obs, -1., 1.),
            "arm_operator": np.clip(arm_obs,   -1., 1.),
        }

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self.agents       = list(self.possible_agents)
        self.mass         = self.INIT_MASS
        self.I            = self.base_I.copy()
        self.joints       = self.initial_joints.copy()
        self.phase        = 1.0
        self.current_step = 0
        self.mission_done = False

        self.debris_frozen_local_pos = None
        self.ghost_debris_lvlh       = None
        self.hover_counter           = 0
        self.hover_bonus_given       = False

        self._hohmann_forward()

        r0 = self.R_TARGET
        v0 = np.sqrt(self.MU / r0)

        self.r_ost    = np.array([r0, 0., 0.], dtype=np.float64)
        self.v_ost    = np.array([0., v0, 0.], dtype=np.float64)

        Q = self._lvlh(self.r_ost, self.v_ost).T   # ECI -> LVLH
        L = Q.T                                      # LVLH -> ECI

        # Along V-bar: chaser at -8 m, obstacle at 0, debris at +8 m, so the obstacle is in the way
        self.r_chaser = self.r_ost + L @ np.array([0., -8., 0.])
        self.v_chaser = np.array([0., v0, 0.], dtype=np.float64)
        self.r_debris = self.r_ost + L @ np.array([0.,  8., 0.])
        self.v_debris = np.array([0., v0, 0.], dtype=np.float64)

        self.r_goal = self.r_debris + L @ np.array([0., -self.CAP_DIST, 0.])
        self.v_goal = self.v_debris.copy()

        self.q     = np.array([1., 0., 0., 0.], dtype=np.float64)
        self.omega = np.zeros(3, dtype=np.float64)

        if self.star_tracker is None:
            self.star_tracker = StarTrackerModel()

        # Baselines for the progress terms
        self.prev_dist_goal = np.linalg.norm(self.r_chaser - self.r_goal)
        self.prev_d1 = self.prev_d2 = self.prev_d3 = 10.0

        self.traj_out_x  = [0.]; self.traj_out_y  = [-8.]; self.traj_out_z  = [0.]
        self.traj_back_x = [];   self.traj_back_y  = [];    self.traj_back_z = []

        sensor = self._sensor_obs()
        obs    = self._build_observations(sensor)
        return obs, {a: {} for a in self.agents}

    def step(self, actions):
        self.current_step += 1

        # Bang-bang actuators
        thrust_map = {0: -self.MAX_THRUST, 1: 0., 2: self.MAX_THRUST}
        torque_map = {0: -self.MAX_TORQUE, 1: 0., 2: self.MAX_TORQUE}
        joint_map  = {0: -self.MAX_JOINT_VEL*self.DT,
                       1: 0.,
                       2:  self.MAX_JOINT_VEL*self.DT}

        pa = actions.get("pilot",        np.ones(6, dtype=int))
        aa = actions.get("arm_operator", np.ones(9, dtype=int))

        F_body = np.array([thrust_map[pa[0]], thrust_map[pa[1]], thrust_map[pa[2]]])
        T_body = np.array([torque_map[pa[3]], torque_map[pa[4]], torque_map[pa[5]]])

        # Joints are locked outside phase 1.5
        if self.phase == 1.5:
            for i in range(9):
                self.joints[i] += joint_map[aa[i]]
                self.joints[i]  = np.clip(self.joints[i],
                                           self.joint_limits[i][0],
                                           self.joint_limits[i][1])

        # Rocket equation; reaction-wheel torque is charged as propellant too
        dm = (np.sum(np.abs(F_body)) + np.sum(np.abs(T_body))) / (self.ISP * self.G0) * self.DT
        self.mass -= dm
        out_of_fuel = self.mass <= self.DRY_MASS
        if out_of_fuel:
            self.mass = self.DRY_MASS

        # Euler's equations with principal inertia, explicit Euler step
        tau_dist  = self._disturbances(self.r_chaser, self.q, self.v_chaser)
        T_total   = T_body + tau_dist
        omega_dot = (T_total - np.cross(self.omega, self.I * self.omega)) / self.I
        self.omega = np.clip(self.omega + omega_dot * self.DT, -0.5, 0.5)
        self.q    += q_dot(self.q, self.omega) * self.DT
        self.q    /= np.linalg.norm(self.q)

        R_b2e = q_to_rot_mat(self.q)
        R_e2b = R_b2e.T
        F_eci = R_b2e @ F_body

        # Explicit Euler for both bodies; at dt = 1 s the error is mostly common-mode in LVLH
        self.v_ost    += self._gravity(self.r_ost)    * self.DT
        self.r_ost    += self.v_ost                   * self.DT
        self.v_chaser += (self._gravity(self.r_chaser) + F_eci/self.mass) * self.DT
        self.r_chaser += self.v_chaser                * self.DT

        Q_e2l = self._lvlh(self.r_ost, self.v_ost).T  # ECI -> LVLH
        L2E   = Q_e2l.T                                # LVLH -> ECI

        if self.phase < 2.0:
            self.r_debris = self.r_ost  + L2E @ np.array([0.,  8., 0.])
            self.v_debris = self.v_ost.copy()
            self.r_goal   = self.r_debris + L2E @ np.array([0., -self.CAP_DIST, 0.])
            self.v_goal   = self.v_debris.copy()
        else:
            # Return leg: goal is the base, the captured debris rides rigidly with the chaser
            self.r_goal = self.r_ost + L2E @ np.array([0., -8., 0.])
            self.v_goal = self.v_ost.copy()
            if self.debris_frozen_local_pos is not None:
                self.r_debris = self.r_chaser + R_b2e @ self.debris_frozen_local_pos

        local_chaser = Q_e2l @ (self.r_chaser - self.r_ost)

        if self.phase < 2.0:
            self.traj_out_x.append(local_chaser[0])
            self.traj_out_y.append(local_chaser[1])
            self.traj_out_z.append(local_chaser[2])
        else:
            self.traj_back_x.append(local_chaser[0])
            self.traj_back_y.append(local_chaser[1])
            self.traj_back_z.append(local_chaser[2])

        dist_goal  = np.linalg.norm(self.r_chaser - self.r_goal)
        dist_ost   = np.linalg.norm(self.r_chaser - self.r_ost)
        vel_mag    = np.linalg.norm(self.v_chaser  - self.v_goal)
        omega_mag  = np.linalg.norm(self.omega)

        vec_debris_body = R_e2b @ (self.r_debris - self.r_chaser)
        dist_debris     = max(np.linalg.norm(self.r_debris - self.r_chaser), 1e-6)
        pointing_acc    = (vec_debris_body / dist_debris)[1]  # cos of angle between nose (+Y) and debris

        out_of_bounds = (np.abs(local_chaser) > 20).any()
        collision     = dist_ost <= self.OST_R

        r_pilot = 0.0

        r_pilot -= 0.005 * dist_goal
        r_pilot += (self.prev_dist_goal - dist_goal) * 0.15
        self.prev_dist_goal = dist_goal

        # Point the nose at the debris on the way out, at the base on the way back
        if self.phase < 2.0:
            r_pilot += pointing_acc * 0.02
        else:
            vec_base_body = R_e2b @ (self.r_goal - self.r_chaser)
            r_pilot += (vec_base_body / max(dist_goal, 1e-6))[1] * 0.02

        if omega_mag > 0.1:
            r_pilot -= omega_mag * 0.05

        # Exponential repulsion inside 2 * OST_R
        if dist_ost < self.OST_R * 2.0:
            p = (self.OST_R*2.0 - dist_ost) / self.OST_R
            r_pilot -= 0.1 * (np.exp(3.0*p) - 1.0)

        if out_of_bounds or collision:
            r_pilot -= 50.0

        r_pilot -= dm * 0.5
        if out_of_fuel:
            r_pilot -= 10.0

        r_arm = 0.0

        if self.phase == 1.0:
            # Small cost while waiting for the pilot
            r_arm -= 0.01

        elif self.phase == 1.5:
            # The arms also pay when the pilot drifts off station
            if dist_goal > self.GOAL_RADIUS * 4.0 or pointing_acc < 0.5:
                r_arm -= 2.0
            else:
                hands  = self._end_effectors()
                deb_b  = R_e2b @ (self.r_debris - self.r_chaser)

                d1 = np.linalg.norm(hands[0] - deb_b)
                d2 = np.linalg.norm(hands[1] - deb_b)
                d3 = np.linalg.norm(hands[2] - deb_b)

                # Per-hand progress, so one arm stalling doesn't hide the others
                r_arm += (self.prev_d1 - d1) * 10.0
                r_arm += (self.prev_d2 - d2) * 10.0
                r_arm += (self.prev_d3 - d3) * 10.0
                self.prev_d1, self.prev_d2, self.prev_d3 = d1, d2, d3

                if d1 < 1.0 and d2 < 1.0 and d3 < 1.0:
                    r_arm += 0.2

                # Keep joints off their limits (same margin as workspace.py)
                margin = 0.20
                for idx in range(9):
                    lo, hi = self.joint_limits[idx]
                    j = self.joints[idx]
                    if j > hi - margin or j < lo + margin:
                        r_arm -= 0.05

                r_arm -= 0.05

        elif self.phase >= 2.0:
            # Nothing for the operator to do on the return leg
            r_arm += 0.0

        # Phase transitions
        terminated_global = False

        if self.phase == 1.0:
            # Enter 1.5 after >5 consecutive steps near the goal, slow, stable and
            # pointing at the debris, with the debris at 25-85 % of every arm's reach
            if dist_goal <= self.GOAL_RADIUS * 2.0:
                if vel_mag < 0.3 and omega_mag < 0.08 and pointing_acc > 0.90:
                    self.hover_counter += 1
                    if self.hover_counter > 5:
                        deb_b      = R_e2b @ (self.r_debris - self.r_chaser)
                        max_reach  = sum(self.arm_lengths)
                        ws_ok      = all(
                            0.25*max_reach < np.linalg.norm(deb_b - bp) < 0.85*max_reach
                            for bp in self.arm_bases
                        )
                        if ws_ok:
                            print(f"[{self.current_step}] phase 1.5: hovering, capture enabled")
                            self.phase = 1.5

                            # Paid once, so the pilot can't farm it by dropping in and out of hover
                            if not self.hover_bonus_given:
                                r_pilot += 30.0
                                self.hover_bonus_given = True
                        else:
                            self.hover_counter = 0
                else:
                    self.hover_counter = 0
            else:
                self.hover_counter = 0

        elif self.phase == 1.5:
            hands = self._end_effectors()
            deb_b = R_e2b @ (self.r_debris - self.r_chaser)
            d1    = np.linalg.norm(hands[0] - deb_b)
            d2    = np.linalg.norm(hands[1] - deb_b)
            d3    = np.linalg.norm(hands[2] - deb_b)

            if dist_goal > self.GOAL_RADIUS * 4.0 or pointing_acc < 0.5:
                print(f"[{self.current_step}] hover lost, back to phase 1.0")
                self.phase = 1.0
                self.hover_counter = 0
                self.prev_d1 = self.prev_d2 = self.prev_d3 = 10.0
                r_pilot -= 15.0
            elif d1 < 0.45 and d2 < 0.45 and d3 < 0.45:
                # Capture: all three hands within 0.45 m of the debris centre
                print(f"[{self.current_step}] captured, phase 2.0 (return)")
                self.phase = 2.0
                r_pilot += 50.0
                r_arm   += 200.0

                hand_center = np.mean(hands, axis=0)
                self.debris_frozen_local_pos = hand_center
                cap_eci = self.r_chaser + R_b2e @ hand_center
                self.ghost_debris_lvlh = Q_e2l @ (cap_eci - self.r_ost)

                # Debris as a point mass at the hand centroid (parallel-axis); CoM shift is ignored
                self.mass += self.DEBRIS_MASS
                hc = hand_center
                self.I[0] += self.DEBRIS_MASS * (hc[1]**2 + hc[2]**2)
                self.I[1] += self.DEBRIS_MASS * (hc[0]**2 + hc[2]**2)
                self.I[2] += self.DEBRIS_MASS * (hc[0]**2 + hc[1]**2)

                # Goal jumped to the base, so reset the progress baseline
                self.prev_dist_goal = np.linalg.norm(self.r_chaser - self.r_goal)

        elif self.phase == 2.0:
            vec_base = R_e2b @ (self.r_goal - self.r_chaser)
            pointing_base = (vec_base / max(dist_goal, 1e-6))[1]

            if (dist_goal < self.GOAL_RADIUS and
                vel_mag < 0.20 and
                pointing_base > 0.90 and
                omega_mag < 0.10):
                print(f"[{self.current_step}] base reached, mission complete")
                r_pilot += 500.0
                r_arm   += 50.0
                self.phase        = 3.0
                self.mission_done = True
                terminated_global = True

        if out_of_bounds or collision or out_of_fuel:
            terminated_global = True

        truncated_global = self.current_step >= self.MAX_STEPS

        if terminated_global or truncated_global:
            self.agents = []   # PettingZoo convention for episode end

        terminations = {a: terminated_global for a in self.possible_agents}
        truncations  = {a: truncated_global  for a in self.possible_agents}
        rewards      = {"pilot": float(r_pilot), "arm_operator": float(r_arm)}

        sensor = self._sensor_obs()
        obs    = self._build_observations(sensor)
        infos  = {a: {"phase": self.phase, "dm": dm} for a in self.possible_agents}

        return obs, rewards, terminations, truncations, infos

    def render(self):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        fig = plt.figure(figsize=(10, 8), facecolor='white')
        ax  = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('white')

        u, v = np.mgrid[0:2*np.pi:15j, 0:np.pi:10j]
        ax.plot_wireframe(self.OST_R*np.cos(u)*np.sin(v),
                          self.OST_R*np.sin(u)*np.sin(v),
                          self.OST_R*np.cos(v),
                          color="red", alpha=0.3, label='Obstacle')
        ax.scatter(0, -8., 0, color='green', s=150, marker='s', label='Base')

        if self.traj_out_x:
            ax.plot(self.traj_out_x, self.traj_out_y, self.traj_out_z,
                    color='#2da44e', lw=2, label='Phase 1: Approach')
        if self.traj_back_x:
            ax.plot(self.traj_back_x, self.traj_back_y, self.traj_back_z,
                    color='purple', lw=2, ls='--', label='Phase 2: Return')

        R_b2e = q_to_rot_mat(self.q)
        Q_e2l = self._lvlh(self.r_ost, self.v_ost).T

        def b2l(p):
            # Body-frame point to LVLH, relative to the obstacle
            return Q_e2l @ (self.r_chaser + R_b2e @ p - self.r_ost)

        d = self.dims / 2
        verts_body = np.array([
            [-d[0],-d[1],-d[2]], [ d[0],-d[1],-d[2]],
            [ d[0], d[1],-d[2]], [-d[0], d[1],-d[2]],
            [-d[0],-d[1], d[2]], [ d[0],-d[1], d[2]],
            [ d[0], d[1], d[2]], [-d[0], d[1], d[2]],
        ])
        vp = np.array([b2l(v_) for v_ in verts_body])
        faces = [
            [vp[0],vp[1],vp[2],vp[3]], [vp[4],vp[5],vp[6],vp[7]],
            [vp[0],vp[1],vp[5],vp[4]], [vp[2],vp[3],vp[7],vp[6]],
            [vp[0],vp[3],vp[7],vp[4]], [vp[1],vp[2],vp[6],vp[5]],
        ]

        ax.add_collection3d(Poly3DCollection(faces, facecolors='#f17e11',
                                              edgecolors='black', alpha=0.8))
        nose  = b2l(np.array([0., self.dims[1]/2+1., 0.]))
        nbase = b2l(np.array([0., self.dims[1]/2,    0.]))

        ax.plot([nbase[0],nose[0]], [nbase[1],nose[1]], [nbase[2],nose[2]],
                color='blue', lw=2)

        for i in range(3):
            bp   = self.arm_bases[i]
            q1,q2,q3 = self.joints[i*3:(i+1)*3]
            L    = self.arm_lengths
            p0   = bp
            p1   = p0 + L[0]*np.array([np.sin(q1)*np.cos(q2),np.cos(q1)*np.cos(q2),np.sin(q2)])
            p2   = p1 + L[1]*np.array([np.sin(q1)*np.cos(q2+q3),np.cos(q1)*np.cos(q2+q3),np.sin(q2+q3)])
            p3   = p2 + L[2]*np.array([np.sin(q1)*np.cos(q2+q3),np.cos(q1)*np.cos(q2+q3),np.sin(q2+q3)])
            pts  = np.array([b2l(p) for p in [p0,p1,p2,p3]])

            ax.plot(pts[:,0], pts[:,1], pts[:,2],
                    color='#d4b21f', lw=3, marker='o', ms=4,
                    label='Arms' if i == 0 else "")

        if self.phase < 2.0:
            ax.scatter(0, 8., 0, color='magenta', s=200, marker='*', label='Debris')
        else:
            if self.ghost_debris_lvlh is not None:
                ax.scatter(*self.ghost_debris_lvlh, facecolors='none',
                           edgecolors='magenta', s=200, marker='*', label='Capture Point')
            dp = Q_e2l @ (self.r_debris - self.r_ost)
            ax.scatter(dp[0], dp[1], dp[2], color='magenta', s=200, marker='*',
                       label='Debris (Captured)')

        ax.set_xlim([-10,10]); ax.set_ylim([-12,12]); ax.set_zlim([-10,10])

        ax.set_xlabel('X R-Bar [m]', color='black')
        ax.set_ylabel('Y V-Bar [m]', color='black')
        ax.set_zlabel('Z H-Bar [m]', color='black')
        ax.tick_params(colors='black')

        fuel = self.INIT_MASS - self.mass + (self.DEBRIS_MASS if self.phase >= 2.0 else 0)
        ax.set_title(f"MARL ADR | Phase:{self.phase} | Fuel:{fuel:.1f}kg | Step:{self.current_step}",
                     color='black', pad=20)

        ax.legend(facecolor='white', labelcolor='black', loc='upper right')

        if self.render_mode == "human":
            plt.show()
        elif self.render_mode == "rgb_array":
            fig.canvas.draw()
            img = np.asarray(fig.canvas.buffer_rgba())[:,:,:3]
            plt.close(fig)
            return img
        return None


class SingleAgentWrapper(gym.Env):
    """
    Single-agent view of ADRParallelEnv for SB3. The partner agent acts with a
    frozen policy loaded from partner_model_path, or randomly if it is None.

    Only the path crosses into SubprocVecEnv workers; the partner model, star
    tracker and env are built lazily inside each worker, so nothing heavy is
    pickled across the fork.
    """

    _OBS_DIM = {"pilot": 18, "arm_operator": 27}
    _ACT_DIM = {"pilot": [3,3,3,3,3,3], "arm_operator": [3]*9}

    def __init__(self, trainee: str, partner_model_path: str | None = None,
                 seed: int = 0):
        super().__init__()
        assert trainee in ["pilot", "arm_operator"]
        self.trainee             = trainee
        self.partner             = "arm_operator" if trainee == "pilot" else "pilot"
        self.partner_model_path  = partner_model_path
        self._seed               = seed

        self._partner_model      = None
        self._partner_loaded     = False

        self._star_tracker       = None
        self._env                = None

        # Spaces from static shapes so the parent never builds a real env
        n_obs = self._OBS_DIM[trainee]
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(n_obs,), dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete(self._ACT_DIM[trainee])

        self._last_obs_partner = None

    def _lazy_init(self):
        """Runs in the worker process."""
        if self._star_tracker is None:
            self._star_tracker = StarTrackerModel()
        if self._env is None:
            self._env = ADRParallelEnv(star_tracker=self._star_tracker)

    def _load_partner(self):
        """Load the partner once per worker. custom_objects pins the spaces so SB3's shape check passes."""
        if self._partner_loaded:
            return
        self._partner_loaded = True
        if self.partner_model_path is not None and os.path.exists(self.partner_model_path):
            n_obs_partner = self._OBS_DIM[self.partner]
            partner_obs_space = spaces.Box(
                low=-1.0, high=1.0, shape=(n_obs_partner,), dtype=np.float32
            )
            partner_act_space = spaces.MultiDiscrete(self._ACT_DIM[self.partner])
            self._partner_model = PPO.load(
                self.partner_model_path,
                device="cpu",
                custom_objects={
                    "observation_space": partner_obs_space,
                    "action_space":      partner_act_space,
                },
            )

    def _partner_action(self):
        self._load_partner()
        if self._partner_model is None:
            return spaces.MultiDiscrete(self._ACT_DIM[self.partner]).sample()
        act, _ = self._partner_model.predict(
            self._last_obs_partner, deterministic=False
        )
        return act

    def reset(self, seed=None, options=None):
        self._lazy_init()
        obs_dict, _ = self._env.reset(seed=seed if seed is not None else self._seed)
        self._last_obs_partner = obs_dict[self.partner]
        return obs_dict[self.trainee], {}

    def step(self, action):
        partner_act = self._partner_action()
        actions     = {self.trainee: action, self.partner: partner_act}
        obs, rews, terms, truncs, infos = self._env.step(actions)

        if self.partner in obs:
            self._last_obs_partner = obs[self.partner]

        obs_t   = obs.get(self.trainee,   np.zeros(self.observation_space.shape, dtype=np.float32))
        rew_t   = rews.get(self.trainee,  0.0)
        term_t  = terms.get(self.trainee, True)
        trunc_t = truncs.get(self.trainee, False)
        info_t  = infos.get(self.trainee,  {})

        return obs_t, rew_t, term_t, trunc_t, info_t

    def close(self):
        if self._env is not None:
            self._env.close()


def make_vec_env(trainee: str, partner_model_path: str | None,
                 num_envs: int = 8, seed: int = 42):
    """SubprocVecEnv of SingleAgentWrapper; the closures capture only picklable primitives."""
    _trainee = trainee
    _path    = partner_model_path
    _seed    = seed

    def make_env_fn(idx):
        def _init():
            env = SingleAgentWrapper(
                trainee=_trainee,
                partner_model_path=_path,
                seed=_seed + idx,
            )
            return env
        return _init

    vec = SubprocVecEnv([make_env_fn(i) for i in range(num_envs)])
    vec = VecMonitor(vec)
    return vec


def load_or_create(model_path: str, vec_env, seed: int, verbose: int = 1):
    """Resume from the newest checkpoint with this prefix, or start a fresh PPO model."""
    pattern = model_path.replace(".zip", "") + "*.zip"
    files   = glob.glob(pattern)
    if files:
        latest  = max(files, key=os.path.getmtime)
        print(f"  [LOAD] {latest}")
        return PPO.load(latest, env=vec_env, device="auto")
    else:
        print(f"  [NEW]  {model_path}")
        return PPO(
            "MlpPolicy", vec_env,
            learning_rate=3e-4, gamma=0.995,
            n_steps=8192, batch_size=1024,
            n_epochs=10, ent_coef=0.01,
            clip_range=0.15, gae_lambda=0.97,
            policy_kwargs=dict(net_arch=[256, 256]),
            verbose=verbose, seed=seed,
            device="cpu",
        )


if __name__ == "__main__":
    import gc
    import time
    import resource

    # Recreating SubprocVecEnvs every round leaks pipes/sockets faster than they
    # are reclaimed and hits the default soft fd limit (256 on macOS), so raise it.
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        new_soft = min(max(hard, 8192), hard) if hard > 0 else 8192
        resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
        print(f"[fd limit] {soft} -> {new_soft} (hard={hard})")
    except (ValueError, resource.error) as e:
        print(f"[fd limit] could not raise: {e}")

    SEED      = 42
    NUM_ENVS  = 8
    NUM_ROUNDS = 1           # round-robin rounds (pilot + arm_operator each)
    STEPS_PER_ROUND = 1_000  # timesteps per agent per round; raise for real training
    CKPT_DIR = "./marl_debris_capture/"
    os.makedirs(CKPT_DIR, exist_ok=True)

    np.random.seed(SEED); random.seed(SEED); torch.manual_seed(SEED)
    set_random_seed(SEED)

    print("=" * 70)
    print(" ADR MARL - IPPO training")
    print("=" * 70)
    print("\n[init] fetching ephemeris and star catalog")
    load('de421.bsp')
    load.open(hipparcos.URL)
    print("[init] cached; each worker loads its own copy\n")

    pilot_latest_path = os.path.join(CKPT_DIR, "pilot_latest.zip")
    arm_latest_path   = os.path.join(CKPT_DIR, "arm_operator_latest.zip")

    # When resuming, existing checkpoints become the partners
    pilot_partner_path = arm_latest_path  if os.path.exists(arm_latest_path)  else None
    arm_partner_path   = pilot_latest_path if os.path.exists(pilot_latest_path) else None

    pilot_model = None
    arm_model   = None

    def _close_vec(vec):
        """Close the VecEnv and give the OS a moment to reclaim its fds."""
        try:
            vec.close()
        except Exception:
            pass
        gc.collect()
        time.sleep(1.5)

    for rnd in range(NUM_ROUNDS):
        print(f"\n{'='*70}")
        print(f" ROUND {rnd+1}/{NUM_ROUNDS}")
        print(f"{'='*70}")

        print(f"\n[Round {rnd+1}] Training: PILOT")
        print(f"          Partner path : {pilot_partner_path or 'random'}")

        vec_pilot = make_vec_env(
            "pilot",
            partner_model_path=pilot_partner_path,
            num_envs=NUM_ENVS, seed=SEED,
        )
        pilot_model = load_or_create(
            os.path.join(CKPT_DIR, "pilot"), vec_pilot, SEED
        )
        cb_pilot = CheckpointCallback(
            save_freq=STEPS_PER_ROUND // NUM_ENVS // 2,
            save_path=CKPT_DIR, name_prefix="pilot",
        )
        pilot_model.learn(
            total_timesteps=STEPS_PER_ROUND,
            callback=cb_pilot,
            reset_num_timesteps=(rnd == 0),
        )
        pilot_model.save(pilot_latest_path.replace(".zip", ""))
        # Drop the model before closing its env so the env can actually be collected
        del pilot_model
        _close_vec(vec_pilot)
        arm_partner_path = pilot_latest_path

        print(f"\n[Round {rnd+1}] Training: ARM_OPERATOR")
        print(f"          Partner path : {arm_partner_path}")

        vec_arm = make_vec_env(
            "arm_operator",
            partner_model_path=arm_partner_path,
            num_envs=NUM_ENVS, seed=SEED,
        )
        arm_model = load_or_create(
            os.path.join(CKPT_DIR, "arm_operator"), vec_arm, SEED
        )
        cb_arm = CheckpointCallback(
            save_freq=STEPS_PER_ROUND // NUM_ENVS // 2,
            save_path=CKPT_DIR, name_prefix="arm_operator",
        )
        arm_model.learn(
            total_timesteps=STEPS_PER_ROUND,
            callback=cb_arm,
            reset_num_timesteps=(rnd == 0),
        )
        arm_model.save(arm_latest_path.replace(".zip", ""))
        del arm_model
        _close_vec(vec_arm)
        pilot_partner_path = arm_latest_path

        print(f"\n[Round {rnd+1}] Checkpoints saved in {CKPT_DIR}")

    # Deterministic rollout of both policies, saved as a GIF
    print("\n" + "="*70)
    print(" training done, final rollout")
    print("="*70)

    import imageio

    pilot_model_final = PPO.load(pilot_latest_path, device="cpu")
    arm_model_final   = PPO.load(arm_latest_path,   device="cpu")

    st_test  = StarTrackerModel()
    test_env = ADRParallelEnv(star_tracker=st_test, render_mode="rgb_array")
    obs, _   = test_env.reset(seed=SEED)
    frames   = []

    for step_i in range(test_env.MAX_STEPS):
        p_act, _ = pilot_model_final.predict(obs["pilot"],        deterministic=True)
        a_act, _ = arm_model_final.predict(obs["arm_operator"],   deterministic=True)
        actions  = {"pilot": p_act, "arm_operator": a_act}

        obs, rews, terms, truncs, _ = test_env.step(actions)

        if step_i % 3 == 0:
            frame = test_env.render()
            if frame is not None:
                frames.append(frame)

        if all(terms.values()) or all(truncs.values()):
            status = "success" if test_env.mission_done else "failed"
            print(f"{status}: step {step_i+1}, phase {test_env.phase:.1f}")
            frame = test_env.render()
            if frame is not None:
                frames.append(frame)
            break

    if frames:
        gif_path = "marl_mission.gif"
        imageio.mimsave(gif_path, frames, fps=20)
        print(f"GIF saved: {gif_path}")

    test_env.render_mode = "human"
    test_env.render()
    test_env.close()