"""
HER-compatible Goal-conditioned PathPlanningEnv.

This module implements a Goal-Conditioned Reinforcement Learning (GCRL) environment 
for 4D path planning. The agent must navigate to a spatial Initial Approach Fix (IAF) 
while satisfying a Required Time of Arrival (RTA).

Core Logic & GCRL Strategy:
---------------------------
1. The 4D Goal (RTA): 
   While the physical simulation occurs in 2D space (x, y), the addition of a 
   temporal constraint (t) moves the problem into the '4D' domain (3D space + time). 
   Since the aircraft flies at a constant speed, the agent must learn 'path stretching' 
   manoeuvers to delay its arrival to meet specific RTA constraints.

2. GCRL Structure:
   The environment uses a Dict observation space compatible with 'gymnasium_robotics' 
   standards to separate the state from the targets:
     - 'observation':   The agent's current state (normalised x, y, t).
     - 'achieved_goal': The current state transformed into the goal format.
     - 'desired_goal':  The target destination and RTA (normalised x, y, t).

3. The Necessity of the Sampler:
   A sampler is used to provide a diverse distribution of target RTAs across episodes. 
   Without a sampler, the agent would only memorise a single trajectory; with it, 
   the agent learns a generalised policy capable of meeting any RTA within bounds.

4. Why Samplers are Critical for HER (The Reinterpretation Logic):
   Hindsight Experience Replay (HER) does not create new behaviour; it only reinterprets 
   existing behaviour. To schedule and sequence aircraft, the RTA must be part of the 
   'desired_goal'. HER then facilitates learning by saying: "You missed the target 
   time of 300s and arrived at 400s. But hey, now you know exactly how to fly a 
   path that takes 400s!" The sampler is critical because it forces the agent to 
   explore the various time-scales that HER needs to populate the replay buffer.

5. Training Compatibility:
   This structure is optimized for HER to solve sparse-reward challenges but 
   is fully compatible with no-HER training (e.g., standard SAC). The 'desired_goal' 
   acts as an additional input to the policy, allowing the agent to learn via 
   standard reward signals.
"""

import numpy as np
import pygame

from matplotlib.path import Path

import bluesky as bs
from bluesky_gym.envs.common.screen_dummy import ScreenDummy
from bluesky_gym.envs.common.base_sampler import BaseSampler
import bluesky_gym.envs.common.functions as fn
from bluesky.traffic import Route

import gymnasium as gym
from gymnasium import spaces

from typing import List


class GoalEnv(gym.Env):
    """
    Abstract GoalEnv contract required by Stable Baselines3 (SB3).
    
    To use Hindsight Experience Replay (HER), SB3 requires the environment 
    to implement this specific 'compute_reward' signature. This allows the 
    replay buffer to recalculate rewards offline when goals are relabeled.
    """
    def compute_reward(self, achieved_goal, desired_goal, infos):
        """
        Must return the reward for the given achieved and desired goals.
        Signature matches the 'gymnasium_robotics' standard used by SB3.
        """
        raise NotImplementedError

# ── shared constants ──────────────────────────────────────────────────────────
POPULATION_WEIGHT  = -1.0
PATH_LENGTH_WEIGHT = -0.0025

SCHIPHOL   = [52.3068953, 4.760783]
NM2KM      = 1.852
NM2M       = 1852.

RUNWAYS_SCHIPHOL_FAF = {
    "18C": {"lat": 52.301851, "lon": 4.737557, "track": 183},
    "36C": {"lat": 52.330937, "lon": 4.740026, "track":   3},
    "18L": {"lat": 52.291274, "lon": 4.777391, "track": 183},
    "36R": {"lat": 52.321199, "lon": 4.780119, "track":   3},
    "18R": {"lat": 52.329170, "lon": 4.708888, "track": 183},
    "36L": {"lat": 52.362334, "lon": 4.711910, "track":   3},
    "06":  {"lat": 52.304278, "lon": 4.776817, "track":  60},
    "24":  {"lat": 52.288020, "lon": 4.734463, "track": 240},
    "09":  {"lat": 52.318362, "lon": 4.796749, "track":  87},
    "27":  {"lat": 52.315940, "lon": 4.712981, "track": 267},
    "04":  {"lat": 52.313783, "lon": 4.802666, "track":  45},
    "22":  {"lat": 52.300518, "lon": 4.783853, "track": 225},
}

ALL_RUNWAYS = list(RUNWAYS_SCHIPHOL_FAF.keys())

OVERLAPPING_RUNWAYS = {
    "18C": ["18L", "18R"],
    "36C": ["36L", "36R"],
    "18L": ["18C", "18R"],
    "36R": ["36L", "36C"],
    "18R": ["18C", "18L"],
    "36L": ["36R", "36C"],
    "06":  ["04", "09"],
    "24":  ["22", "27"],
    "09":  ["06"],
    "27":  ["24"],
    "04":  ["06"],
    "22":  ["24"],
}

FAF_DISTANCE = 25   # km
IAF_DISTANCE = 30   # km
IAF_ANGLE    = 60   # degrees

MIN_DISTANCE = FAF_DISTANCE + IAF_DISTANCE
MAX_DISTANCE = 300. # km

MAX_TIME = 3600 * 6 # 6 hours in seconds, Note: This is just a random choice but it should be long enough!

MAX_DIS_NEXT_WPT = 15  # km
MIN_DIS_NEXT_WPT = 15  # km

SPEED    = 125   # m/s
ALTITUDE = 3000  # m
SIM_DT   = 5     # s
ACTION_TIME = 120 # s

ACTION_FREQUENCY = int(ACTION_TIME / SIM_DT)

DISTANCE_MARGIN = 4.5 # km    

RTA_TOLERANCE = 1 * 60 / MAX_TIME # 1 minutes

# ─────────────────────────────────────────────────────────────────────────────

class PathPlanningGoalEnv(GoalEnv):
    """
    HER-compatible goal-conditioned path planning.

    observation_space layout (required by HerReplayBuffer):
      {
        "observation"   : Box(3,)  — normalised (x, y, t) of the aircraft
        "achieved_goal" : Box(3,)  — same encoding as desired_goal, computed
                                     from current aircraft (x, y, t)
        "desired_goal"  : Box(3,)  — FAF position of the target runway,
                                     encoded as (sin_brg*dist, cos_brg*dist, rta)
      }

    Parameters
    ----------
    runways : list[str] | None
        Pool of runway IDs to sample from.  Defaults to all 12 runways.
    action_mode : str
        'wpt'  - agent outputs a (dx, dy) vector converted to a lat/lon waypoint.
        'hdg'  - agent outputs a (sin_hdg, cos_hdg) vector converted to a heading.
    """

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 1000}

    def __init__(
        self,
        render_mode=None,
        runways: list | None = None,
        action_mode: str = "hdg",
        rta_sampler: BaseSampler | None = None,
    ):
        self.runways     = runways if runways is not None else ALL_RUNWAYS
        self.action_mode = action_mode
        self._rta_sampler = rta_sampler

        if self._rta_sampler is not None:
            self._rta_sampler.min_fn = self._min_dist  # type: ignore

        self.window_width  = 512
        self.window_height = 512
        self.window_size   = (self.window_width, self.window_height)

        # ── observation space (GoalEnv layout) ────────────────────────────────
        obs_shape  = (3,)  # (x, y, t)
        goal_shape = (3,)  # (goal_x, goal_y, rta)
        act_shape  = (2,)  # (sin_hdg, cos_hdg) or (dx, dy)

        # Both goals use the same normalised encoding, so give them the same bounds
        goal_space = spaces.Box(-1.5, 1.5, shape=goal_shape, dtype=np.float64)

        self.observation_space = spaces.Dict({
            "observation":   spaces.Box(-1.5, 1.5, shape=obs_shape, dtype=np.float64),
            "achieved_goal": goal_space,
            "desired_goal":  goal_space,  # identical — not a copy, same object is fine
        })

        self.action_space = spaces.Box(-1, 1, shape=act_shape, dtype=np.float64)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # ── bluesky init ───────────────────────────────────────────────────────
        bs.init(mode="sim", detached=True)
        bs.scr = ScreenDummy()
        bs.stack.stack(f"DT {SIM_DT};FF")

        # ── population / grid data ─────────────────────────────────────────────
        self.pop_array = np.genfromtxt(
            "bluesky_gym/envs/data/population_1km.csv", delimiter=" "
        )
        self.x_array = np.genfromtxt(
            "bluesky_gym/envs/data/x_array.csv", delimiter=" "
        )
        self.y_array = np.genfromtxt(
            "bluesky_gym/envs/data/y_array.csv", delimiter=" "
        )
        self.x_max      = np.max(self.x_array)
        self.y_max      = np.max(self.y_array)
        self.cell_size  = 1000
        self.projection_size = 30

        # ── bookkeeping ────────────────────────────────────────────────────────
        self.step_reward   = 0
        self.segment_reward = 0
        self.total_reward   = 0
        self.population_weight  = POPULATION_WEIGHT
        self.path_length_weight = PATH_LENGTH_WEIGHT
        self.average_noise = 0
        self.average_path  = 0
        self.wpt_reach  = False
        self.terminated = False
        self.truncated  = False
        self.lat = 0
        self.lon = 0
        self.lat_list = []
        self.lon_list = []
        self.simt = 0
        self.death_cause = None

        # ── current goal (set properly in reset) ──────────────────────────────
        self.current_runway = self.runways[0]
        self.goal_vector    = self._compute_goal_vector(self.current_runway)

        # ── compute non-overlapping runways ──────────────────────────────────
        self._non_overlapping_runways = self._compute_non_overlapping_runways()

        self._set_terminal_conditions(self.runways)

        self.window = None
        self.clock  = None

    @property
    def use_rta(self) -> bool:
        return self._rta_sampler is not None

    # ──────────────────────────────────────────────────────────────────────────
    # GoalEnv contract
    # ──────────────────────────────────────────────────────────────────────────
    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        infos: List[dict],
    ) -> np.ndarray:
        """
        Computes the reward for a batch of transitions. Required by SB3's HerReplayBuffer.

        Called offline by the replay buffer with synthetic goals — desired_goal is
        replaced with what the agent actually achieved in a past episode, making the
        time error zero by construction.

        Why there is no RTA penalty here:
        ----------------------------------
        HER teaches RTA timing through goal relabelling, not penalisation. When the
        agent arrives at t=400s instead of the desired t=300s, HER creates a synthetic
        transition where desired_goal.t = 400s. Adding an RTA penalty here would
        always be zero on HER transitions and would conflict with the relabelling
        logic. The RTA penalty belongs only in _get_terminated(), where it shapes
        the live reward signal without interfering with HER.

        achieved_goal: the goal the agent actually reached, encoded as (x, y, t).
        desired_goal:  the HER-relabelled target goal, encoded as (x, y, t).
        infos:         per-transition dicts containing 'death_cause' and 'step_reward'
                    as populated by _get_info().
        """
        success = np.array([
            i.get("death_cause") in ("success", "wrong_runway") 
            for i in infos
        ])

        terminal_failure = np.array([
            i.get("death_cause") in ("restrict", "timeout", "out_of_bounds") 
            for i in infos
        ])

        step_rewards = np.array([i.get("step_reward", 0.0) for i in infos], dtype=np.float32)
        goal_reward = np.where(success, 10.0, 0.0)
        fail_penalty = np.where(terminal_failure, -1.0, 0.0)

        return (goal_reward + step_rewards + fail_penalty).astype(np.float32)

    # ──────────────────────────────────────────────────────────────────────────
    # Core Gymnasium API
    # ──────────────────────────────────────────────────────────────────────────

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        bs.traf.reset()

        self.average_noise  = 0
        self.average_path   = 0
        self.total_reward   = 0
        self.segment_reward = 0
        self.step_reward    = 0
        self.terminated     = False
        self.truncated      = False
        self.wpt_reach      = False
        self.simt           = 0
        self.death_cause    = None

        # ── spawn aircraft ─────────────────────────────────────────────────────
        spawn_lat, spawn_lon, spawn_heading = self._get_spawn()
        bs.traf.cre("kl001", "a320", spawn_lat, spawn_lon, spawn_heading, ALTITUDE, SPEED)
        acrte = Route._routes.get("kl001")
        acrte.delrte(0)

        bs.traf.ap.setdest(0, "EHAM")
        bs.traf.ap.setLNAV(0, True)
        bs.traf.ap.route[0].addwptMode(0, "FLYOVER")

        self.lat = bs.traf.lat[0]
        self.lon = bs.traf.lon[0]

        # ── sample goal ────────────────────────────────────────────────────────
        self.current_runway = self.np_random.choice(self.runways)
        self.goal_vector    = self._compute_goal_vector(self.current_runway)

        # ─── compute non-overlapping runways ──────────────────────────────────
        self._non_overlapping_runways = self._compute_non_overlapping_runways()

        observation = self._get_obs()
        info        = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        self.step_reward    = 0
        self.segment_reward = 0
        self._set_action(action)

        if self.action_mode == "wpt":
            while not self.wpt_reach:
                bs.sim.step()
                self.simt += bs.sim.simdt
                self._update_wpt_reach()
                self._update_reward()
                terminated = self._get_terminated()
                truncated  = self._get_truncated()
                if terminated or truncated:
                    break
                if self.render_mode == "human":
                    self._render_frame()
            self.wpt_reach = False

        elif self.action_mode == "hdg":
            for _ in range(ACTION_FREQUENCY):
                bs.sim.step()
                self.simt += bs.sim.simdt
                self._update_reward()
                terminated = self._get_terminated()
                truncated  = self._get_truncated()
                if terminated or truncated:
                    break
                if self.render_mode == "human":
                    self._render_frame()

        observation = self._get_obs()
        reward      = self._get_reward()
        self.total_reward += reward
        info        = self._get_info()

        return observation, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass

    # ──────────────────────────────────────────────────────────────────────────
    # Observation helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _get_obs(self) -> dict:
        """
        Returns the GoalEnv observation dict.

          observation   - normalised (x, y) of the aircraft (same as before)
          achieved_goal - aircraft position encoded *identically* to the goal
                          vector so compute_reward() can do a direct comparison
          desired_goal  - target runway FAF vector (constant within an episode)
        """
        brg, dis = bs.tools.geo.kwikqdrdist(
            SCHIPHOL[0], SCHIPHOL[1], bs.traf.lat[0], bs.traf.lon[0]
        )
        brg = np.radians(brg)
        dis = dis * NM2KM / MAX_DISTANCE

        x = np.sin(brg) * dis
        y = np.cos(brg) * dis
        t = self.simt / MAX_TIME

        obs_vec = np.array([x, y, t], dtype=np.float64)

        t_achieved = t if self.use_rta else 0.0

        achieved_vec = np.array([x, y, t_achieved], dtype=np.float64)

        return {
            "observation":   obs_vec,
            "achieved_goal": achieved_vec,
            "desired_goal":  self.goal_vector.copy(),
        }
    
    def _min_dist(self, X, runway: List[str] | str) -> float:
        rwy = runway[0] if isinstance(runway, list) else runway
        rwy_info = RUNWAYS_SCHIPHOL_FAF[rwy]
        
        iaf_lat, iaf_lon = fn.get_point_at_distance(
            rwy_info["lat"], rwy_info["lon"],
            FAF_DISTANCE + IAF_DISTANCE,
            rwy_info["track"] - 180,
        )

        _, dis = bs.tools.geo.kwikqdrdist(
            iaf_lat, iaf_lon, self.lat, self.lon
        )

        dis = dis * NM2KM
        dis = min(max(dis, 0), MAX_DISTANCE)

        return dis

    def _compute_goal_vector(self, runway: str) -> np.ndarray:
        """Encodes the runway IAF as a 3-D vector (x, y, t)."""
        rwy_info = RUNWAYS_SCHIPHOL_FAF[runway]
        
        # Target the IAF (Initial Approach Fix)
        # Note: IAF is usually further out than FAF. 
        # Here we use FAF_DISTANCE + IAF_DISTANCE to find the entry point.
        iaf_lat, iaf_lon = fn.get_point_at_distance(
            rwy_info["lat"], rwy_info["lon"],
            FAF_DISTANCE + IAF_DISTANCE,
            rwy_info["track"] - 180,
        )

        goal_brg, goal_dis = bs.tools.geo.kwikqdrdist(
            SCHIPHOL[0], SCHIPHOL[1], iaf_lat, iaf_lon
        )
        goal_brg = np.radians(goal_brg)
        goal_dis = goal_dis * NM2KM / MAX_DISTANCE

        goal_x = np.sin(goal_brg) * goal_dis
        goal_y = np.cos(goal_brg) * goal_dis

        ac_brg, ac_dis = bs.tools.geo.kwikqdrdist(
            SCHIPHOL[0], SCHIPHOL[1], self.lat, self.lon
        )

        ac_brg = np.radians(ac_brg)
        ac_dis = ac_dis * NM2KM / MAX_DISTANCE

        """
        ac_x = np.sin(ac_brg) * ac_dis
        ac_y = np.cos(ac_brg) * ac_dis
        """

        goal_t = 0.0
        if self.use_rta:
            goal_dist = self._rta_sampler.sample(np.array([ac_dis, ac_brg]), runway)  # type: ignore
            slack = 100 * self.np_random.uniform(0, 1)  # add some random slack to make it less deterministic
            goal_dist = 1000 * (goal_dist + slack)  # convert from km to m and add slack
            goal_t = goal_dist / SPEED / MAX_TIME  # convert distance to time and normalise

        return np.array([goal_x, goal_y, goal_t], dtype=np.float64)

    def _get_info(self) -> dict:
        on_time = abs(self.goal_vector[2] - (self.simt / MAX_TIME)) <= RTA_TOLERANCE if self.use_rta else True
        correct_runway = (self.death_cause == "success")

        is_success = on_time and correct_runway

        hdg = np.radians(bs.traf.hdg[0])

        return {
            "is_success":         is_success,   # required by GoalSuccessLoggerCallback
            "death_cause":        self.death_cause,
            "sim_time":           self.simt,
            "step_reward":        self.step_reward,
            "total_reward":       self.total_reward,
            "average_path_rew":   self.average_path,
            "average_noise_rew":  self.average_noise,
            "population_weight":  self.population_weight,
            "path_length_weight": self.path_length_weight,
            "current_runway":     self.current_runway,
            "goal_vector":        self.goal_vector.tolist(),
            "on_time":            on_time,
            "correct_runway":     correct_runway,
            "heading":                hdg,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Reward helpers (dense, used during live rollout)
    # ──────────────────────────────────────────────────────────────────────────

    def _get_reward(self):
        return self.segment_reward

    def _update_reward(self):
        path_length         = self._get_path_length()         * self.path_length_weight
        population_exposure = self._get_population_exposure() * self.population_weight
        self.average_path  += path_length
        self.average_noise += population_exposure
        
        tick_reward = path_length + population_exposure
        self.step_reward    += tick_reward
        self.segment_reward += tick_reward

    # ──────────────────────────────────────────────────────────────────────────
    # Terminal conditions
    # ──────────────────────────────────────────────────────────────────────────

    def _get_terminated(self):  
        def _get_rta_penalty_mult():
            if not self.use_rta:
                return 1.0
            
            abs_x = abs(self.goal_vector[2] - (self.simt / MAX_TIME))
            if abs_x <= RTA_TOLERANCE:
                return 1.0 - (abs_x / RTA_TOLERANCE)**2  # quadratic penalty within the tolerance window
            return 0.0


        self.terminated = False
        shapes = bs.tools.areafilter.basic_shapes
        line_ac = Path(np.array([[self.lat, self.lon], [bs.traf.lat[0], bs.traf.lon[0]]]))

        self.lat = bs.traf.lat[0]
        self.lon = bs.traf.lon[0]

        target_sink = Path(np.reshape(shapes[f"SINK{self.current_runway}"].coordinates, (-1, 2)))
        if target_sink.intersects_path(line_ac):
            self.segment_reward += (10.0 * _get_rta_penalty_mult())
            self.death_cause = "success"
            self.terminated = True
            return self.terminated
        
        target_restrict = Path(np.reshape(shapes[f"RESTRICT{self.current_runway}"].coordinates, (-1, 2)))
        if target_restrict.intersects_path(line_ac):
            self.segment_reward += -1.0
            self.death_cause = "restrict"
            self.terminated = True
            return self.terminated
        
        for rwy in self._non_overlapping_runways:
            rwy_sink = Path(np.reshape(shapes[f"SINK{rwy}"].coordinates, (-1, 2)))
            if rwy_sink.intersects_path(line_ac):
                self.segment_reward += -1.0
                self.death_cause = "wrong_runway"
                self.terminated = True
                return self.terminated
            
            rwy_restrict = Path(np.reshape(shapes[f"RESTRICT{rwy}"].coordinates, (-1, 2)))
            if rwy_restrict.intersects_path(line_ac):
                self.segment_reward += -1.0
                self.death_cause = "restrict"
                self.terminated = True
                return self.terminated

        return self.terminated

    def _get_truncated(self):
        if self.simt >= MAX_TIME:
            self.truncated = True
            self.segment_reward += -1
            self.death_cause = "timeout"
            return self.truncated

        dis_origin = (
            bs.tools.geo.kwikdist(
                SCHIPHOL[0], SCHIPHOL[1], bs.traf.lat[0], bs.traf.lon[0]
            )
            * NM2KM
        )
        if dis_origin > MAX_DISTANCE * 1.05:
            self.truncated = True
            self.segment_reward += -1
            self.death_cause = "out_of_bounds"
        return self.truncated

    # ──────────────────────────────────────────────────────────────────────────
    # Action
    # ──────────────────────────────────────────────────────────────────────────

    def _set_action(self, action):
        if self.action_mode == "wpt":
            distance = max(
                max(abs(action[0] * MAX_DIS_NEXT_WPT), abs(action[1] * MAX_DIS_NEXT_WPT)),
                MIN_DIS_NEXT_WPT,
            )
            bearing  = np.rad2deg(np.arctan2(action[0], action[1]))
            ac_lat   = bs.traf.lat[0]
            ac_lon   = bs.traf.lon[0]
            new_lat, new_lon = fn.get_point_at_distance(ac_lat, ac_lon, distance, bearing)
            bs.traf.ap.route[0].addwptStack(0, f"{new_lat}, {new_lon}")

        elif self.action_mode == "hdg":
            bearing = np.rad2deg(np.arctan2(action[0], action[1]))
            bs.traf.ap.selhdgcmd(0, bearing)

    # ──────────────────────────────────────────────────────────────────────────
    # Physics helpers (unchanged)
    # ──────────────────────────────────────────────────────────────────────────

    def _update_wpt_reach(self):
        acrte = Route._routes.get("kl001")
        if bs.traf.actwp.lat[0] == acrte.wplat[-1]:
            self.wpt_reach = True

    def _get_path_length(self):
        return bs.traf.tas[0] * SIM_DT / 1852.0

    def _get_population_exposure(self):
        brg, dist = bs.tools.geo.kwikqdrdist(
            SCHIPHOL[0], SCHIPHOL[1], bs.traf.lat[0], bs.traf.lon[0]
        )
        x = np.sin(np.radians(brg)) * dist * NM2M
        y = np.cos(np.radians(brg)) * dist * NM2M
        z = bs.traf.alt[0]

        x_index_min = int(((x + self.x_max) / self.cell_size) - self.projection_size)
        x_index_max = int(((x + self.x_max) / self.cell_size) + self.projection_size)
        y_index_min = int(((self.y_max - y) / self.cell_size) - self.projection_size)
        y_index_max = int(((self.y_max - y) / self.cell_size) + self.projection_size)

        distance2 = (
            (self.x_array[y_index_min:y_index_max, x_index_min:x_index_max] - x) ** 2
            + (self.y_array[y_index_min:y_index_max, x_index_min:x_index_max] - y) ** 2
            + z ** 2
        )
        return np.sum(
            self.pop_array[y_index_min:y_index_max, x_index_min:x_index_max] / distance2
        )

    def _get_spawn(self):
        spawn_bearing  = self.np_random.uniform(0, 360)
        min_spawn_dist = (MIN_DISTANCE + DISTANCE_MARGIN) / MAX_DISTANCE
        spawn_distance = self.np_random.uniform(min_spawn_dist, 0.95) * MAX_DISTANCE
        spawn_lat, spawn_lon = fn.get_point_at_distance(
            SCHIPHOL[0], SCHIPHOL[1], spawn_distance, spawn_bearing
        )
        spawn_heading = (spawn_bearing + 180 + 360) % 360
        return spawn_lat, spawn_lon, spawn_heading

    def _set_terminal_conditions(self, runway_list):
        self.line_arc_pg      = []
        self.line_restrict_pg = []

        for rwy in runway_list:
            num_points = 36
            rwy_info   = RUNWAYS_SCHIPHOL_FAF[rwy]

            faf_lat, faf_lon = fn.get_point_at_distance(
                rwy_info["lat"], rwy_info["lon"],
                FAF_DISTANCE,
                rwy_info["track"] - 180,
            )

            cw_bound  = ((rwy_info["track"] - 180 + 360) % 360) + (IAF_ANGLE / 2)
            ccw_bound = ((rwy_info["track"] - 180 + 360) % 360) - (IAF_ANGLE / 2)
            angles    = np.linspace(cw_bound, ccw_bound, num_points)
            lat_iaf, lon_iaf = fn.get_point_at_distance(faf_lat, faf_lon, IAF_DISTANCE, angles)

            command = f"POLYLINE SINK{rwy}"
            for i in range(len(lat_iaf)):
                command += f" {lat_iaf[i]} {lon_iaf[i]}"
            bs.stack.stack(command)
            bs.stack.stack(
                f"POLYLINE RESTRICT{rwy} {lat_iaf[0]} {lon_iaf[0]} "
                f"{faf_lat} {faf_lon} {lat_iaf[-1]} {lon_iaf[-1]}"
            )
            bs.stack.stack(f"COLOR RESTRICT{rwy} red")

            if self.render_mode == "human":
                env_max_distance = np.sqrt(MAX_DISTANCE**2 + MAX_DISTANCE**2)
                lat_ref, lon_ref = bs.tools.geo.kwikpos(
                    SCHIPHOL[0], SCHIPHOL[1], 315, env_max_distance / NM2KM
                )
                self.screen_coords = [lat_ref, lon_ref]

                coords = np.empty(2 * num_points, dtype=np.float32)
                coords[0::2] = lat_iaf
                coords[1::2] = lon_iaf
                line_arc      = np.reshape(coords, (len(coords) // 2, 2))
                line_restrict = np.array(
                    [[lat_iaf[0], lon_iaf[0]], [faf_lat, faf_lon], [lat_iaf[-1], lon_iaf[-1]]]
                )

                qdr, dis = bs.tools.geo.kwikqdrdist(
                    self.screen_coords[0], self.screen_coords[1],
                    line_arc[:, 0], line_arc[:, 1],
                )
                dis   = dis * NM2KM
                x_arc = ((np.sin(np.deg2rad(qdr)) * dis) / (MAX_DISTANCE * 2)) * self.window_width
                y_arc = ((-np.cos(np.deg2rad(qdr)) * dis) / (MAX_DISTANCE * 2)) * self.window_width
                self.line_arc_pg.append([(float(x), float(y)) for x, y in zip(x_arc, y_arc)])

                qdr, dis = bs.tools.geo.kwikqdrdist(
                    self.screen_coords[0], self.screen_coords[1],
                    line_restrict[:, 0], line_restrict[:, 1],
                )
                dis        = dis * NM2KM
                x_restrict = ((np.sin(np.deg2rad(qdr)) * dis) / (MAX_DISTANCE * 2)) * self.window_width
                y_restrict = ((-np.cos(np.deg2rad(qdr)) * dis) / (MAX_DISTANCE * 2)) * self.window_width
                self.line_restrict_pg.append(
                    [(float(x), float(y)) for x, y in zip(x_restrict, y_restrict)]
                )

    def _compute_non_overlapping_runways(self):
        """
        Compute a list of non-overlapping runways, excluding the current runway
        """
        runway_overlaps = OVERLAPPING_RUNWAYS.get(self.current_runway, [])
        return [
            rwy for rwy in self.runways
            if rwy != self.current_runway
            and rwy not in runway_overlaps
        ]

    def _render_frame(self):
        # Initialize Pygame, Window, Surface, and Fonts exactly once
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            pygame.font.init() # Ensure font module is loaded
            
            self.window = pygame.display.set_mode(self.window_size)
            self.surface = pygame.Surface(self.window_size)
            self.font = pygame.font.SysFont("Arial", 10)

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        # Handle Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return # Must return immediately to avoid drawing to a dead display

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return
                
                # Pause on spacebar
                if event.key == pygame.K_SPACE:
                    paused = True
                    while paused:
                        for pause_event in pygame.event.get():
                            if pause_event.type == pygame.KEYDOWN and pause_event.key == pygame.K_SPACE:
                                paused = False
                            elif pause_event.type == pygame.QUIT:
                                pygame.quit()
                                return
                        # Prevent 100% CPU usage while paused
                        self.clock.tick(15) 

        # Clear the reused surface
        self.surface.fill((255, 255, 255))

        overlapping_rwy = OVERLAPPING_RUNWAYS.get(self.current_runway, [])
        # --- Draw Runways ---
        for idx, rwy in enumerate(self.runways):
            is_goal        = rwy == self.current_runway
            is_overlapping = rwy in overlapping_rwy

            if is_goal:
                arc_color      = (0, 0, 0)
                restrict_color = (255, 0, 0)
                width          = 3

            elif is_overlapping:
                arc_color      = (180, 100, 100)   # slightly reddish gray
                restrict_color = (255, 120, 120)   # softer red
                width          = 2

            else:
                arc_color      = (180, 180, 180)
                restrict_color = (220, 180, 180)
                width          = 1

            if idx < len(self.line_arc_pg):
                pygame.draw.lines(self.surface, arc_color,      False, self.line_arc_pg[idx],      width)
                pygame.draw.lines(self.surface, restrict_color, False, self.line_restrict_pg[idx], max(1, width - 1))

        # --- Draw Aircraft ---
        ac_lat, ac_lon = bs.traf.lat[0], bs.traf.lon[0]
        qdr, dis = bs.tools.geo.kwikqdrdist(
            self.screen_coords[0], self.screen_coords[1], ac_lat, ac_lon
        )
        dis  = dis * NM2KM
        x_ac = ((np.sin(np.deg2rad(qdr)) * dis) / (MAX_DISTANCE * 2)) * self.window_width
        y_ac = ((-np.cos(np.deg2rad(qdr)) * dis) / (MAX_DISTANCE * 2)) * self.window_height
        pygame.draw.circle(self.surface, (0, 0, 0), (x_ac, y_ac), 5)

        # --- Draw Waypoint ---
        wpt_lat, wpt_lon = bs.traf.actwp.lat[0], bs.traf.actwp.lon[0]
        qdr, dis = bs.tools.geo.kwikqdrdist(
            self.screen_coords[0], self.screen_coords[1], wpt_lat, wpt_lon
        )
        dis   = dis * NM2KM
        x_wpt = ((np.sin(np.deg2rad(qdr)) * dis) / (MAX_DISTANCE * 2)) * self.window_width
        y_wpt = ((-np.cos(np.deg2rad(qdr)) * dis) / (MAX_DISTANCE * 2)) * self.window_height
        pygame.draw.circle(self.surface, (255, 0, 0), (x_wpt, y_wpt), 5)

        # --- Draw Heading Line ---
        hdg = bs.traf.hdg[0]  # degrees, 0=North
        HDG_LEN = 20  # pixels
        hx = HDG_LEN *  np.sin(np.deg2rad(hdg))
        hy = HDG_LEN * -np.cos(np.deg2rad(hdg))
        pygame.draw.line(
            self.surface, (0, 0, 0),
            (x_ac, y_ac), (x_ac + hx, y_ac + hy), 2
        )
        
        # --- Draw Text Information ---
        # Calculate real-world distance based on our normalized observation vector
        obs = self._get_obs()
        dist_norm = np.linalg.norm(obs["achieved_goal"][:2] - obs["desired_goal"][:2]) # Both are (x, y, t) and we only care about (x, y)
        dist_km = dist_norm * MAX_DISTANCE

        time_to_go = obs["desired_goal"][2] - obs["achieved_goal"][2]
        time_to_go *= MAX_TIME

        info_texts = [
            f"Runway: {self.current_runway}",
            f"sim time: {self.simt:.1f} s",
            f"Distance to Goal: {dist_km:.2f} km",
            f"Time to Goal: {time_to_go:.2f} s",
            f"Segment Reward: {self.segment_reward:.2f}",
            f"Total Reward: {self.total_reward:.2f}"
        ]

        # Blit each line of text with a 25px vertical offset
        for i, text in enumerate(info_texts):
            text_surf = self.font.render(text, True, (0, 0, 0)) # True = anti-aliased, (0,0,0) = Black
            self.surface.blit(text_surf, (15, 15 + (i * 10)))

        # Blit everything to the main window and update
        self.window.blit(self.surface, (0, 0))
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])