"""
HER-compatible Goal-conditioned PathPlanningEnv.

Changes vs. pathplanning_goal_env.py
--------------------------------------
1. Inherits from gymnasium_robotics.GoalEnv (which itself inherits gym.Env).
   SB3's HerReplayBuffer requires this interface.

2. observation_space is restructured into the three keys that HER expects:
     - "observation"   : the agent's own state  (x, y, t)
     - "achieved_goal" : what the agent has reached so far — here we use the
                         agent's current (x, y, t) position encoded identically
                         to the goal vector.
     - "desired_goal"  : the target runway FAF encoded as (goal_x, goal_y, rta)

3. compute_reward(achieved_goal, desired_goal, info) is implemented.

4. _get_obs() returns the new dict layout.

5. Everything else (bluesky sim, action modes, rendering) is unchanged.
"""

#TODO: Add RTA (Required Time of Arrival) to the goal vector, so that the agent can plan to arrive on time. Add a flag to specify whether to use RTA or keep it constant at 0.0. This will allow us to train agents that can not only reach the goal location, but also learn to time their arrival, which is crucial for real-world applications. 

import numpy as np
import pygame

from matplotlib.path import Path

import bluesky as bs
from bluesky_gym.envs.common.screen_dummy import ScreenDummy
import bluesky_gym.envs.common.functions as fn
from bluesky.traffic import Route

import gymnasium as gym
from gymnasium import spaces

from typing import List

class GoalEnv(gym.Env):
    """Abstract GoalEnv contract.  Subclasses must implement compute_reward() and _get_obs()."""
    def compute_reward(self, achieved_goal, desired_goal, infos):
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

FAF_DISTANCE = 25   # km
IAF_DISTANCE = 30   # km
IAF_ANGLE    = 60   # degrees

MIN_DISTANCE = FAF_DISTANCE + IAF_DISTANCE
MAX_DISTANCE = 300

MAX_TIME = 3600 * 6 # 6 hours in seconds, Note: This is just a random choice but it should be long enough!

MAX_DIS_NEXT_WPT = 15  # km
MIN_DIS_NEXT_WPT = 15  # km

SPEED    = 125   # m/s
ALTITUDE = 3000  # m
SIM_DT   = 5     # s
ACTION_TIME = 120 # s

ACTION_FREQUENCY = int(ACTION_TIME / SIM_DT)

GOAL_GRACE_LENGTH = 1000 * (IAF_DISTANCE - FAF_DISTANCE) #/ 2 # m
WRONG_GOAL_GRACE = GOAL_GRACE_LENGTH / SPEED  # seconds, tune based on overlap width / SPEED

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
        use_rta: bool = False,
    ):
        self.runways     = runways if runways is not None else ALL_RUNWAYS
        self.action_mode = action_mode
        self.use_rta     = use_rta

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
        self.segment_noise  = 0
        self.total_noise    = 0
        self.segment_length = 0
        self.total_length   = 0
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
        self.wrong_runway_timestamp = None

        # ── current goal (set properly in reset) ──────────────────────────────
        self.current_runway = self.runways[0]
        self.goal_vector    = self._compute_goal_vector(self.current_runway)

        self._set_terminal_conditions(self.runways)

        self.window = None
        self.clock  = None

    # ──────────────────────────────────────────────────────────────────────────
    # GoalEnv contract
    # ──────────────────────────────────────────────────────────────────────────
    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        infos: List[dict],
    ) -> np.ndarray:
        if self.use_rta:
            raise NotImplementedError("RTA-based reward is not implemented yet. Set use_rta=False or implement the RTA logic in compute_reward().")
        
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
        self.wrong_runway_timestamp = None

        # ── sample goal ────────────────────────────────────────────────────────
        self.current_runway = self.np_random.choice(self.runways)
        self.goal_vector    = self._compute_goal_vector(self.current_runway)

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
        t = 0.0 

        obs_vec = np.array([x, y, t], dtype=np.float64)

        return {
            "observation":   obs_vec,
            "achieved_goal": obs_vec.copy(),
            "desired_goal":  self.goal_vector.copy(),
        }

    def _compute_goal_vector(self, runway: str) -> np.ndarray:
        """Encodes the runway IAF as a 4-D vector (x, y, t, heading)."""
        rwy_info = RUNWAYS_SCHIPHOL_FAF[runway]
        
        # Target the IAF (Initial Approach Fix)
        # Note: IAF is usually further out than FAF. 
        # Here we use FAF_DISTANCE + IAF_DISTANCE to find the entry point.
        iaf_lat, iaf_lon = fn.get_point_at_distance(
            rwy_info["lat"], rwy_info["lon"],
            FAF_DISTANCE + IAF_DISTANCE,
            rwy_info["track"] - 180,
        )

        brg, dis = bs.tools.geo.kwikqdrdist(
            SCHIPHOL[0], SCHIPHOL[1], iaf_lat, iaf_lon
        )
        brg = np.radians(brg)
        dis = dis * NM2KM / MAX_DISTANCE

        x = np.sin(brg) * dis
        y = np.cos(brg) * dis
        t = 0.0 # Placeholder for RTA

        return np.array([x, y, t], dtype=np.float64)

    def _get_info(self) -> dict:
        obs = self._get_obs()
        is_success = self.death_cause == "success"

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
        self.terminated = False
        shapes = bs.tools.areafilter.basic_shapes
        line_ac = Path(np.array([[self.lat, self.lon], [bs.traf.lat[0], bs.traf.lon[0]]]))

        # Always check target runway first — success takes absolute priority
        target_sink = Path(np.reshape(shapes[f"SINK{self.current_runway}"].coordinates, (-1, 2)))
        if target_sink.intersects_path(line_ac):
            self.segment_reward += 10.0
            self.death_cause = "success"
            self.terminated = True
            self.lat = bs.traf.lat[0]
            self.lon = bs.traf.lon[0]
            return True
        
        if self.wrong_runway_timestamp is not None and (self.simt - self.wrong_runway_timestamp) > WRONG_GOAL_GRACE:
            self.segment_reward += -1.0
            self.death_cause = "wrong_runway"
            self.terminated = True
            self.lat = bs.traf.lat[0]
            self.lon = bs.traf.lon[0]
            return True

        # Check wrong runways with grace period
        for rwy in self.runways:
            line_sink = Path(np.reshape(shapes[f"SINK{rwy}"].coordinates, (-1, 2)))
            line_restrict = Path(np.reshape(shapes[f"RESTRICT{rwy}"].coordinates, (-1, 2)))

            if rwy != self.current_runway and line_sink.intersects_path(line_ac):
                if self.wrong_runway_timestamp is None:
                    self.wrong_runway_timestamp = self.simt
                
                self.lat = bs.traf.lat[0]
                self.lon = bs.traf.lon[0]
                return False  # still within grace period

            if line_restrict.intersects_path(line_ac) and self.wrong_runway_timestamp is None:
                self.segment_reward -= 1.0
                self.terminated = True
                self.death_cause = "restrict"
                self.lat = bs.traf.lat[0]
                self.lon = bs.traf.lon[0]
                return True

        self.lat = bs.traf.lat[0]
        self.lon = bs.traf.lon[0]
        return False

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
        spawn_distance = max(self.np_random.uniform(0, 0.9) * MAX_DISTANCE, MIN_DISTANCE)
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

        # --- Draw Runways ---
        for idx, rwy in enumerate(self.runways):
            is_goal        = rwy == self.current_runway
            arc_color      = (0,   0,   0)   if is_goal else (180, 180, 180)
            restrict_color = (255, 0,   0)   if is_goal else (220, 180, 180)
            width          = 3               if is_goal else 1

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

        # --- Draw Vector to Goal ---
        rwy_info = RUNWAYS_SCHIPHOL_FAF[self.current_runway]
        goal_lat, goal_lon = fn.get_point_at_distance(
            rwy_info["lat"], rwy_info["lon"],
            FAF_DISTANCE,
            rwy_info["track"] - 180,
        )

        g_qdr, g_dis = bs.tools.geo.kwikqdrdist(
            self.screen_coords[0], self.screen_coords[1], goal_lat, goal_lon
        )
        g_dis_km = g_dis * NM2KM
        x_goal = ((np.sin(np.deg2rad(g_qdr)) * g_dis_km) / (MAX_DISTANCE * 2)) * self.window_width
        y_goal = ((-np.cos(np.deg2rad(g_qdr)) * g_dis_km) / (MAX_DISTANCE * 2)) * self.window_height
        
        # Get the raw displacement in screen pixels
        dx = x_goal - x_ac
        dy = y_goal - y_ac
        dist = np.sqrt(dx**2 + dy**2)

        # 2. Normalize and Scale (only if dist > 0 to avoid division by zero)
        
        # Calculate the end point of the unit-direction vector
        if dist > 0:
            POINTER_LEN = 40  # Constant length in pixels
            ux = (dx / dist) * POINTER_LEN
            uy = (dy / dist) * POINTER_LEN
        else :
            ux, uy = 0, 0  # No direction if we're exactly at the goal
        x_end = x_ac + ux
        y_end = y_ac + uy

        pointer_color = (0, 120, 255) # Deep sky blue
        
        # 3. Draw the line
        pygame.draw.line(self.surface, pointer_color, (x_ac, y_ac), (x_end, y_end), 3)

        # 4. Add the Arrowhead so it looks like a proper vector
        angle = np.arctan2(uy, ux)
        arrow_size = 8
        arrow_angle = np.pi / 6 # 30 degrees
        
        # Right whisker
        pygame.draw.line(self.surface, pointer_color, (x_end, y_end), 
                        (x_end - arrow_size * np.cos(angle - arrow_angle), 
                        y_end - arrow_size * np.sin(angle - arrow_angle)), 3)
        # Left whisker
        pygame.draw.line(self.surface, pointer_color, (x_end, y_end), 
                        (x_end - arrow_size * np.cos(angle + arrow_angle), 
                        y_end - arrow_size * np.sin(angle + arrow_angle)), 3)

        # --- Draw Text Information ---
        # Calculate real-world distance based on our normalized observation vector
        obs = self._get_obs()
        dist_norm = np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"])
        dist_km = dist_norm * MAX_DISTANCE

        info_texts = [
            f"Runway: {self.current_runway}",
            f"sim time: {self.simt:.1f} s",
            f"Distance to Goal: {dist_km:.2f} km",
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