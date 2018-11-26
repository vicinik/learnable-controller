import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

class CartPoleEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max
            ])

        self.action_space = spaces.Box(low=-10.0, high=10.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None
        self._full_range = True

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def normalize_angle(self, angle):
        # Convert theta in the range [-PI, PI]
        n = abs(angle) // (2*math.pi)
        if (angle < 0):
            angle += n*2*math.pi
        else:
            angle -= n*2*math.pi

        if (angle < -math.pi):
            angle = 2*math.pi - abs(angle)
        elif (angle > math.pi):
            angle = -(2*math.pi - angle)

        thr_passed = False
        if abs(angle) > self.theta_threshold_radians:
            thr_passed = True

        return angle, thr_passed

    def normalize_position(self, x):
        thr_passed = False
        if (x > self.x_threshold):
            x -= self.x_threshold*2
            thr_passed = True
        elif (x < -self.x_threshold):
            x += self.x_threshold*2
            thr_passed = True

        return x, thr_passed

    def step(self, action):
        state = self.state
        x, x_dot, theta, theta_dot = state
        force = action
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta* temp) / (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc  = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        x  = x + self.tau * x_dot
        x, x_thr_passed = self.normalize_position(x)
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta, theta_thr_passed = self.normalize_angle(theta)
        theta_dot = theta_dot + self.tau * thetaacc
        self.state = (self._to_scalar(x),
                      self._to_scalar(x_dot),
                      self._to_scalar(theta),
                      self._to_scalar(theta_dot))

        reward = (np.pi/2 - abs(theta)) + (self.x_threshold/2 - abs(x))/2
        done = x_thr_passed or (not self._full_range and theta_thr_passed)

        return np.array(self.state), reward, done, {}

    def _to_scalar(self, x):
        if isinstance(x, np.ndarray):
            return np.asscalar(x)
        return x

    def reset(self):
        if self._full_range:
            # Full range training: pendulum will start from a completely random position
            self.state = [
                self.np_random.uniform(low=-0.2, high=0.2),
                self.np_random.uniform(low=-0.2, high=0.2),
                self.np_random.uniform(low=-np.pi, high=np.pi),
                self.np_random.uniform(low=-0.2, high=0.2),
            ]

            # For showcases: pendulum hangs down straight
            # self.state = [0.0, 0.0, np.pi, 0.0]
        else:
            # Train the upper position (keep the rod up)
            self.state = self.np_random.uniform(low=-0.2, high=0.2, size=(4,))

        self.steps_beyond_done = None
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold*2
        scale = screen_width/world_width
        carty = 150 # TOP OF CART
        polewidth = 10.0
        polelen = scale * self.length * 2
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
            axleoffset =cartheight/4.0
            cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
            pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            pole.set_color(.8,.6,.4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5,.5,.8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0,carty), (screen_width,carty))
            self.track.set_color(0,0,0)
            self.viewer.add_geom(self.track)

        if self.state is None: return None

        x = self.state
        cartx = x[0]*scale+screen_width/2.0 # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
