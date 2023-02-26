import habitat_sim
from agents.sem_exp import Sem_Exp_Env_Agent
from habitat_sim.simulator import Simulator
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
import numpy as np
from utils.integration import visualize, Collector
import torch
from os import path
import cv2
from itertools import product
from tqdm.auto import tqdm

class TopdownScanner(Sem_Exp_Env_Agent):
    def __init__(self, args, rank, config_env, dataset):
        super().__init__(args, rank, config_env, dataset)
        self.sim:Simulator = None 
        self.nonsemantic_topdown = None
        self.follower: ShortestPathFollower = None
        self.next_pos: np.ndarray = None
        self.rotate_cnt: int = None
        self.turn_angle = args.turn_angle

        self.meter_per_px = args.map_resolution / 100
        self.full_map_size = args.map_size_cm / args.map_resolution
        self.num_scan_part = args.num_scan_part
        self.start_pos: np.ndarray = None
        self.scan_done = False

        self.finished_names = []
        self.all_finished = False
        self.last_dist = 100000
        self.try_cnt = 0
        self.max_try = 15
        self.abandon_dist = 0
        self.last_action = 0


    def _ij_to_coord(self, i):
        return (i - self.num_scan_part / 2) * self.full_map_size / self.num_scan_part * self.meter_per_px
    def _navigable_near(self, center: np.ndarray, step: int, n_step: int, height_step: int=1):
        for step_x in range(-int(n_step//2), int(n_step//2) + 1):
            for step_y in range(-int(n_step // 2), int(n_step // 2) + 1):
                pos = center + np.array([step_x * step, 0, step_y * step])
                if self.sim.pathfinder.is_navigable(pos, max_y_delta=height_step):
                    return pos
        return None
    def _navigable_near_random(self, center: np.ndarray, rand_range: float, num_max_tries: int=1000, height_step: int=1):
        # print(center, rand_range)
        for _ in range(num_max_tries):
            dx = np.random.rand() - 0.5
            dy = np.random.rand() - 0.5
            pos = center + np.array([dx, 0, dy]) * rand_range
            if self.sim.pathfinder.is_navigable(pos, max_y_delta=height_step):
                return pos
            
        return None

    def _square_scan_generator(self):
        scene_name = self.habitat_env.sim.config.SCENE
        failure_count = 0
        for i, j in tqdm(product(range(self.num_scan_part), range(self.num_scan_part)), scene_name):
            d = [self._ij_to_coord(i), 0, self._ij_to_coord(j)]
            ideal_pos = self.start_pos + np.array(d)
            res:np.ndarray = self._navigable_near_random(
                center=ideal_pos,
                rand_range=self.full_map_size / self.num_scan_part * self.meter_per_px
            )
            # print(scene_name, ideal_pos, res)
            self.last_dist = 1000000
            self.try_cnt = self.max_try
            self.abandon_dist = 0
            self.last_action=2
            if res is None:
                failure_count += 1
                continue
            yield res
        print(scene_name, 'failure:', int(failure_count / self.num_scan_part ** 2 * 100), '%')

    
    def _update_sim(self):
        self.sim = self.habitat_env.sim._sim
        #assert 0, (self.sim.__class__, self.sim.__dir__())
        self.scan_pos = self._square_scan_generator()
        self.start_pos = self._current_pos()
        self.scan_done = False
        self.next_pos = self._decides_next_pos()
    def _update_follower(self):
        self.follower= ShortestPathFollower(
            sim=self.habitat_env.sim,
            goal_radius=1,
            return_one_hot=False,
        )
    def reset(self):
        res = super().reset()
        if self.scene_name in self.finished_names:
            self.all_finished = True
        self._update_sim()
        self._update_follower()
        return res
    def _finish_scene(self):
        self.scan_done = True
        self.finished_names.append(self.scene_name)
    def _decides_next_pos(self):
        try:
            return next(self.scan_pos)
        except StopIteration:
            scene_name = self.habitat_env.sim.config.SCENE
            print(scene_name, 'average abandon dist:', self.abandon_dist / (self.num_scan_part**2))
            self._finish_scene()
            return self.start_pos
    def _current_pos(self) -> np.ndarray:
        agent_state = self.sim.get_agent(0).get_state()
        return agent_state.position
    def _total_rotate(self) -> int:
        return (360 + self.turn_angle - 1) // self.turn_angle
    def _plan_rotate(self) -> int:
        self.rotate_cnt -= 1
        return 2
    def _process_done(self, done):
        scene_name = self.habitat_env.sim.config.SCENE
        return bool(self.scan_done) 
    def _abandon(self):
        scene_name = self.habitat_env.sim.config.SCENE
        print(scene_name, 'tries used up, abandoning at', self._current_pos())
        dist = np.linalg.norm((self.next_pos - self._current_pos()) * np.array([1,0,1]))
        cnt = 0
        while dist > 0.5 and cnt < 10:
            agent_state = self.sim.get_agent(0).get_state()
            agent_state.position = self.next_pos
            self.sim.get_agent(0).set_state(agent_state)

            cnt += 1
            dist = np.linalg.norm((self.next_pos - self._current_pos()) * np.array([1,0,1]))

        self.next_pos = self._current_pos()
        print(scene_name, 'jump to', self.next_pos)
        # self.next_pos = self._current_pos()
    def _plan(self, planner_inputs):
        scene_name = self.habitat_env.sim.config.SCENE
        if self.all_finished:
            return 2
        if self.next_pos is None:
            self.next_pos = self._decides_next_pos()
        last_dist = self.last_dist
        dist = np.linalg.norm((self.next_pos - self._current_pos()) * np.array([1,0,1])) # 地面不平，如果不忽略高度的话会反复传送到目标上空
        self.last_dist = dist
        # print(scene_name, "posistion", self._current_pos())
        if dist <= 0.5:
            if self.rotate_cnt is None:
                print(scene_name, 'having reached at', self._current_pos())
                print(scene_name, 'dist', dist)
                self.rotate_cnt = self._total_rotate()
            assert self.rotate_cnt > 0
            print(scene_name, 'rotating:', self.rotate_cnt)
            self.last_action = self._plan_rotate()
            if self.rotate_cnt <= 0:
                print(scene_name, 'rotation done')
                self.rotate_cnt = None
                self.next_pos = self._decides_next_pos()
            return self.last_action
        else:
            if np.abs(dist - last_dist) < 0.2:
                self.try_cnt -= 1
            else:
                self.try_cnt = self.max_try
            if self.try_cnt <= 0:
                self.abandon_dist += dist
                self._abandon()
                print(scene_name, 'abandon dist:', dist)
            self.last_action = self.follower.get_next_action(self.next_pos)
            return self.last_action
        
class TopdownCollector(Collector):
    def __init__(self, scene_names: 'list[str]', args) -> None:
        super().__init__(
            scene_names=scene_names, 
            path="{}/dump/{}/topdown/".format(args.dump_location, args.exp_name)
        )
        self.finished_names = []
        self.preview_size = args.preview_size

    def _preview(self):
        return self.preview_size > 0
    
    def _collect(self, i: int, old_full_map: torch.Tensor):
        self.finished_names.append(self.names[i])

        obstacle, mask, semantic =old_full_map[[0]].clamp(min=0, max=1).detach(), old_full_map[[1]].clamp(min=0, max=1).detach(), old_full_map[4:].detach()
        semantic = torch.cat([semantic, obstacle], dim=0)
        tensor_name = path.join(self.path, self.names[i] + '-sparse' + '.topdown')
        torch.save(semantic.to_sparse(), tensor_name)
        if self._preview():
            vis = visualize(torch.cat([mask, semantic]), self.preview_size)
            img_name = path.join(self.path, self.names[i] + '-' + str(i) + '.jpg')
            cv2.imwrite(img_name, vis)

    def update(self, i: int, new_name: str, old_full_map: torch.Tensor):
        self._collect(i, old_full_map)
        return super().update(i, new_name)

class WrappedTopdownCollector:
    def __init__(self, scene_names: 'list[str]', args, started=False) -> None:
        if started:
            self.collector = TopdownCollector(scene_names, args)
        else:
            self.collector = None
    def update(self, *args, **kwargs):
        if self.collector is not None:
            return self.collector.update(*args, **kwargs)
        else:
            return None
    def __getattr__(self, __name: str):
        if self.collector is not None:
            return getattr(self.collector, __name)
        else:
            return None