import math
import config
import random
import numpy as np
from queue import PriorityQueue
from itertools import chain

class Node_base:
    def __init__(self, disk, bandwidth, x, y):
        self.type = "base"

        self.disk = disk
        self.bandwidth = bandwidth

        self.task_queue = PriorityQueue()
        self.download_finish_time = 0
        self.image_list = [0] * config.IMAGE_NUM
        self.image_download_time = [0] * config.IMAGE_NUM

        self.x = x
        self.y = y


class Node_CPU(Node_base):
    def __init__(self, cpu_freq, mem, disk, bandwidth, x, y):
        super().__init__(disk, bandwidth, x, y)
        self.type = "CPU"

        self.cpu_freq = cpu_freq
        self.mem = mem


class Node_GPU(Node_base):
    def __init__(self, computing_power, gpu_mem, gpu_band, disk, bandwidth, x, y):
        super().__init__(disk, bandwidth, x, y)
        self.type = "GPU"

        self.computing_power = computing_power
        self.gpu_mem = gpu_mem
        self.gpu_band = gpu_band


class Image:
    def __init__(self, image_size):
        self.image_size = image_size


class Task:
    def __init__(self, task_type, task_size, image_id, mem, cpu_cycle, flops, gpu_mem, start_time, ddl, x, y):
        self.type = task_type
        self.task_size = task_size
        self.image_id = image_id

        self.mem = mem
        self.cpu_cycle = cpu_cycle
        self.flops = flops
        self.gpu_mem = gpu_mem

        self.start_time = start_time
        self.ddl = ddl
        self.x = x
        self.y = y

        self.cpu_freq_using = 0


class Env:
    def __init__(self):
        self.seed = config.RANDOM_SEED
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.__init_env()

    def seed(self, seed):
        self.seed = seed

    def __init_env(self):
        self.node_cpu_num = config.EDGE_NODE_NUM_CPU
        self.node_gpu_num = config.EDGE_NODE_NUM_GPU
        self.node_num = self.node_cpu_num + self.node_gpu_num
        self.image_num = config.IMAGE_NUM
        # self.n_observations = 5 * self.node_num + 2 * self.node_num + 1 + 5  # The length of #state
        self.n_observations = 5 * self.node_cpu_num + 6 * self.node_gpu_num + 9

        self.node = []
        self.node_cpu = []
        self.node_gpu = []
        self.image = []
        self.task = []

        self.time = -1
        self.reward = 0
        self.rewards = []

        self.total_time = 0
        self.download_time = 0
        self.trans_time = 0
        self.comp_time = 0

        self.task_finish_list = []
        self.num_on_time = 0
        self.total_task = 0
        self.done = False  # Whether all tasks are completed

    def reset(self):
        self.__init_env()

        # 1. create nodes
        for _ in range(self.node_cpu_num):
            self.node.append(Node_CPU(
                random.randint(config.node_cpu_freq_min, config.node_cpu_freq_max),
                random.randint(config.node_mem_min, config.node_mem_max),
                random.randint(config.node_disk_min, config.node_disk_max),
                random.uniform(config.node_band_min, config.node_band_max),

                random.random() * config.max_x,
                random.random() * config.max_y))

        for _ in range(self.node_gpu_num):
            self.node.append(Node_GPU(
                random.randint(config.node_gpu_computing_power_min, config.node_gpu_computing_power_max),

                random.randint(config.node_gpu_mem_min, config.node_gpu_mem_max),
                random.randint(config.node_gpu_band_min, config.node_gpu_band_max),
                random.randint(config.node_disk_min, config.node_disk_max),
                random.uniform(config.node_band_min, config.node_band_max),

                random.random() * config.max_x,
                random.random() * config.max_y))


        for _ in range(self.image_num):
            self.image.append(Image(
                random.uniform(config.image_size_min, config.image_size_max)))

        for i in range(2000):
            for _ in range(random.randint(config.min_tasks, config.max_tasks)):
                randn = int(np.random.normal(self.image_num // 2, 8, 1))
                image_id = randn if 0 <= randn <= self.image_num - 1 else random.randint(0, self.image_num - 1)

                if random.getrandbits(1):
                    task_type = "CPU"
                    cpu_cycle = random.random() * config.task_cpu_freq_max
                    mem = random.random() * config.task_mem_max
                    flops = 0
                    gpu_mem = 0
                else:
                    task_type = "GPU"
                    mem = random.random() * config.task_mem_max
                    flops = random.random() * config.task_gpu_computing_power_max
                    cpu_cycle = (random.random() + 4) * flops
                    gpu_mem = random.random() * config.task_gpu_mem_max

                ddl = i + random.randint(3, 5)
                # ddl=i
                self.task.append(Task(
                    task_type=task_type, image_id=image_id, mem=mem, cpu_cycle=cpu_cycle,
                    flops=flops, gpu_mem=gpu_mem,
                    start_time=i, ddl=ddl,
                    x=random.random() * config.max_x,
                    y=random.random() * config.max_y,
                    task_size=random.random() * config.task_size_max / 1000))
                self.total_task += 1

    def image_download(self, task, idx):
        if self.node[idx].image_list[task.image_id] == 0:

            self.node[idx].image_list[task.image_id] = 1
            self.node[idx].disk -= self.image[task.image_id].image_size

            download_time = self.image[task.image_id].image_size / (
                        2 * self.node[idx].bandwidth)  # Time required to download: Time period
            self.node[idx].image_download_time[task.image_id] = max(self.node[idx].download_finish_time, self.time) + download_time  # Download completion time: point in time

            self.node[idx].download_finish_time = self.node[idx].image_download_time[task.image_id]
            download_finish_time = self.node[idx].image_download_time[task.image_id]

        elif self.node[idx].image_list[task.image_id] == 1:
            download_finish_time = self.node[idx].image_download_time[task.image_id]
        else:
            download_finish_time = max(0, self.time)

        return download_finish_time

    def _add_task(self, tasks, actions, t_on_n):
        rewards = []

        for i in range(len(tasks)):
            task = tasks[i]
            idx = actions[i]
            if self.node[idx].image_list[task.image_id] == 0:
                self.node[idx].image_list[task.image_id] = 1
                self.node[idx].disk -= self.image[task.image_id].image_size
                download_time = self.image[task.image_id].image_size / (2 * self.node[idx].bandwidth)
                self.node[idx].image_download_time[task.image_id] = max(self.node[idx].download_finish_time, self.time) + download_time
                self.node[idx].download_finish_time = self.node[idx].image_download_time[task.image_id]
                download_finish_time = self.node[idx].image_download_time[task.image_id]
            elif self.node[idx].image_list[task.image_id] == 1:
                download_finish_time = self.node[idx].image_download_time[task.image_id]
            else:
                download_finish_time = max(0, self.time)

            comp_time = None
            if (task.type == "CPU" and self.node[idx].type == "CPU") or (task.type == "GPU" and self.node[idx].type == "CPU"):
                self.node[idx].mem -= task.mem
                task.cpu_freq_using = self.node[idx].cpu_freq / t_on_n[idx]
                self.node[idx].cpu_freq -= task.cpu_freq_using
                comp_time = task.cpu_cycle / task.cpu_freq_using

            elif task.type == "GPU" and self.node[idx].type == "GPU":
                self.node[idx].gpu_mem -= task.gpu_mem
                task_gpu_band = self.node[idx].gpu_band / t_on_n[idx]
                task_gpu_computing_power = self.node[idx].computing_power / t_on_n[idx]
                comp_time = config.alpha * task.task_size / task_gpu_band + task.gpu_mem / (
                            task_gpu_computing_power * math.pow((1 - config.lamda), (t_on_n[idx] - 1)))
            else:
                raise Exception("Task is assigned to an illegal node")


            trans_time = task.task_size / self.uplink_trans_rate(task, self.node[idx], t_on_n[idx])
            task_finish_time = download_finish_time + comp_time + trans_time
            self.node[idx].task_queue.put((task_finish_time, random.random(), task))
            delay = task.ddl - task_finish_time
            encourage = 1
            if delay >= 0:
                self.num_on_time += 1

            reward = delay * encourage
            rewards.append(reward)  # Add the reward to the reward list
            self.total_time += task_finish_time - task.start_time  # Calculate the total time required to complete the task
            self.download_time += download_finish_time - task.start_time
            self.trans_time += trans_time
            self.comp_time += comp_time
            task.x = self.node[idx].x
            task.y = self.node[idx].y

        return rewards

    def cal_dist(self, task, node):
        x = np.array([task.x, task.y])
        y = np.array([node.x, node.y])
        return np.sqrt(sum(np.power((x - y), 2)))

    def uplink_trans_rate(self, task, node, t_on_n):
        trans_power = 23
        noise_power = -174
        dist = self.cal_dist(task, node) / 1000 + 1e-5
        channel_gain = dist ** (-2) * 10 ** (-2)
        gamma = (trans_power * channel_gain) / noise_power ** 2
        eta = math.log(1 + gamma, 2)
        return (node.bandwidth / t_on_n) * eta


    def env_up(self):
        self.time += 1
        for idx, n in enumerate(self.node):
            while not n.task_queue.empty():
                curr_task = n.task_queue.get()
                if self.time >= curr_task[0]:
                    if n.type == "CPU":
                        n.mem += curr_task[2].mem
                        n.cpu_freq += curr_task[2].cpu_freq_using
                    if n.type == "GPU":
                        n.gpu_mem += curr_task[2].gpu_mem
                    self.task_finish_list.append(str(self.time - curr_task[2].start_time))
                else:
                    n.task_queue.put(curr_task)
                    break

            for i in range(len(n.image_download_time)):
                if n.image_list[i] == 1 and self.time >= n.image_download_time[i]:
                    n.image_list[i] = 2
        if not self.task:
            self.done = 1
            return self.done, None, None
        return False, False, None

    def get_obs(self, task, divide=False):
        obs = {}
        node_mem_list_cpu = [n.mem / 10 for n in self.node if n.type == "CPU"]
        cpu_freq_list = [n.cpu_freq / 3 for n in self.node if n.type == "CPU"]
        node_mem_list_gpu = [n.gpu_mem / 10 for n in self.node if n.type == "GPU"]
        gpu_computing_power_list = [n.computing_power / 3 for n in self.node if n.type == "GPU"]
        gpu_bandwidth_list = [n.gpu_band for n in self.node if n.type == "GPU"]
        bandwidth_list = [n.bandwidth for n in self.node]

        obs["node"] = np.hstack((node_mem_list_cpu, cpu_freq_list, node_mem_list_gpu, gpu_computing_power_list, gpu_bandwidth_list, bandwidth_list))

        node_image_list = [n.image_list[task.image_id] for n in self.node]
        download_time_list = []
        for n in self.node:
            if n.image_list[task.image_id] == 2:
                download_time_list.append(0)
            elif n.image_list[task.image_id] == 1:
                download_time_list.append(n.image_download_time[task.image_id] - self.time)
            else:
                download_time_list.append(max(self.time, n.download_finish_time) + self.image[
                    task.image_id].image_size / n.bandwidth - self.time)
        obs["image"] = np.hstack((node_image_list, download_time_list, self.image[task.image_id].image_size))

        task_type_flag = np.array([1, 0]) if task.type == "CPU" else np.array([0, 1])
        obs["task"] = np.hstack((task_type_flag, task.mem, task.cpu_cycle / 3, task.flops / 3, task.gpu_mem, task.image_id, task.ddl - self.time))

        if divide:
            return obs["node"], obs["image"], obs["task"]

        return list(chain(*obs.values()))



    def step(self, tasks, actions, t_on_n):

        rewards = self._add_task(tasks, actions, t_on_n)
        observations = []
        for task in tasks:
            observation = self.get_obs(task)
            observations.append(observation)
        if not self.task:
            done = 1
        else:
            done = 0
        return observations, rewards, done, None


if __name__ == '__main__':
    env = Env()
    env.reset()
    print(len(env.get_obs(env.task[0])))
