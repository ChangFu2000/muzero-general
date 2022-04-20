import random

import gym
from gym import spaces
import json
import numpy as np

from defender.defend_minimal import Defender


class MetaNet(gym.Env):
    def __init__(self, config):
        self.env = None

        self.envs = list()
        for path in config['path']:
            conf = config.copy()
            conf['path'] = path
            env = BaseMetaNet(conf)
            self.envs.append(env)
        
        self.observation_space = spaces.Box(low=-1, high=1e5, shape=(config['observation_space'], ), dtype=np.float32)
        self.action_space = spaces.Discrete(config['action_space'])

    def reset(self):
        self.env = random.choice(self.envs)
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)


class BaseMetaNet(gym.Env):
    def __init__(self, config, seed=None):
        self.name = None
        # machines composing the whole network
        self.machines = None
        # topology is represeted as an adjacent matrix using numpy
        self.topology = None
        # initially reachable subnetworks
        self.init_reachable = None
        # target (critical) machines
        self.targets = None
        # firewalls control the connections between subnetworks
        self.firewalls = None
        # number of machines
        self.num_machines = None
        # name of exploitable services
        self.num_services = None

        # other information about the network
        self.info = {'step_count': 0}

        self.config = config
        self.parse()

        if seed is not None:
            self.seed(seed)
        
        self.observation_space = spaces.Box(low=-1, high=1e5, shape=(self.config['observation_space'], ), dtype=np.float32)
        self.action_space = spaces.Discrete(self.config['action_space'])

        self.defender = Defender(config)

    # dimension of the state of the network, which is the input dimension of agent
    @property
    def state_dim(self):
        # |M| * (|state| + |E| + 1) + 2
        return self.num_machines * (State.dim + self.num_services * 2 + 1) + 2

    # number of possible actions
    @property
    def action_dim(self):
        # |M| + |M| * |E|
        return self.num_machines * (self.num_services + 1)

    def parse(self):
        path = self.config['path']

        self.num_machines = self.config['num_machines']
        self.num_services = self.config['num_services']
        with open(path, 'r') as f:
            config = json.load(f)
            # print(config)

            # add machines
            self.machines = list()
            for machine in config['machines']:
                m = Machine(machine['id'], machine['ip'])
                # add services
                s = Service(machine['defense'], machine['detect'], self.config['W'])
                m.service = s
                self.machines.append(m)
            self.machines = tuple(self.machines)

            # add topology info
            self.topology = np.array(config['topology'])
            self.init_reachable = tuple(config['init_reachable'])
            self.targets = tuple(config['targets'])

            # add firewalls
            self.firewalls = Firewall()
            self.firewalls.rules = list()
            for rule in config['firewalls']:
                r = Rule(rule['src'], rule['dst'], rule['name'])
                self.firewalls.rules.append(r)
            self.firewalls.rules = tuple(self.firewalls.rules)

    # print out the information of the network
    def print_info(self):
        print('number of machines: %d' % self.num_machines)
        print('number of services:', self.num_services)
        print('state dimension: %d' % self.state_dim)
        print('action dimension: %d' % self.action_dim)
        
        print()
        print('network topology:')
        print(self.visualize())

    # build an Action object according to an action index
    def build_attacker_action(self, action_idx):
        action = Action()
        machine_idx, service_idx = divmod(action_idx, self.num_services + 1)

        if service_idx == 0:
            action.type = 'scan'
        else:
            action.type = 'exploit'
            action.service = service_idx - 1
        action.target = self.machines[machine_idx]
        return action

    def build_defender_action(self, action_idx):
        action = Action()
        machine_idx, service_idx = divmod(action_idx, self.num_services + 1)

        action.type = 'defend'
        action.service = service_idx
        action.target = self.machines[machine_idx]
        return action

    # update reachable machines after compromising a machine
    def update_reachable(self, machine):
        reachable = self.topology[machine.id]
        for i, j in enumerate(reachable):
            if j:
                self.machines[i].state.reachable = True

    # current state of the network (seen by the attacker)
    # [m1.scanned, m1.reachable, m1.compromised, m1.attack, m1.defense, ...]
    def attacker_state(self):
        s = list()
        for machine in self.machines:
            info = list()
            info += machine.state.to_list()
            info += machine.service.attack

            # attach services information
            # first number indicates the knowledge about the service
            # 0 for unknown, -1 for absent, 1 for present
            if machine.state.scanned:
                info += machine.service.defense
                info += [machine.service.detect]
            else:
                info += [-1] * (self.num_services + 1)
            s += info
        return np.array(s, np.float32)

    # current state of the network (seen by the defender)
    # [m1.defense, m1.detect, m2.defense, m2.attack, ...]
    def defender_state(self):
        s = list()
        for machine in self.machines:
            s += machine.service.defense
            # whether the defender can increase the detect status
            # s += [machine.service.detect]
        # add topology info
        # s = np.hstack((s, self.topology.flatten()))
        return np.array(s, np.float32)
        
    # initialize the state of the network
    def reset(self):
        # reset machine states
        for machine in self.machines:
            machine.state = State()
            machine.service.reset()

        # initialize reachable subnetworks
        for reachable in self.init_reachable:
            self.machines[reachable].state.reachable = True

        # intialize information
        self.info = {'step_count': 0}

        return np.append(self.attacker_state(), [0., 0.]).astype(np.float32)

    def get_reward(self, agent, type, action, default=0):
        return self.config['reward_scheme'].get(agent, dict()).get(type, dict()).get(action, default)

    # update network state given the attacker's action
    def attacker_step(self, action_idx):
        action = self.build_attacker_action(action_idx)
        done = False
        reward = 0

        # if the machine is reachable
        if action.target.state.reachable:      
            if action.type == 'scan':
                action.target.state.scanned = True
                reward += self.get_reward('attacker', 'cost', 'scan')
            elif action.type == 'exploit':
                if self.config['scan_before_exploit'] and not action.target.state.scanned:
                    reward += self.get_reward('attacker', 'penalty', 'compromise_unscanned_machine')
                else:
                    reward += self.get_reward('attacker', 'cost', 'exploit')
                    if_success = action.target.service.increment_attack(action.service)
                    # print(service.attack, service.defend)
                    # if the exploit succeeds
                    if if_success:
                        if action.target.state.compromised:
                            reward += self.get_reward('attacker', 'penalty', 'compromise_again')
                        else:
                            action.target.state.compromised = True
                            self.update_reachable(action.target)     # update reachable machines
                            if action.target.ip in self.targets:
                                reward += self.get_reward('attacker', 'reward', 'compromise_critical')
                                done = True
                            else:
                                reward += self.get_reward('attacker', 'reward', 'compromise_noncritical')
                    else:
                        if np.random.rand() < action.target.service.detect / self.config['W']:
                            reward += self.get_reward('attacker', 'penalty', 'caught_by_defender')
                            done = True
            else:
                raise Exception('Unknown type of operation')
        else:
            reward += self.get_reward('attacker', 'penalty', 'access_unreachable_machine')

        self.info['step_count'] += 1
        if self.info['step_count'] >= self.config['maximum_actions']:
            reward += self.get_reward('attacker', 'penalty', 'exceed_maximum_action')
            done = True

        return_state = self.attacker_state() # if np.random.rand() < 0.2 else [-1] * 48

        return return_state, reward, done, self.info

    # update network state given the defender's action
    def defender_step(self, action_idx):
        action = self.build_defender_action(action_idx)
        done = False
        reward = 0

        action.target.service.increment_defense(action.service)
        reward += self.get_reward('defender', 'cost', 'defend')

        # self.info.step_count += 1
        return self.attacker_state(), reward, done, self.info

    def step(self, action_idx):
        state_1, reward_1, done_1, info_1 = self.attacker_step(action_idx)
        action = self.defender.act(self.defender_state())
        state_2, reward_2, done_2, info_2 = self.defender_step(action)
        return np.append(state_1, [action_idx, reward_1]).astype(np.float32), reward_1, done_1, info_1

    # visualize the network
    def visualize(self):
        graph = 'graph TD\n'
        for machine in self.machines:
            target = True if machine.ip in self.targets else False
            if target:
                graph += 'm_%d((m%d))\n' % (machine.id, machine.id)
            else:
                graph += 'm_%d(m%d)\n' % (machine.id, machine.id)

        # subnetwork connections and services info
        for i in range(self.topology.shape[0]):
            for j in range(i+1, self.topology.shape[1]):
                if self.topology[i][j]:
                    graph += 'm_%d --- m_%d\n' % (i, j)

        for reachable in self.init_reachable:
            graph += 'attacker --- m_%d\n' % (reachable)

        return graph

    def close(self):
        self.close()


'''
State of a machine regard to the attacker. 
'''
class State(object):
    dim = 3

    def __init__(self):
        # whether the machine has been scanned by the attacker
        self.scanned = False
        self.reachable = False
        self.compromised = False   

    def to_list(self):
        return [self.scanned, self.reachable, self.compromised]

'''
Machines are primitive building blocks of the network, where one or more services are running. 
'''
class Machine(object):
    def __init__(self, id, ip):
        self.id = id
        self.ip = ip
        self.state = State()
        self.service = None
        self.info = dict()


'''
A service is a software running on a machine that can be potentially exploited. 
'''
class Service(object):
    def __init__(self, defense, detect, W):
        self.W = W
        self.init_defense = defense
        self.init_detect = detect
        self.reset()

    def increment_attack(self, service):
        if self.attack[service] < self.W:
            self.attack[service] += 1
        return self.attack[service] > self.defense[service]

    def increment_defense(self, service):
        if service < len(self.init_defense):
            if self.defense[service] < self.W:
                self.defense[service] += 1
        else:
            if self.detect < self.W:
                self.detect += 1

    def reset(self):
        self.attack = [0] * len(self.init_defense)
        self.defense = self.init_defense
        self.detect = self.init_detect

    def to_list(self):
        return [self.defense]


class Rule(object):
    def __init__(self, src, dst, name):
        self.src = src
        self.dst = dst
        self.name = name

'''
Firewalls allows and prevents certain communications between subnetworks. 
'''
class Firewall(object):
    def __init__(self):
        # default: if not specified, use default policy, where 0 denial and 1 is permit
        self.default = 1
        self.rules = None


'''
Actions are scanning and exploits that the attacker performs at each time step.
'''
class Action(object):
    def __init__(self):
        # scan or exploit
        self.type = None
        # the target machine to scan or exploit
        self.target = None
        # the name of the service to be exploited if type == 'exploit' otherwise None
        self.service = None

    def to_str(self):
        res = self.type + ' '
        if self.type == 'exploit':
            res += self.service + ' on '
        res += self.target.ip
        return res
