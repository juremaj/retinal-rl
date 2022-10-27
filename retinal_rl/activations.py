import matplotlib.pyplot as plt
import numpy as np

import os
import torch
import torchvision.datasets as datasets

from PIL import Image as im

from sample_factory.utils.utils import log, AttrDict
from sample_factory.envs.create_env import create_env
from sample_factory.algorithms.utils.multi_agent_wrapper import MultiAgentWrapper
from sample_factory.algorithms.utils.arguments import load_from_checkpoint
from sample_factory.algorithms.appo.model import create_actor_critic
from sample_factory.algorithms.appo.learner import LearnerWorker
from sample_factory.algorithms.appo.model_utils import get_hidden_size
from sample_factory.algorithms.appo.actor_worker import transform_dict_observations

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# getting environment and actor critic from checkpoint

def make_env_func(cfg, env_config):
    return create_env(cfg.env, cfg=cfg, env_config=env_config)

def get_env_ac(cfg):
    cfg = load_from_checkpoint(cfg)
    render_action_repeat = cfg.render_action_repeat if cfg.render_action_repeat is not None else cfg.env_frameskip
    
    if render_action_repeat is None:
        log.warning('Not using action repeat!')
        render_action_repeat = 1
    log.debug('Using action repeat %d during evaluation', render_action_repeat)

    cfg.env_frameskip = 1  # for evaluation
    cfg.num_envs = 1

    env = make_env_func(cfg, AttrDict({'worker_index': 0, 'vector_index': 0}))
    env = MultiAgentWrapper(env)
    
    if hasattr(env.unwrapped, 'reset_on_init'):
        env.unwrapped.reset_on_init = False
    
    actor_critic = create_actor_critic(cfg, env.observation_space, env.action_space)
    
    device = torch.device('cpu' if cfg.device == 'cpu' else 'cuda')
    actor_critic.model_to_device(device)
        
    policy_id = cfg.policy_index
    checkpoints = LearnerWorker.get_checkpoints(LearnerWorker.checkpoint_dir(cfg, policy_id))
    checkpoint_dict = LearnerWorker.load_checkpoint(checkpoints, device)
    actor_critic.load_state_dict(checkpoint_dict['model'])
    
    return env, actor_critic


# Running experience with the extracted agent and exporting interaction

def obs_to_img(obs):
    img = np.array(obs[0]).astype(np.uint8)
    img = np.transpose(img, (1,2,0))
    return img

def simulate(cfg, env, actor_critic):
    device = torch.device('cpu' if cfg.device == 'cpu' else 'cuda')
    
    obs = env.reset() # this is the first observation when agent is spawned
    obs_torch = AttrDict(transform_dict_observations(obs))
    for key, x in obs_torch.items():
        obs_torch[key] = torch.from_numpy(x).to(device).float()
        
    # initialising rnn state and observation
    obs = env.reset() 
    rnn_states = torch.zeros([env.num_agents, get_hidden_size(cfg)], dtype=torch.float32, device=device) # this is how to initialize

    # initialise matrices for saving outputs
    t_max = int(cfg.analyze_max_num_frames)
    all_img = np.zeros((cfg.res_h, cfg.res_w, 3, t_max)).astype(np.uint8)
    all_fc_act = np.zeros((cfg.hidden_size, t_max))
    all_rnn_act = np.zeros((cfg.hidden_size, t_max))
    all_v_act = np.zeros(t_max)
    all_actions = np.zeros((2, t_max))
    all_health = np.zeros(t_max)

    num_frames = 0 # counter

    with torch.no_grad():
        while t_max>num_frames:

            policy_outputs = actor_critic(obs_torch, rnn_states, with_action_distribution=True)
            actions = policy_outputs.actions
            rnn_states = policy_outputs.rnn_states
            actions = actions.cpu().numpy() # to feed to vizdoom (CPU based)

            # here only valid for no action repeat
            obs = env.step(actions)[0]        
            obs_torch = AttrDict(transform_dict_observations(obs))
            for key, x in obs_torch.items():
                obs_torch[key] = torch.from_numpy(x).to(device).float()

            # extracting all activations to save
            img = obs_to_img(obs).astype(np.uint8) # to save later
            fc_act = actor_critic.encoder.base_encoder.forward(obs_torch['obs']).cpu().detach().numpy()
            rnn_act = rnn_states.cpu().detach().numpy()
            v_act = actor_critic.critic_linear(rnn_states).cpu().detach().numpy()
            actions = np.array(actions)
            health = env.unwrapped.get_info()['HEALTH'] # environment info (health etc.)

            # saving
            all_img[:,:,:,num_frames] = img
            all_fc_act[:,num_frames] = fc_act
            all_rnn_act[:,num_frames] = rnn_act
            all_v_act[num_frames] = v_act
            all_actions[:,num_frames] = actions
            all_health[num_frames] = health

            num_frames+=1
            if num_frames % int(t_max/10) == 0:
                print(f'Simulated {num_frames} / {int(t_max)} environment steps.')

    analyze_out = {'all_img': all_img,
                   'all_fc_act':all_fc_act,
                   'all_rnn_act':all_rnn_act,
                   'all_v_act':all_v_act,
                   'all_actions':all_actions,
                   'all_health':all_health}

    np.save(f'{os.getcwd()}/train_dir/{cfg.experiment}/analyze_out.npy', analyze_out, allow_pickle=True)

def load_sim_out(cfg):
    sim_out = np.load(f'{os.getcwd()}/train_dir/{cfg.experiment}/analyze_out.npy', allow_pickle=True).tolist()
    return sim_out

def get_acts_dataset(cfg, actor_critic):

    bck_np = np.load(os.getcwd() + '/data/doom_pad.npy') # saved 'doom-looking' background
    
    if cfg.analyze_ds_name == 'CIFAR':
        trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=None)
        testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=None)
        rewards_dict = {0:2, 1:1, 2:4, 3:5, 4:7, 5:6, 6:8, 7:9, 8:3, 9:0} # defined by the class-reward assignments in the .wad file

    elif cfg.analyze_ds_name == 'MNIST':
        trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
        testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)
        rewards_dict = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9} # defined by the class-reward assignments in the .wad file (matching digits in case of mnist)

    n_stim = len(trainset)
    offset = (28,50)

    all_fc_act = np.zeros((cfg.hidden_size, n_stim))
    all_img = np.zeros((cfg.res_h, cfg.res_w, 3, n_stim))
    all_lab = np.zeros(n_stim)

    with torch.no_grad():
        for i in range(n_stim):
                obs = pad_dataset(cfg, i, trainset, bck_np, offset)
                fc_act_torch = actor_critic.encoder.base_encoder.forward(obs) # activation of output fc layer
                
                all_fc_act[:,i] = fc_act_torch.cpu().detach().numpy()
                all_img[:,:,:,i] = obs_to_img(obs.cpu()).astype(np.uint8)
                all_lab[i] = rewards_dict[trainset[i][1]] # getting label for sample and converting based on health assignment    

                # progress
                if i % int(n_stim/20) == 0:
                    print(f'Forward pass through {i}/{n_stim} dataset entries')

    analyze_ds_out = {'all_img': all_img,
                    'all_fc_act':all_fc_act,
                   'all_lab':all_lab}
    
    return analyze_ds_out

## linear decoder analysis
def get_class_accuracy(cfg, ds_out, mode='multi', thr=5, permute=False):
    # mode can be 'multi' or 'bin', thr determines threshold for binarisation, permute will randomly shuffle labels (to get chance preformance)
    
    X = ds_out['all_fc_act'].T
    
    if mode == 'multi':
        y = ds_out['all_lab']
    elif mode == 'bin':
        y = ds_out['all_lab']<thr
        
    perm_str = '' if not permute else 'permuted '
    
    if permute:
        y = np.random.permutation(y)
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    logreg = LogisticRegression(max_iter=10000)
    logreg.fit(X_train, y_train)
    
    score_train = logreg.score(X_train, y_train)
    score_test = logreg.score(X_test, y_test)
    
    out_str = f'{perm_str}{mode} classification scores:\n  -Train: {np.round(score_train,4)}\n  -Test: {np.round(score_test,4)}\n\n'
    print(out_str) # ADD SAVING THIS STRING
    
    with open(f'train_dir/{cfg.experiment}/analyze_class_score.txt', 'a') as f:
        f.writelines(out_str)

def obs_to_img(obs):
    img = np.array(obs[0]).astype(np.uint8)
    img = np.transpose(img, (1,2,0))
    return img


def pad_dataset(cfg, i, trainset, bck_np, offset): # i:index in dataset, bck_np: numpy array of background (grass/sky), offset:determines position within background
    device = torch.device('cpu' if cfg.device == 'cpu' else 'cuda')
    in_im = trainset[i][0]
    in_np = np.array(np.transpose(in_im, (2,0,1)))
    out_np = bck_np # background
    out_np[:,offset[0]:offset[0]+32, offset[1]:offset[1]+32] = in_np # replacing pixels in the middle
    out_np_t = np.transpose(out_np, (1, 2, 0)) # reformatting for PIL conversion
    out_im = im.fromarray(out_np_t)
    out_torch = torch.from_numpy(out_np[None,:,:,:]).float().to(device) 
    
    return out_torch