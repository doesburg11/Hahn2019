"""Bounded diagnostics only; does not train or modify production models.

Glorot/OU comparisons implement documented library defaults, not recovered
settings from the authors. Neighbor permutation is an out-of-distribution
sensitivity diagnostic, not a causal test of cohesion.
"""
import argparse
from dataclasses import asdict, replace
import json
import hashlib
import subprocess
from pathlib import Path
import sys
import numpy as np
import torch


def observation_batch(Swarm, EnvConfig, neighbors, worlds=20):
    return np.concatenate([Swarm(EnvConfig(neighbors=neighbors), seed=1000+i).observe()
                           for i in range(worlds)])


def initialization_probe(mlp, Swarm, EnvConfig):
    rows=[]
    for algorithm,neighbors,depth in [('dqn',5,10),('ddpg',1,5)]:
        data=observation_batch(Swarm,EnvConfig,neighbors)
        for seed in [0,1,2,3,4,5,6,7,42,43]:
            for mode in ['torch_default','glorot_zero_bias']:
                torch.manual_seed(seed)
                net=mlp(data.shape[1],5 if algorithm=='dqn' else 1,depth,16)
                if mode=='glorot_zero_bias':
                    for layer in net:
                        if isinstance(layer,torch.nn.Linear):
                            torch.nn.init.xavier_uniform_(layer.weight)
                            torch.nn.init.zeros_(layer.bias)
                observations=torch.tensor(data,requires_grad=True)
                outputs=net(observations)
                squared_norm=torch.zeros(len(data))
                for j in range(outputs.shape[1]):
                    gradient=torch.autograd.grad(outputs[:,j].sum(),observations,retain_graph=True)[0]
                    squared_norm += gradient.detach().square().sum(1)
                rows.append({'algorithm':algorithm,'seed':seed,'initialization':mode,
                             'mean_input_jacobian_frobenius':float(squared_norm.sqrt().mean()),
                             'mean_output_sd_across_states':float(outputs.detach().std(0,unbiased=False).mean())})
    return rows


def ou_probe():
    # Use common standard normal draws, comparing process definitions only.
    draws=np.random.default_rng(123).normal(size=(4000,1000))
    rows=[]
    for name,dt,random_reset in [('reconstruction',1.,False),('dt1_random_reset',1.,True),
                                  ('dt001_zero_reset',.01,False),('keras_rl_2019_default',.01,True)]:
        x=np.random.default_rng(456).normal(0,.3,4000) if random_reset else np.zeros(4000)
        values=[]
        for j in range(draws.shape[1]):
            x=x+.15*(-x)*dt+.3*np.sqrt(dt)*draws[:,j]
            if j in [0,9,99,999]:
                values.append({'frame':j+1,'std':float(x.std())})
        rows.append({'mode':name,'dt':dt,'reset':'normal(0,0.3)' if random_reset else 'zero',
                     'innovation_std':float(.3*np.sqrt(dt)), 'ar_coefficient':1-.15*dt,'samples':values})
    return rows


def learned_neighbor_probe(repo, Learner, LearnConfig, Swarm, EnvConfig, wrap_angle):
    checkpoints={
        'original_dqn_s42':repo/'output/paper_budget_20261020_235544/dqn_s42/policy.pt',
        'original_dqn_s43':repo/'output/paper_budget_20261020_235544/dqn_s43/policy.pt',
        'original_ddpg_s43':repo/'output/paper_budget_20261020_235544/ddpg_s43/policy.pt',
        'wrapped_ddpg_s42':repo/'output/ddpg_wrapped_20261021_094302/ddpg_s42/policy.pt',
        'wrapped_ddpg_s43':repo/'output/ddpg_wrapped_20261021_094302/ddpg_s43/policy.pt',
    }
    rows=[]
    for name,path in checkpoints.items():
        if not path.exists():
            rows.append({'condition':name,'missing':True});continue
        checkpoint=torch.load(path,map_location='cpu',weights_only=True)
        config=LearnConfig(**checkpoint['metadata']['learning'])
        neighbors=checkpoint['metadata']['environment']['neighbors']
        data=observation_batch(Swarm,EnvConfig,neighbors)
        learner=Learner(data.shape[1],config)
        learner.actor.load_state_dict(checkpoint['actor'])
        altered=data.copy()
        permutation=np.random.default_rng(789).permutation(len(data))
        altered[:,6:]=data[permutation,6:]
        before,after=learner.actions(data),learner.actions(altered)
        row={'condition':name,'states':len(data),'mode':asdict(config)}
        if config.algorithm=='dqn':
            row['greedy_action_change_fraction']=float((before!=after).mean())
            row['original_action_counts']=np.bincount(before,minlength=5).tolist()
        else:
            scale=np.pi/180 if config.action_units=='degrees' else 1.
            row['mean_physical_action_change_degrees']=float(np.rad2deg(np.abs(wrap_angle((after-before)*scale))).mean())
        rows.append(row)
    return rows


def ecology_probe(Swarm, EnvConfig, baseline_turns, displacement):
    base=EnvConfig(n_prey=10)
    variants={
        'base':{},'speed_0.1':{'speed':.1},'speed_1.0':{'speed':1.},
        'target_radius_3':{'target_radius':3.},'target_radius_20':{'target_radius':20.},
        'target_hold_1':{'target_hold':1},'target_hold_50':{'target_hold':50},
        'burst_multiplier_1':{'burst_multiplier':1.},'burst_multiplier_3':{'burst_multiplier':3.},
        'no_confusion':{'confusion':False},
    }
    rows=[]
    for variant,overrides in variants.items():
        config=replace(base,**overrides)
        for policy in ['turnaway','boids','random']:
            for seed in [100,101,102]:
                env=Swarm(config,seed)
                rng=np.random.default_rng(seed+10000)
                captures=0; samples=[]
                for frame in range(2000):
                    turns=baseline_turns(env,policy,rng)
                    _,_,_,_,info=env.step(turns,focal_episode=False)
                    if frame>=100:captures+=int(info['caught'].sum())
                    if frame>=100 and frame%100==0:
                        distances=np.linalg.norm(displacement(env.positions[:,None,:],env.positions[None,:,:],config.side),axis=-1)
                        samples.append(float(distances[np.triu_indices(config.n_prey,1)].mean()/config.side))
                rows.append({'variant':variant,'overrides':overrides,'policy':policy,'seed':seed,
                             'captures':captures,'frames':1900,'mean_pair_distance_over_side':float(np.mean(samples))})
        print('ecology complete:',variant,flush=True)
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if not (args.repo/"hahn2019").is_dir():
        parser.error("--repo must point to the Hahn2019 repository root")
    sys.path.insert(0,str(args.repo))
    from hahn2019.environment import Swarm,EnvConfig,wrap_angle,displacement
    from hahn2019.learning import mlp,Learner,LearnConfig
    from hahn2019.policies import baseline_turns
    torch.set_num_threads(1)
    result={'source_revision': subprocess.check_output(['git','-C',str(args.repo),'rev-parse','HEAD'],text=True).strip(),
            'diagnostic_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'observation_distribution':'20 seeded reset worlds, 10 prey each; no transitions; not a training-state sample',
            'initialization_seeds':[0,1,2,3,4,5,6,7,42,43],
            'ecology_seeds':[100,101,102],
            'ecology_population':10,'ecology_frames_per_seed':2000,'ecology_burnin':100,
            'initialization':initialization_probe(mlp,Swarm,EnvConfig),
            'ou':ou_probe(),
            'neighbor_permutation':learned_neighbor_probe(args.repo,Learner,LearnConfig,Swarm,EnvConfig,wrap_angle),
            'ecology':ecology_probe(Swarm,EnvConfig,baseline_turns,displacement)}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('SAVED',args.output)


if __name__=='__main__':main()
