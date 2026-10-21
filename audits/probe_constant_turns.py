"""Compare constant-turn controls and count actions on learned-policy rollouts."""
import argparse
from pathlib import Path
import subprocess,json,sys
import numpy as np
import torch


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if not (args.repo/'hahn2019').is_dir():parser.error('invalid Hahn2019 repository')
    sys.path.insert(0,str(args.repo))
    from hahn2019.environment import Swarm,EnvConfig,displacement
    from hahn2019.learning import Learner,LearnConfig,actions_to_turns
    torch.set_num_threads(1)
    rows=[]
    for condition in ['constant_-90','constant_-45','constant_0','constant_45','constant_90','dqn_s42','dqn_s43']:
        learner=None
        if condition.startswith('dqn'):
            checkpoint=torch.load(args.repo/'output/paper_budget_20261020_235544'/condition/'policy.pt',map_location='cpu',weights_only=True)
            config=LearnConfig(**checkpoint['metadata']['learning'])
            learner=Learner(21,config)
            learner.actor.load_state_dict(checkpoint['actor'])
        for seed in [100,101,102]:
            env=Swarm(EnvConfig(n_prey=40),np.random.SeedSequence([seed,991]))
            captures=0;distances=[];counts=np.zeros(5,dtype=np.int64)
            for frame in range(5000):
                if learner is None:
                    turns=np.full(40,np.deg2rad(int(condition.split('_')[1])))
                else:
                    action=learner.actions(env.observe())
                    counts += np.bincount(action,minlength=5)
                    turns=actions_to_turns(action,config)
                _,_,_,_,info=env.step(turns,focal_episode=False)
                if frame>=100:captures+=int(info['caught'].sum())
                if frame>=100 and frame%100==0:
                    d=np.linalg.norm(displacement(env.positions[:,None,:],env.positions[None,:,:],40.),axis=-1)
                    distances.append(float(d[np.triu_indices(40,1)].mean()/40))
            rows.append({'condition':condition,'seed':seed,'captures':captures,'frames':4900,
                         'mean_pair_distance_over_side':float(np.mean(distances)),
                         'all_frame_action_counts':counts.tolist() if learner is not None else None})
        print(condition,'capture rate',sum(r['captures'] for r in rows if r['condition']==condition)/14700,flush=True)
    result={'source_revision':subprocess.check_output(['git','-C',str(args.repo),'rev-parse','HEAD'],text=True).strip(),
            'frames_per_seed':5000,'burnin':100,'n_prey':40,'seeds':[100,101,102],
            'action_counts_include_burnin':True,'rows':rows}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
