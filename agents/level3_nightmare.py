"""Distilled policy/value network with an exact finite-horizon minimax guard."""
import hashlib
import json
from functools import lru_cache
from time import perf_counter
import numpy as np
from models.agents.base_agent import BaseAgent
from models.agents.level2_learned import DIRECTIONS, forward
from backend.core.levels import RULES_VERSION
from backend.core.paths import CHECKPOINT_ROOT, LEVELS_PATH

MODEL_DIR=CHECKPOINT_ROOT/'warden_nightmare'
FEATURE_VERSION=1

def features(level,thief,warden,collected,turn):
    x=np.zeros(256,dtype=np.float32)
    grid=np.asarray(level.grid).ravel()
    x[:49]=grid==1;x[49:98]=grid==2
    x[98+thief[0]*7+thief[1]]=1;x[147+warden[0]*7+warden[1]]=1
    for i,(r,c) in enumerate(level.diamonds):
        if not collected[i]:x[196+r*7+c]=1
    x[245:253]=np.asarray((*thief,*warden,thief[0]-warden[0],thief[1]-warden[1],*level.exit),dtype=np.float32)/6
    x[253]=turn/level.max_turns;x[254:]=collected
    return x

def mask_value(collected):return int(collected[0])+2*int(collected[1])

@lru_cache(maxsize=1)
def load_model():
    metadata=json.loads((MODEL_DIR/'metadata.json').read_text(encoding='utf-8'))
    if metadata['feature_version']!=FEATURE_VERSION or metadata['rules_version']!=RULES_VERSION:
        raise RuntimeError('Nightmare rules/feature version mismatch')
    if hashlib.sha256(LEVELS_PATH.read_bytes()).hexdigest()!=metadata['maps_sha256']:
        raise RuntimeError('Nightmare map mismatch; rebuild and retrain')
    for filename,key in (('weights.npz','weights_sha256'),('tables.npz','tables_sha256')):
        if hashlib.sha256((MODEL_DIR/filename).read_bytes()).hexdigest()!=metadata[key]:
            raise RuntimeError(f'Nightmare checkpoint checksum mismatch: {filename}')
    shapes={'w1':(256,256),'b1':(256,),'w2':(256,128),'b2':(128,),'w3':(128,5),'b3':(5,)}
    with np.load(MODEL_DIR/'weights.npz',allow_pickle=False) as archive:
        weights={k:archive[k].copy() for k in shapes}
    for k,shape in shapes.items():
        if weights[k].shape!=shape or not np.isfinite(weights[k]).all():raise RuntimeError(f'Invalid Nightmare weight {k}')
        weights[k].flags.writeable=False
    with np.load(MODEL_DIR/'tables.npz',allow_pickle=False) as archive:
        tables={k:archive[k].copy() for k in archive.files}
    for entry in metadata['maps']:
        prefix=f"map{entry['id']}";positions=tables[prefix+'_positions'];values=tables[prefix+'_values']
        n=len(positions)
        if positions.shape!=(n,2) or values.shape!=(61,4,n,n) or values.dtype!=np.int16 or (np.abs(values)>1000).any():
            raise RuntimeError('Invalid minimax table')
        tables[prefix+'_index']={tuple(map(int,p)):i for i,p in enumerate(positions)}
        positions.flags.writeable=False;values.flags.writeable=False
    return weights,tables,metadata

def validate_deployment():
    _,_,metadata=load_model()
    report=json.loads((MODEL_DIR/'evaluation.json').read_text(encoding='utf-8'))
    if report['weights_sha256']!=metadata['weights_sha256'] or report['tables_sha256']!=metadata['tables_sha256']:
        raise RuntimeError('Evaluate this Nightmare checkpoint before deploying')
    scores=report['results']
    if scores['nightmare']['overall']['warden_win_rate']<scores['easy_plus']['overall']['warden_win_rate']:
        raise RuntimeError('Nightmare did not beat the easy-plus baseline')
    if report['bellman_violations']!=0:raise RuntimeError('Minimax certificate verification failed')

def table_value(level_id,turn,collected,thief,warden):
    _,tables,_=load_model();prefix=f'map{level_id}';index=tables[prefix+'_index']
    return int(tables[prefix+'_values'][turn,mask_value(collected),index[tuple(thief)],index[tuple(warden)]])

class NightmareAgent(BaseAgent):
    def __init__(self,seed=None):
        self.weights,self.tables,self.metadata=load_model()

    def select_action(self,env,time_limit_ms=None):
        started=perf_counter()
        if env.player!=-1:raise ValueError('NightmareAgent controls the Warden only')
        actions=env.legal_actions()
        if not actions:raise ValueError('No legal actions')
        prefix=f'map{env.level.id}';index=self.tables[prefix+'_index'];values=self.tables[prefix+'_values']
        mask=mask_value(env.collected);ti=index[env.thief_pos]
        q=np.array([values[env.turn+1,mask,ti,index[a]] for a in actions])
        # Negative values mean the thief loses. Only exact optimal moves are allowed.
        best=np.flatnonzero(q==q.min())
        output=forward(features(env.level,env.thief_pos,env.warden_pos,env.collected,env.turn),self.weights)
        direction_indexes=[DIRECTIONS.index((a[0]-env.warden_pos[0],a[1]-env.warden_pos[1])) for a in actions]
        chosen_index=int(best[np.argmax(output[np.array(direction_indexes)[best]])])
        action=actions[chosen_index];score=int(q[chosen_index])
        return action,{'chosen':list(action),'nodes':len(actions),'latency_ms':round((perf_counter()-started)*1000,4),
                       'action_scores':{f'{r},{c}':float(i==chosen_index) for i,(r,c) in enumerate(actions)},
                       'algorithm':'Nightmare-Minimax-MLP-v1','score_kind':'probability',
                       'model_id':self.metadata['model_id'],'learned_weight':1.0,
                       'minimax_values':{f'{r},{c}':int(q[i])/1000 for i,(r,c) in enumerate(actions)},
                       'search_horizon':env.level.max_turns-env.turn,
                       'optimal_outcome':'warden_win' if score<0 else 'thief_win' if score>0 else 'draw',
                       'neural_value':float(np.clip(output[4],-1,1)),
                       'policy_source':'neural_policy_with_exact_minimax_guard'}
