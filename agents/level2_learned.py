"""Small trained pursuit policy. Serving uses NumPy only; no graph search."""
import hashlib
import json
import random
from functools import lru_cache
from time import perf_counter
import numpy as np
from .base_agent import BaseAgent
from backend.core.levels import RULES_VERSION
from backend.core.paths import CHECKPOINT_ROOT, LEVELS_PATH

DIRECTIONS = ((-1,0),(0,1),(1,0),(0,-1))
FEATURE_VERSION = 1
MODEL_DIR = CHECKPOINT_ROOT/'warden_easy_plus'

def features(level, thief, warden):
    """Walls + two position planes + normalized coordinates/displacement (153)."""
    x = np.zeros(153,dtype=np.float32)
    x[:49] = (np.asarray(level.grid).ravel()==1)
    x[49+thief[0]*7+thief[1]] = 1
    x[98+warden[0]*7+warden[1]] = 1
    x[147:] = np.asarray((*thief,*warden,thief[0]-warden[0],thief[1]-warden[1]),dtype=np.float32)/6
    return x

def forward(x, weights):
    h1 = np.maximum(x@weights['w1']+weights['b1'],0)
    h2 = np.maximum(h1@weights['w2']+weights['b2'],0)
    return h2@weights['w3']+weights['b3']

def softmax(logits, mask):
    logits = np.where(mask,logits,-1e9)
    values = np.exp(logits-np.max(logits,axis=-1,keepdims=True))*mask
    return values/values.sum(axis=-1,keepdims=True)

@lru_cache(maxsize=1)
def load_model():
    metadata = json.loads((MODEL_DIR/'metadata.json').read_text(encoding='utf-8'))
    path = MODEL_DIR/'weights.npz'
    if metadata['feature_version'] != FEATURE_VERSION or metadata['rules_version']!=RULES_VERSION or hashlib.sha256(path.read_bytes()).hexdigest()!=metadata['sha256']:
        raise RuntimeError('Warden checkpoint version/checksum mismatch')
    if hashlib.sha256(LEVELS_PATH.read_bytes()).hexdigest()!=metadata['maps_sha256']:
        raise RuntimeError('Maps changed; retrain the Warden checkpoint')
    shapes = {'w1':(153,128),'b1':(128,),'w2':(128,64),'b2':(64,),'w3':(64,4),'b3':(4,)}
    with np.load(path,allow_pickle=False) as archive:
        weights = {key:archive[key].copy() for key in shapes}
    for key,shape in shapes.items():
        if weights[key].shape!=shape or not np.isfinite(weights[key]).all():
            raise RuntimeError(f'Invalid Warden checkpoint tensor: {key}')
        weights[key].flags.writeable = False
    if not 0 < metadata['learned_weight'] <= 1: raise RuntimeError('Invalid policy mixture')
    return weights,metadata

def validate_deployment():
    """Never silently deploy a freshly retrained, unevaluated checkpoint."""
    _,metadata=load_model()
    report=json.loads((MODEL_DIR/'evaluation.json').read_text(encoding='utf-8'))
    if report['checkpoint_sha256']!=metadata['sha256'] or report['selected_learned_weight']!=metadata['learned_weight']:
        raise RuntimeError('Run evaluate_warden.py for this checkpoint before deployment')
    scores=report['results']
    if scores['learned_easy_plus']['overall']['capture_rate']<=scores['random_easy']['overall']['capture_rate']:
        raise RuntimeError('Warden checkpoint did not pass the easy-baseline evaluation')

class LearnedAgent(BaseAgent):
    def __init__(self, seed=None, learned_weight=None):
        self.rng = random.Random(seed)
        self.weights,self.metadata = load_model()
        self.learned_weight = self.metadata['learned_weight'] if learned_weight is None else learned_weight

    def probabilities(self, env):
        if env.player!=-1: raise ValueError('LearnedAgent controls the Warden only')
        legal = set(env.legal_actions())
        if not legal: raise ValueError('No legal actions')
        r,c = env.warden_pos
        mask = np.array([(r+dr,c+dc) in legal for dr,dc in DIRECTIONS])
        logits = forward(features(env.level,env.thief_pos,env.warden_pos),self.weights)
        trained = softmax(logits,mask)
        probabilities = (1-self.learned_weight)*mask/mask.sum()+self.learned_weight*trained.astype(np.float64)
        probabilities /= probabilities.sum()  # Stable even for a single legal action.
        return {(r+dr,c+dc):float(probabilities[i]) for i,(dr,dc) in enumerate(DIRECTIONS) if mask[i]}

    def select_action(self, env, time_limit_ms=None):
        started = perf_counter()
        probabilities = self.probabilities(env)
        actions = list(probabilities)
        action = self.rng.choices(actions,weights=list(probabilities.values()),k=1)[0]
        return action,{'chosen':list(action),'nodes':len(actions),
                       'latency_ms':round((perf_counter()-started)*1000,4),
                       'action_scores':{f'{r},{c}':p for (r,c),p in probabilities.items()},
                       'algorithm':'LearnedPursuit-v1','score_kind':'probability',
                       'model_id':self.metadata['model_id'],'learned_weight':self.learned_weight}
