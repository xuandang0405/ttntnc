export type Cell = [number,number];
export type Difficulty = 1|2|3;
export const difficultyName=(value:Difficulty)=>value===3?'Ác mộng':value===2?'Dễ+':'Dễ';
export interface Level {id:number;name:string;subtitle:string;grid:number[][];diamonds:Cell[];thief_start:Cell;warden_start:Cell;exit:Cell;max_turns:number;image:string;floor_patches:Cell[];unlocked:boolean;completed:boolean}
export interface State {rules_version:number;level_id:number;thief_pos:Cell;warden_pos:Cell;collected:boolean[];alert:boolean;turn:number;player:number;max_turns:number;status:string;reason:string|null;result:number|null;legal_actions:Cell[]}
export interface Telemetry {chosen:Cell;nodes:number;latency_ms:number;action_scores:Record<string,number>;algorithm:string;score_kind:string;model_id:string|null;learned_weight:number|null;minimax_values:Record<string,number>|null;search_horizon:number|null;optimal_outcome:string|null;neural_value:number|null;policy_source:string|null}
export interface Game {game_id:string;seed:number;ai_level:Difficulty;state:State;telemetry:Telemetry|null;path:Cell[]}
export interface History {id:string;level_id:number;ai_level:Difficulty;status:string;turn:number;seed:number;collected:number;created_at:string;rules_version:number}
export interface Metrics {games:number;wins:number;draws:number;completed_levels:number[];unlocked_levels:number[];next_level:number;campaign_complete:boolean;rules_version:number}
export const cellLabel = ([r,c]:Cell) => `${'ABCDEFG'[c]}${r+1}`;
