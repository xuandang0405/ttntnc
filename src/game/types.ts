export type Color='white'|'black'
export type Kind='pawn'|'knight'|'bishop'|'rook'|'queen'|'king'
export type Piece={square:string;type:Kind;color:Color;symbol:string}
export type MoveEvent={uci:string;san:string;from:string;to:string;piece:Omit<Piece,'square'>;captured:(Omit<Piece,'square'>&{square:string})|null;rook:{from:string;to:string}|null;promotion:Kind|null;is_en_passant:boolean;move_number:number;ply?:number}
export type Telemetry={chosen:string;algorithm:string;nodes:number;latency_ms:number;eval_score:number;top_moves:Record<string,number>;legal_count:number;selected_probability:number}
export type GameState={game_id:string;revision:number;fen:string;initial_fen:string;board:(string|null)[];pieces:Piece[];legal_actions:string[];turn:Color;in_check:boolean;checked_king:string|null;material_cp:number;ply:number;max_ply:number;terminal:boolean;result:number;reason:string|null;mode:'ai'|'local';human_color:Color;seed:number;paused:boolean;history:MoveEvent[];captured:Record<Color,Kind[]>;created_at:string;elapsed_ms:number;telemetry:Telemetry|null;can_undo:boolean}
export type Session={game_id:string;token:string}
export type GameResponse={game_id:string;state:GameState;telemetry:Telemetry|null;move:MoveEvent|null;token?:string}
export type NewOptions={mode:'ai'|'local';human_color:Color;seed?:number;fen?:string}
export type Preferences={sound:boolean;animation:import('./motion').AnimationMode;hints:boolean;coordinates:boolean;theme:'stone'|'walnut'|'night';volume:number}
export type Playback={id:number;before:GameState;move:MoveEvent;duration:number}
export const names:Record<Kind,string>={pawn:'Tốt',knight:'Mã',bishop:'Tượng',rook:'Xe',queen:'Hậu',king:'Vua'}
export const values:Record<Kind,number>={pawn:1,knight:3.2,bishop:3.3,rook:5,queen:9,king:0}
export const sideName=(side:Color)=>side==='white'?'Trắng':'Đen'
export const opposite=(side:Color):Color=>side==='white'?'black':'white'
export function squareCoords(square:string,flipped:boolean){const file=square.charCodeAt(0)-97,rank=Number(square[1])-1;return flipped?{x:7-file,y:rank}:{x:file,y:7-rank}}
export const defaultPreferences:Preferences={sound:true,animation:'system',hints:true,coordinates:true,theme:'stone',volume:45}
