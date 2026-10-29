import {memo,useEffect,useRef} from 'react'
import type {Bundle} from './assets'
import type {GameState,Playback,Preferences} from './types'
import {names} from './types'
import {startBoardRenderer} from './renderer'
import type {RenderClock} from './renderer'
import type {AnimationPhase} from './motion'

export function Board({game,bundle,selected,hint,flipped,preferences,animated,playback,disabled,onSquare,onDrop,onComplete,onPhase,heatmap}:{game:GameState;bundle:Bundle;selected:string|null;hint:string|null;flipped:boolean;preferences:Preferences;animated:boolean;playback:Playback|null;disabled:boolean;onSquare:(square:string)=>void;onDrop:(from:string,to:string)=>void;onComplete:(id:number)=>void;onPhase:(phase:AnimationPhase)=>void;heatmap:boolean}){
 const canvas=useRef<HTMLCanvasElement>(null)
 const completeRef=useRef(onComplete);completeRef.current=onComplete
 const phaseRef=useRef(onPhase);phaseRef.current=onPhase
 const clock=useRef<RenderClock>({id:null,elapsed:0})
 const destinations=new Set(preferences.hints&&selected?game.legal_actions.filter(m=>m.startsWith(selected)).map(m=>m.slice(2,4)):[])
 const squares=Array.from({length:64},(_,index)=>{const x=index%8,y=Math.floor(index/8);return flipped?String.fromCharCode(104-x)+(y+1):String.fromCharCode(97+x)+(8-y)})
 const counts=new Map<string,number>();for(const move of game.legal_actions){const s=move.slice(2,4);counts.set(s,(counts.get(s)||0)+1)}
 const latest=game.history.at(-1)
 useEffect(()=>startBoardRenderer(canvas.current,{game,bundle,flipped,animated,selected,playback,clock:clock.current,onComplete:id=>completeRef.current(id),onPhase:phase=>phaseRef.current(phase)}),[game,bundle,flipped,animated,selected,playback])

 function keyboard(event:React.KeyboardEvent<HTMLButtonElement>,index:number){const delta:Record<string,number>={ArrowLeft:-1,ArrowRight:1,ArrowUp:-8,ArrowDown:8};if(event.key in delta){event.preventDefault();const next=Math.max(0,Math.min(63,index+delta[event.key]));document.querySelector<HTMLButtonElement>(`[data-index="${next}"]`)?.focus()}}
 return <div className={`board-frame theme-${preferences.theme}`}>
  <div className="board-corner tl"/><div className="board-corner tr"/><div className="board-corner bl"/><div className="board-corner br"/>
  <div className="chessboard" role="grid" aria-label="Bàn cờ vua 8 × 8" aria-busy={!!playback}>
   {squares.map((square,index)=>{const piece=game.pieces.find(p=>p.square===square),target=destinations.has(square),last=latest&&(latest.from===square||latest.to===square),suggested=hint&&(hint.slice(0,2)===square||hint.slice(2,4)===square);return <button key={square} data-square={square} data-index={index} role="gridcell" aria-label={`${square}${piece?`, ${names[piece.type]} ${piece.color==='white'?'trắng':'đen'}`:', trống'}${target?', nước đi hợp lệ':''}`} aria-selected={selected===square} className={`square ${(Math.floor(index/8)+index%8)%2?'dark':'light'} ${selected===square?'selected':''} ${last?'last':''} ${suggested?'hinted':''} ${game.checked_king===square?'checked':''}`} onClick={()=>onSquare(square)} onKeyDown={e=>keyboard(e,index)} draggable={!disabled&&!!piece&&(game.mode==='local'||piece.color===game.human_color)} onDragStart={e=>{e.dataTransfer.setData('text/plain',square);e.dataTransfer.effectAllowed='move'}} onDragOver={e=>{if(!disabled)e.preventDefault()}} onDrop={e=>{e.preventDefault();const from=e.dataTransfer.getData('text/plain');if(/^[a-h][1-8]$/.test(from))onDrop(from,square)}}>
    {heatmap&&counts.has(square)&&<span className="heat-cell" style={{opacity:.12+Math.min(counts.get(square)!/6,.45)}}/>}
    {target&&<span className={piece?'capture-ring':'legal-dot'}/>}
    {preferences.coordinates&&index%8===0&&<span className="rank-label">{square[1]}</span>}
    {preferences.coordinates&&index>=56&&<span className="file-label">{square[0]}</span>}
   </button>})}
   <canvas ref={canvas} className="piece-canvas" aria-hidden="true"/>
  </div>
 </div>
}
export const MemoBoard=memo(Board)
