import type {MoveEvent} from './types'

export type AnimationMode='system'|'full'|'off'
export type AnimationPhase='move'|'attack'|'impact'|'die'|'settle'

export const IDLE_FRAME_MS=[240,280,240,280]
export const ATTACK_FRAME_MS=[140,140,140,180]
export const DIE_FRAME_MS=[170,200,230,260]
export const IMPACT_FRAME_MS=[70,90,120,160]
export const CAPTURE={attack:480,impact:760,attackEnd:1080,arrive:1160,victimEnd:1620,complete:1760} as const

export function motionEnabled(mode:AnimationMode,reduced:boolean){
 return mode==='full'||(mode==='system'&&!reduced)
}

export function migrateAnimationMode(value:unknown):AnimationMode{
 // Old checked toggles meant that the player wanted effects enabled.
 if(value===true)return 'full'
 if(value===false)return 'off'
 return value==='full'||value==='off'||value==='system'?value:'system'
}

export function frameIndex(entry:{frameDurationsMs:number[];frameIndices?:number[];loop?:boolean},elapsed:number){
 const durations=entry.frameDurationsMs,total=durations.reduce((a,b)=>a+b,0)
 if(total<=0)return 0
 let time=Math.max(0,elapsed)
 if(entry.loop)time%=total
 else if(time>=total)return -1
 for(let i=0;i<durations.length;i++){
  if(time<durations[i])return entry.frameIndices?.[i]??i
  time-=durations[i]
 }
 return -1
}

export function ease(t:number){
 t=Math.max(0,Math.min(1,t))
 return t<.5?4*t*t*t:1-(-2*t+2)**3/2
}

export function moveDuration(move:Pick<MoveEvent,'from'|'to'|'captured'|'rook'>){
 if(move.captured)return CAPTURE.complete
 if(move.rook)return 900
 const distance=Math.hypot(move.to.charCodeAt(0)-move.from.charCodeAt(0),Number(move.to[1])-Number(move.from[1]))
 return Math.round(Math.max(650,Math.min(1100,540+distance*80)))
}

export function captureProgress(elapsed:number){
 if(elapsed<CAPTURE.attack)return .74*ease(elapsed/CAPTURE.attack)
 if(elapsed<CAPTURE.attackEnd)return .74+.1*ease((elapsed-CAPTURE.attack)/(CAPTURE.attackEnd-CAPTURE.attack))
 return .84+.16*ease((elapsed-CAPTURE.attackEnd)/(CAPTURE.arrive-CAPTURE.attackEnd))
}

export function phaseAt(capture:boolean,elapsed:number):AnimationPhase{
 if(!capture||elapsed<CAPTURE.attack)return 'move'
 if(elapsed<CAPTURE.impact)return 'attack'
 if(elapsed<CAPTURE.impact+180)return 'impact'
 if(elapsed<CAPTURE.victimEnd)return 'die'
 return 'settle'
}

export const phaseLabels:Record<AnimationPhase,string>={move:'Quân cờ đang di chuyển…',attack:'Đang ra đòn…',impact:'Đánh trúng quân địch!',die:'Quân bị ăn đang tan biến…',settle:'Hoàn tất nước đi…'}
