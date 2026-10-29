import type {Action,Bundle} from './assets'
import type {Color,GameState,Kind,Piece,Playback} from './types'
import {squareCoords} from './types'
import {CAPTURE,captureProgress,ease,frameIndex,phaseAt} from './motion'
import type {AnimationPhase} from './motion'

type Point={x:number;y:number}
export type RenderClock={id:number|null;elapsed:number}
type Options={game:GameState;bundle:Bundle;flipped:boolean;animated:boolean;selected:string|null;playback:Playback|null;clock:RenderClock;onComplete:(id:number)=>void;onPhase:(phase:AnimationPhase)=>void}

export function startBoardRenderer(element:HTMLCanvasElement|null,options:Options){
 if(!element)return
 const context=element.getContext('2d');if(!context)return
 const ctx=context,size=1024,cell=size/8
 const {game,bundle,flipped,animated,selected,playback,clock,onComplete,onPhase}=options
 element.width=size;element.height=size
 ctx.imageSmoothingEnabled=true;ctx.imageSmoothingQuality='high'
 if(clock.id!==(playback?.id??null)){clock.id=playback?.id??null;clock.elapsed=0}
 let raf=0,lastTime:number|null=null,lastPaint=-Infinity,done=false,lastPhase:AnimationPhase|null=null
 const idleEnabled=animated&&!game.paused&&!game.terminal
 const tint=(color:Color)=>color==='white'?'#efc96e':'#bc83ef'
 const position=(square:string):Point=>{const c=squareCoords(square,flipped);return {x:(c.x+.5)*cell,y:(c.y+.92)*cell}}

 function shadow(p:Point,alpha=.17){
  ctx.save();ctx.globalAlpha=alpha;ctx.fillStyle='#152327';ctx.beginPath();ctx.ellipse(p.x,p.y,cell*.29,cell*.065,0,0,Math.PI*2);ctx.fill();ctx.restore()
 }

 function sprite(color:Color,kind:Kind|'effect',action:Action,time:number,p:Point,transform:{scale?:number;alpha?:number;angle?:number}={}){
  const entry=bundle.entries.get(`${color}/${kind}/${action}`);if(!entry)return
  const image=bundle.images.get(entry.file);if(!image)return
  const index=frameIndex(entry,time);if(index<0)return
  const factor=cell/320*(transform.scale??1)
  ctx.save();ctx.translate(p.x,p.y);ctx.rotate(transform.angle??0);ctx.globalAlpha=transform.alpha??1
  ctx.drawImage(image,index*entry.frameWidth,0,entry.frameWidth,entry.frameHeight,-entry.anchor.x*factor,-entry.anchor.y*factor,entry.frameWidth*factor,entry.frameHeight*factor)
  ctx.restore()
 }

 function aura(color:Color,p:Point,time:number,strength=1){
  const wave=(Math.sin(time/430)+1)/2
  ctx.save();ctx.strokeStyle=tint(color);ctx.fillStyle=tint(color);ctx.lineWidth=cell*.012
  ctx.globalAlpha=(.12+wave*.15)*strength
  ctx.beginPath();ctx.ellipse(p.x,p.y-cell*.025,cell*(.28+wave*.025),cell*.072,0,0,Math.PI*2);ctx.stroke()
  // Two slow motes per piece keep the whole mobile board below 100 particles.
  for(let i=0;i<2;i++){
   const phase=time/1100+i*Math.PI
   const x=p.x+Math.cos(phase)*cell*.24,y=p.y-cell*(.2+(Math.sin(phase*.7)+1)*.17)
   ctx.globalAlpha=(.24+wave*.23)*strength;ctx.beginPath();ctx.arc(x,y,cell*(.009+wave*.003),0,Math.PI*2);ctx.fill()
  }
  ctx.restore()
 }

 function idle(piece:Pick<Piece,'type'|'color'>,p:Point,time:number,live:boolean,emphasis=false){
  shadow(p)
  if(live)aura(piece.color,p,time,emphasis?1.7:1)
  const wave=live?Math.sin(time/520):0
  sprite(piece.color,piece.type,'idle',live?time:0,{x:p.x,y:p.y-wave*cell*.009},{scale:1+wave*.006})
 }

 function trail(color:Color,from:Point,to:Point,progress:number){
  ctx.save();ctx.strokeStyle=tint(color);ctx.lineWidth=cell*.035;ctx.lineCap='round';ctx.globalAlpha=.23*Math.sin(Math.PI*progress)
  ctx.beginPath();ctx.moveTo(from.x,from.y-cell*.035);ctx.lineTo(to.x,to.y-cell*.035);ctx.stroke();ctx.restore()
 }

 function burst(color:Color,p:Point,time:number,duration=650){
  const progress=Math.max(0,Math.min(1,time/duration));if(progress>=1)return
  // One soft flash at contact, followed by a ring and fading fragments.
  const glow=ctx.createRadialGradient(p.x,p.y-cell*.34,0,p.x,p.y-cell*.34,cell*.62)
  glow.addColorStop(0,color==='white'?'#fff0b3':'#ead0ff');glow.addColorStop(.35,tint(color));glow.addColorStop(1,'#ffffff00')
  ctx.save();ctx.globalAlpha=Math.max(0,1-time/300)*.55;ctx.fillStyle=glow
  ctx.fillRect(p.x-cell*.65,p.y-cell,cell*1.3,cell*1.3);ctx.restore()
  ctx.save();ctx.strokeStyle=tint(color);ctx.fillStyle=tint(color);ctx.lineWidth=cell*.018
  ctx.globalAlpha=(1-progress)*.8
  ctx.beginPath();ctx.ellipse(p.x,p.y-cell*.33,cell*(.12+progress*.54),cell*(.08+progress*.26),0,0,Math.PI*2);ctx.stroke()
  for(let i=0;i<14;i++){
   const angle=i*Math.PI*2/14,spread=cell*(.12+.65*ease(progress)),radius=cell*.025*(1-progress)
   const x=p.x+Math.cos(angle)*spread,y=p.y-cell*.36+Math.sin(angle)*spread*.7-progress*cell*.22
   ctx.globalAlpha=(1-progress)*.9;ctx.beginPath();ctx.arc(x,y,Math.max(.2,radius),0,Math.PI*2);ctx.fill()
  }
  ctx.restore()
 }

 function draw(now:number){
  ctx.clearRect(0,0,size,size)
  const elapsed=clock.elapsed,active=!!playback&&animated&&elapsed<playback.duration
  const board=active?playback.before:game,event=active?playback.move:null
  const pieces=[...board.pieces].sort((a,b)=>squareCoords(a.square,flipped).y-squareCoords(b.square,flipped).y)
  for(const piece of pieces){
   if(event&&(piece.square===event.from||piece.square===event.rook?.from||piece.square===event.captured?.square))continue
   const offset=piece.square.charCodeAt(0)*119+Number(piece.square[1])*173
   idle(piece,position(piece.square),now+offset,idleEnabled,piece.square===selected)
  }
  if(active&&event){
   const capture=!!event.captured,from=position(event.from),to=position(event.to)
   const progress=capture?captureProgress(elapsed):ease(elapsed/playback.duration)
   const foot={x:from.x+(to.x-from.x)*progress,y:from.y+(to.y-from.y)*progress}
   const point={...foot}
   const travelFraction=capture?Math.min(elapsed/CAPTURE.attack,1):elapsed/playback.duration
   point.y-=cell*(event.piece.type==='knight'?.45:.035)*Math.sin(Math.PI*travelFraction)
   const phase=phaseAt(capture,elapsed)
   if(lastPhase!==phase){lastPhase=phase;onPhase(phase)}

   if(event.captured){
    const victim=event.captured,target=position(victim.square)
    shadow(target,.17*Math.max(0,1-(elapsed-CAPTURE.impact)/(CAPTURE.victimEnd-CAPTURE.impact)))
    if(elapsed<CAPTURE.impact)idle(victim,target,now,idleEnabled)
    else if(elapsed<CAPTURE.victimEnd){
     const t=(elapsed-CAPTURE.impact)/(CAPTURE.victimEnd-CAPTURE.impact)
     sprite(victim.color,victim.type,'die',elapsed-CAPTURE.impact,{x:target.x,y:target.y+cell*.1*t},{alpha:1-t*.8,angle:t*.08})
    }
   }

   if(progress<1)trail(event.piece.color,from,foot,progress)
   shadow(foot,.15)
   if(capture&&elapsed>=CAPTURE.attack&&elapsed<CAPTURE.attackEnd){
    const strike=Math.sin(Math.PI*(elapsed-CAPTURE.attack)/(CAPTURE.attackEnd-CAPTURE.attack))
    const target=position(event.captured!.square),distance=Math.hypot(target.x-point.x,target.y-point.y)||1
    const lunge={x:point.x+(target.x-point.x)/distance*cell*.1*strike,y:point.y+(target.y-point.y)/distance*cell*.1*strike}
    sprite(event.piece.color,event.piece.type,'attack',elapsed-CAPTURE.attack,lunge,{scale:1+strike*.025,angle:(to.x>=from.x?1:-1)*strike*.045})
   }else if(capture&&elapsed>=CAPTURE.arrive){
    idle({color:event.piece.color,type:event.promotion||event.piece.type},to,now,idleEnabled)
   }else{
    const promotion=event.promotion,blend=promotion?Math.max(0,Math.min(1,(elapsed/playback.duration-.78)/.22)):0
    sprite(event.piece.color,event.piece.type,'move',elapsed,point,{alpha:1-blend})
    if(promotion&&blend>0){sprite(event.piece.color,promotion,'idle',now,to,{alpha:blend});burst(event.piece.color,to,(elapsed/playback.duration-.78)/.22*420,500)}
   }

   if(event.rook){
    const start=position(event.rook.from),end=position(event.rook.to),p=ease(elapsed/playback.duration)
    const rook={x:start.x+(end.x-start.x)*p,y:start.y+(end.y-start.y)*p}
    trail(event.piece.color,start,rook,p);shadow(rook);sprite(event.piece.color,'rook','move',elapsed,rook)
   }
   if(capture&&elapsed>=CAPTURE.impact){
    const target=position(event.captured!.square),time=elapsed-CAPTURE.impact
    sprite(event.piece.color,'effect','impact',time,{x:target.x,y:target.y-cell*.4},{scale:1.05})
    burst(event.piece.color,target,time)
   }
  }else if(playback&&!done){done=true;onComplete(playback.id)}
 }

 function render(now:number){
  const hidden=document.hidden
  if(lastTime!==null&&playback&&!hidden)clock.elapsed+=Math.min(50,Math.max(0,now-lastTime))
  lastTime=now
  const moving=!!playback&&!done
  if(moving||now-lastPaint>=1000/30){draw(now);lastPaint=now}
  if(!hidden&&animated&&(idleEnabled||moving))raf=requestAnimationFrame(render)
 }
 function visibility(){
  cancelAnimationFrame(raf);lastTime=null
  if(!document.hidden)render(performance.now())
 }
 document.addEventListener('visibilitychange',visibility)
 render(performance.now())
 return()=>{cancelAnimationFrame(raf);document.removeEventListener('visibilitychange',visibility)}
}
