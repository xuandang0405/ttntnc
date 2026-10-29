import {useCallback,useEffect,useRef,useState} from 'react'
import {api,ApiError} from './api'
import {loadAssets} from './assets'
import type {Bundle} from './assets'
import type {GameResponse,GameState,NewOptions,Playback,Preferences,Session} from './types'
import {playSound} from './sound'
import {moveDuration} from './motion'
import type {AnimationPhase} from './motion'

const SESSION_KEY='ky-dai.session.v1'
function savedSession():Session|null{try{return JSON.parse(localStorage.getItem(SESSION_KEY)||'null')}catch{return null}}
function saveSession(s:Session){try{localStorage.setItem(SESSION_KEY,JSON.stringify(s))}catch{/* The active game can still be played without storage. */}}

export function useGame(preferences:Preferences,animated:boolean){
 const [game,setGame]=useState<GameState|null>(null),[bundle,setBundle]=useState<Bundle|null>(null)
 const [loading,setLoading]=useState(true),[busy,setBusy]=useState<string|null>(null),[error,setError]=useState<string|null>(null)
 const [playback,setPlayback]=useState<Playback|null>(null),[hint,setHint]=useState<string|null>(null),[notice,setNotice]=useState('')
 const [phase,setPhase]=useState<AnimationPhase>('move')
 const animationDone=useRef<{id:number;resolve:()=>void}|null>(null)
 const session=useRef<Session|null>(null),gameRef=useRef<GameState|null>(null),lock=useRef(false),ids=useRef(0)
 const prefs=useRef(preferences);prefs.current=preferences
 const motion=useRef(animated);motion.current=animated
 const publish=useCallback((response:GameResponse)=>{gameRef.current=response.state;setGame(response.state);setHint(null)},[])

 const boot=useCallback(async()=>{
  setLoading(true);setError(null)
  try{
   const [assets,response]=await Promise.all([loadAssets(),(async()=>{const s=savedSession();if(s){try{const res=await api.get(s);session.current=s;return res}catch(e){if(!(e instanceof ApiError)||![401,404].includes(e.status))throw e}}const res=await api.create({mode:'ai',human_color:'white'});session.current={game_id:res.game_id,token:res.token!};saveSession(session.current);return res})()])
   setBundle(assets);publish(response)
  }catch(e){setError(e instanceof Error?e.message:'Không thể mở bàn cờ.')}finally{setLoading(false)}
 },[publish])
 useEffect(()=>{void boot()},[boot])
 useEffect(()=>()=>{animationDone.current?.resolve();animationDone.current=null},[])
 useEffect(()=>{if(!notice)return;const timer=setTimeout(()=>setNotice(''),3500);return()=>clearTimeout(timer)},[notice])

 const perform=useCallback(async(path:string,extra:Record<string,unknown>={})=>{
  const s=session.current,before=gameRef.current;if(!s||!before||lock.current)return
  lock.current=true;setBusy(path==='ai-move'?'ai':'request');setError(null);setHint(null)
  try{
   const response=await api.action(path,s,before.revision,extra);publish(response)
   if(response.move){
    const duration=moveDuration(response.move)
    if(prefs.current.sound)playSound(response.state.terminal?'end':response.state.in_check?'check':response.move.captured?'capture':'move',prefs.current.volume)
    if(motion.current){
     const id=++ids.current
     const finished=new Promise<void>(resolve=>{animationDone.current={id,resolve}})
     setBusy('animation');setPhase('move');setPlayback({id,before,move:response.move,duration})
     // The renderer finishes its last visible frame before the next turn starts.
     await finished
    }
   }
  }catch(e){setError(e instanceof Error?e.message:'Thao tác chưa thành công.')}finally{lock.current=false;setBusy(null)}
 },[publish])
 useEffect(()=>{
  if(!game||loading||busy||playback||error||game.mode!=='ai'||game.terminal||game.paused||game.turn===game.human_color)return
  const timer=setTimeout(()=>void perform('ai-move',{level:1,time_ms:100}),450)
  return()=>clearTimeout(timer)
 },[game,loading,busy,playback,error,perform])

 async function newGame(options:NewOptions){
  if(lock.current)return false
  lock.current=true;setBusy('request');setError(null)
  try{const response=await api.create(options);session.current={game_id:response.game_id,token:response.token!};saveSession(session.current);publish(response);setNotice('Bàn cờ đã sẵn sàng. Chúc bạn một ván đấu hay!');return true}
  catch(e){setError(e instanceof Error?e.message:'Không thể tạo ván.');return false}
  finally{lock.current=false;setBusy(null)}
 }
 async function reload(){
  if(!session.current){await boot();return}
  setError(null);setBusy('request');lock.current=true
  try{publish(await api.get(session.current))}catch(e){setError(e instanceof Error?e.message:'Không thể kết nối.')}finally{setBusy(null);lock.current=false}
 }
 async function showHint(){
  if(!session.current||lock.current)return
  lock.current=true;setBusy('hint')
  try{const h=await api.hint(session.current);setHint(h.uci);setNotice(`Gợi ý ngẫu nhiên: ${h.uci.slice(0,2)} → ${h.uci.slice(2,4)}. Không phải nước đi tối ưu.`)}catch(e){setError(e instanceof Error?e.message:'Không thể lấy gợi ý.')}finally{lock.current=false;setBusy(null)}
 }
 async function exportPgn(){
  if(!session.current)return
  try{const blob=await api.export(session.current),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='ky-dai-van-co.pgn';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);setNotice('Đã xuất ván cờ PGN.')}catch(e){setNotice(e instanceof Error?e.message:'Không thể xuất ván.')}
 }
 return {game,bundle,loading,busy,error,playback,phase,setPhase,hint,notice,setNotice,newGame,reload,showHint,exportPgn,
  move:(uci:string)=>perform('human-move',{uci}),undo:()=>perform('undo'),pause:()=>game&&perform('pause',{paused:!game.paused}),finish:(kind:'resign'|'draw')=>perform('finish',{kind}),
  complete:(id:number)=>{const pending=animationDone.current;if(pending?.id!==id)return;animationDone.current=null;setPlayback(null);pending.resolve()}}
}
