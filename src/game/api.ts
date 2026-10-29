import type {GameResponse,NewOptions,Session} from './types'

const BASE=(import.meta as ImportMeta&{env:Record<string,string>}).env.VITE_API_BASE_URL||''
export class ApiError extends Error{constructor(public status:number,public code:string,message:string){super(message)}}
async function request<T>(path:string,session?:Session,body?:unknown):Promise<T>{
 const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),15000)
 try{
  const response=await fetch(`${BASE}${path}`,{method:body===undefined?'GET':'POST',headers:{...(body!==undefined?{'Content-Type':'application/json'}:{}),...(session?{Authorization:`Bearer ${session.token}`}:{})},body:body===undefined?undefined:JSON.stringify(body),signal:controller.signal})
  const json=await response.json()
  if(!response.ok)throw new ApiError(response.status,json.error?.code||'REQUEST_FAILED',(json.error?.fields?.[0]?.message||json.error?.message||'Không thể thực hiện thao tác.').replace(/^Value error, /,''))
  return json as T
 }catch(error){if(error instanceof ApiError)throw error;throw new ApiError(0,'CONNECTION_ERROR','Không kết nối được máy chủ. Kiểm tra FastAPI rồi thử lại.')}finally{clearTimeout(timer)}
}
export const api={
 create:(options:NewOptions)=>request<GameResponse>('/api/reset',undefined,options),
 get:(s:Session)=>request<GameResponse>(`/api/games/${s.game_id}`,s),
 action:(path:string,s:Session,revision:number,extra:Record<string,unknown>={})=>request<GameResponse>(`/api/${path}`,s,{game_id:s.game_id,revision,...extra}),
 hint:(s:Session)=>request<{uci:string;san:string;kind:string;probability:number}>(`/api/games/${s.game_id}/hint`,s),
 async export(s:Session){const response=await fetch(`${BASE}/api/games/${s.game_id}/pgn`,{headers:{Authorization:`Bearer ${s.token}`}});if(!response.ok)throw new Error('Không thể xuất ván cờ.');return response.blob()}
}
