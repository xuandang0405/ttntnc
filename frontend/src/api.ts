export class ApiError extends Error {
  constructor(public status:number,public code:string,message:string){super(message);}
}
const BASE = import.meta.env.VITE_API_BASE ?? '';
export async function api<T>(path:string,body?:unknown):Promise<T>{
  let response:Response;
  try {
    response = await fetch(`${BASE}/api${path}`,{method:body === undefined?'GET':'POST',credentials:'include',headers:body === undefined?{}:{'Content-Type':'application/json'},body:body === undefined?undefined:JSON.stringify(body)});
  } catch {throw new ApiError(0,'offline','Không kết nối được máy chủ. Kiểm tra Docker rồi thử lại.');}
  const data = await response.json();
  if(!response.ok)throw new ApiError(response.status,data.error?.code ?? 'request_failed',data.error?.message ?? 'Không thể hoàn thành thao tác. Vui lòng thử lại.');
  return data;
}
