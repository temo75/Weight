const OPENAI_MODEL="gpt-4.1-mini";
function corsHeaders(origin){return {"Access-Control-Allow-Origin":origin||"*","Access-Control-Allow-Headers":"Content-Type","Access-Control-Allow-Methods":"POST, OPTIONS","Content-Type":"application/json; charset=utf-8"}}
export default{async fetch(request,env){
 const origin=request.headers.get("Origin")||"*";
 if(request.method==="OPTIONS")return new Response(null,{status:204,headers:corsHeaders(origin)});
 if(request.method!=="POST")return new Response(JSON.stringify({error:"POST only"}),{status:405,headers:corsHeaders(origin)});
 try{
  if(!env.OPENAI_API_KEY)return new Response(JSON.stringify({error:"AI key is not configured"}),{status:500,headers:corsHeaders(origin)});
  const b=await request.json(),image=String(b?.image||"");
  if(!image.startsWith("data:image/"))return new Response(JSON.stringify({error:"image is required"}),{status:400,headers:corsHeaders(origin)});
  const r=await fetch("https://api.openai.com/v1/responses",{method:"POST",headers:{"Content-Type":"application/json","Authorization":"Bearer "+env.OPENAI_API_KEY},body:JSON.stringify({model:OPENAI_MODEL,input:[{role:"user",content:[
   {type:"input_text",text:"Read ONLY the MAIN CURRENT WEIGHT DISPLAY on the scale in the ENTIRE image. Ignore all printed specifications, maximum capacity, units, labels, dates and other numbers. Return ONLY the exact numeric string shown as the current weight. Preserve every digit, decimal point and trailing zero exactly. Never add, remove, round, correct, calculate or guess. If it shows 17.4 return exactly 17.4. If it shows 82.20 return exactly 82.20. If you cannot read the current weight confidently, return exactly NO_WEIGHT."},
   {type:"input_image",image_url:image,detail:"high"}]}],max_output_tokens:20})});
  const d=await r.json();
  if(!r.ok)return new Response(JSON.stringify({error:"OpenAI request failed",detail:d?.error?.message||"unknown error"}),{status:502,headers:corsHeaders(origin)});
  const raw=String(d?.output_text||"").trim();
  if(raw==="NO_WEIGHT"||!/^\d{1,3}(?:\.\d{1,2})?$/.test(raw))return new Response(JSON.stringify({error:"No exact weight found"}),{status:422,headers:corsHeaders(origin)});
  const weight=Number(raw);
  if(!Number.isFinite(weight)||weight<=0||weight>500)return new Response(JSON.stringify({error:"Invalid weight"}),{status:422,headers:corsHeaders(origin)});
  return new Response(JSON.stringify({weight,raw}),{status:200,headers:corsHeaders(origin)});
 }catch(e){return new Response(JSON.stringify({error:"Worker error",detail:String(e?.message||e)}),{status:500,headers:corsHeaders(origin)})}
}};
