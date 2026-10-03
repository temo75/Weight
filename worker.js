const OPENAI_MODEL = "gpt-4.1-mini";

function corsHeaders(origin){
  return {
    "Access-Control-Allow-Origin": origin || "*",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Allow-Methods": "POST, OPTIONS",
    "Content-Type": "application/json; charset=utf-8"
  };
}

export default {
  async fetch(request, env) {
    const origin = request.headers.get("Origin") || "*";

    if(request.method === "OPTIONS"){
      return new Response(null,{status:204,headers:corsHeaders(origin)});
    }

    if(request.method !== "POST"){
      return new Response(JSON.stringify({error:"POST only"}),{
        status:405,headers:corsHeaders(origin)
      });
    }

    try{
      const body = await request.json();
      const image = String(body?.image || "");

      if(!image.startsWith("data:image/")){
        return new Response(JSON.stringify({error:"image is required"}),{
          status:400,headers:corsHeaders(origin)
        });
      }

      const apiResponse = await fetch("https://api.openai.com/v1/responses",{
        method:"POST",
        headers:{
          "Content-Type":"application/json",
          "Authorization":"Bearer "+env.OPENAI_API_KEY
        },
        body:JSON.stringify({
          model:OPENAI_MODEL,
          input:[{
            role:"user",
            content:[
              {
                type:"input_text",
                text:
                  "Read ONLY the weight number visible in this image. " +
                  "It may be handwritten or displayed on a digital scale. " +
                  "Return ONLY the number, preserving the decimal point exactly. " +
                  "Do not guess. Do not add kg or any words. " +
                  "If the image shows 82.20, return exactly 82.20."
              },
              {
                type:"input_image",
                image_url:image,
                detail:"high"
              }
            ]
          }],
          max_output_tokens:20
        })
      });

      const data = await apiResponse.json();

      if(!apiResponse.ok){
        return new Response(JSON.stringify({
          error:"OpenAI request failed",
          detail:data?.error?.message || "unknown error"
        }),{
          status:502,headers:corsHeaders(origin)
        });
      }

      const text = String(data?.output_text || "")
        .trim()
        .replace(",", ".");

      const match = text.match(/\b\d{2,3}(?:\.\d{1,2})?\b/);
      if(!match){
        return new Response(JSON.stringify({error:"No weight found"}),{
          status:422,headers:corsHeaders(origin)
        });
      }

      const weight = Number(match[0]);
      if(!Number.isFinite(weight) || weight < 30 || weight > 250){
        return new Response(JSON.stringify({error:"Invalid weight"}),{
          status:422,headers:corsHeaders(origin)
        });
      }

      return new Response(JSON.stringify({
        weight,
        raw:match[0]
      }),{
        status:200,headers:corsHeaders(origin)
      });
    }catch(error){
      return new Response(JSON.stringify({
        error:"Worker error",
        detail:String(error?.message || error)
      }),{
        status:500,headers:corsHeaders(origin)
      });
    }
  }
};
