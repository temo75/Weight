/**
 * ═══════════════════════════════════════════════════════════════
 * !nstantly✓ OCR Module — powered by ALTRU.dev OCR Engine
 * ═══════════════════════════════════════════════════════════════
 *
 * Drop-in unified barcode scanning + text OCR for !nstantly✓.
 *
 * BARCODE SCANNING (3-tier auto-fallback):
 *   Tier 1: Native BarcodeDetector API (Chrome/Edge/Android — 0KB)
 *   Tier 2: Canvas heuristic barcode detection (all browsers)
 *   Tier 3: OCR digit extraction from printed barcode text
 *
 * TEXT OCR (PaddleOCR via ONNX Runtime Web):
 *   - Merchant/store name extraction
 *   - Expiry date detection
 *   - Promo/coupon detail extraction
 *   - Smart auto-categorization
 *
 * USAGE:
 *   const ocr = new InstantlyOCR();
 *   await ocr.init(msg => statusEl.textContent = msg);
 *
 *   // Full analysis (barcode + text + category)
 *   const result = await ocr.analyze(canvasOrImage);
 *   // result.barcode → { format, rawValue, tier }
 *   // result.merchantName → "Costco"
 *   // result.expiryDate → "12/25/2026"
 *   // result.category → "coupon"
 *   // result.suggestedCard → { name, barcodeData, barcodeType, ... }
 *
 *   // Just barcode
 *   const codes = await ocr.scanBarcode(source);
 *
 *   // Just text
 *   const text = await ocr.extractText(source);
 *
 *   // Live camera scanning
 *   ocr.startCameraScan(videoEl, result => { ... });
 *   ocr.stopCameraScan();
 *
 * PLATFORMS: Browser PWA, Android (Capacitor), iOS (Capacitor)
 * DEPENDENCIES: onnxruntime-web (loaded from CDN on first OCR use)
 * LICENSE: MIT — ALTRU.dev, Code for Humanity
 * ═══════════════════════════════════════════════════════════════
 */

(function(global) {
'use strict';

var ORT_CDN = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.min.js';


/* ═══════════════════════════════════════════════════════════════
   EMBEDDED ENGINE: Preprocessing + PaddleOCR ONNX + Layout
   (from ALTRU.dev OCR Engine — MIT License)
   ═══════════════════════════════════════════════════════════════ */

/* ═══════════════════════════════════════════════════════════════
   SECTION 1: PREPROCESSING PIPELINE
   ═══════════════════════════════════════════════════════════════ */

function createCanvas(w, h) {
  if (typeof OffscreenCanvas !== 'undefined') return new OffscreenCanvas(w, h);
  const c = document.createElement('canvas'); c.width = w; c.height = h; return c;
}

function getCtx(canvas) {
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('Failed to get 2D context');
  return ctx;
}

function clamp(v, lo = 0, hi = 255) { return v < lo ? lo : v > hi ? hi : v; }
function lum(r, g, b) { return 0.299 * r + 0.587 * g + 0.114 * b; }

function imageDataFromSource(source) {
  if (source instanceof ImageData) return source;
  const w = source.naturalWidth || source.width;
  const h = source.naturalHeight || source.height;
  const c = createCanvas(w, h);
  const ctx = getCtx(c);
  ctx.drawImage(source, 0, 0);
  return ctx.getImageData(0, 0, w, h);
}

function imageDataToCanvas(imageData) {
  const c = createCanvas(imageData.width, imageData.height);
  getCtx(c).putImageData(imageData, 0, 0);
  return c;
}

// ── Grayscale ──
function toGrayscale(imageData) {
  const { data, width, height } = imageData;
  const out = new Uint8ClampedArray(data.length);
  for (let i = 0; i < data.length; i += 4) {
    const g = lum(data[i], data[i+1], data[i+2]);
    out[i] = out[i+1] = out[i+2] = g;
    out[i+3] = data[i+3];
  }
  return new ImageData(out, width, height);
}

// ── CLAHE ──
function clahe(imageData, tileSize = 8, clipLimit = 2.0) {
  const gray = toGrayscale(imageData);
  const { data, width, height } = gray;
  const out = new Uint8ClampedArray(data.length);
  const tilesX = Math.max(1, Math.round(width / tileSize));
  const tilesY = Math.max(1, Math.round(height / tileSize));
  const tw = width / tilesX, th = height / tilesY;

  const cdfs = [];
  for (let ty = 0; ty < tilesY; ty++) {
    cdfs[ty] = [];
    for (let tx = 0; tx < tilesX; tx++) {
      const hist = new Float64Array(256);
      const x0 = Math.round(tx * tw), y0 = Math.round(ty * th);
      const x1 = Math.round((tx+1) * tw), y1 = Math.round((ty+1) * th);
      const area = Math.max(1, (x1-x0) * (y1-y0));
      for (let y = y0; y < y1 && y < height; y++)
        for (let x = x0; x < x1 && x < width; x++)
          hist[data[(y*width+x)*4]]++;
      const limit = Math.max(1, clipLimit * (area/256));
      let excess = 0;
      for (let i = 0; i < 256; i++) { if (hist[i]>limit) { excess+=hist[i]-limit; hist[i]=limit; } }
      const inc = excess/256;
      for (let i = 0; i < 256; i++) hist[i] += inc;
      const cdf = new Float64Array(256);
      cdf[0] = hist[0];
      for (let i = 1; i < 256; i++) cdf[i] = cdf[i-1]+hist[i];
      const cMin = cdf[0], cMax = cdf[255], den = Math.max(1, cMax-cMin);
      for (let i = 0; i < 256; i++) cdf[i] = Math.round(((cdf[i]-cMin)/den)*255);
      cdfs[ty][tx] = cdf;
    }
  }
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const pi = (y*width+x)*4, v = data[pi];
      const ftx = (x/tw)-0.5, fty = (y/th)-0.5;
      const tx0 = clamp(Math.floor(ftx),0,tilesX-1), ty0 = clamp(Math.floor(fty),0,tilesY-1);
      const tx1 = Math.min(tx0+1,tilesX-1), ty1 = Math.min(ty0+1,tilesY-1);
      const fx = Math.max(0,Math.min(1,ftx-tx0)), fy = Math.max(0,Math.min(1,fty-ty0));
      const r = (1-fy)*((1-fx)*cdfs[ty0][tx0][v]+fx*cdfs[ty0][tx1][v])+fy*((1-fx)*cdfs[ty1][tx0][v]+fx*cdfs[ty1][tx1][v]);
      out[pi]=out[pi+1]=out[pi+2]=clamp(Math.round(r)); out[pi+3]=255;
    }
  }
  return new ImageData(out, width, height);
}

// ── Sauvola Binarization ──
function sauvola(imageData, windowSize = 15, k = 0.2, R = 128) {
  const { width, height } = imageData;
  const gray = toGrayscale(imageData);
  const src = gray.data;
  const out = new Uint8ClampedArray(src.length);
  const n = width * height;
  const integral = new Float64Array(n), integralSq = new Float64Array(n);
  for (let y = 0; y < height; y++) {
    let rs = 0, rss = 0;
    for (let x = 0; x < width; x++) {
      const idx = y*width+x, v = src[idx*4];
      rs += v; rss += v*v;
      integral[idx] = rs + (y>0 ? integral[(y-1)*width+x] : 0);
      integralSq[idx] = rss + (y>0 ? integralSq[(y-1)*width+x] : 0);
    }
  }
  const half = Math.floor(windowSize/2);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const y1=Math.max(0,y-half)-1, y2=Math.min(height-1,y+half);
      const x1=Math.max(0,x-half)-1, x2=Math.min(width-1,x+half);
      const area=(y2-(y1<0?-1:y1))*(x2-(x1<0?-1:x1));
      let sum=integral[y2*width+x2], ssq=integralSq[y2*width+x2];
      if(y1>=0){sum-=integral[y1*width+x2];ssq-=integralSq[y1*width+x2];}
      if(x1>=0){sum-=integral[y2*width+x1];ssq-=integralSq[y2*width+x1];}
      if(y1>=0&&x1>=0){sum+=integral[y1*width+x1];ssq+=integralSq[y1*width+x1];}
      const mean=sum/area, std=Math.sqrt(Math.max(0,(ssq/area)-mean*mean));
      const thresh=mean*(1+k*(std/R-1));
      const pi=(y*width+x)*4, v=src[pi]>thresh?255:0;
      out[pi]=out[pi+1]=out[pi+2]=v; out[pi+3]=255;
    }
  }
  return new ImageData(out, width, height);
}

// ── Deskew ──
function estimateSkewAngle(imageData, range = 10, steps = 40) {
  const gray = toGrayscale(imageData);
  const { width, height } = gray;
  const scale = Math.min(1, 400 / Math.max(width, height));
  const sw = Math.round(width*scale), sh = Math.round(height*scale);
  const tmp = createCanvas(width, height);
  getCtx(tmp).putImageData(gray, 0, 0);
  const tc = createCanvas(sw, sh);
  const ctx = getCtx(tc);
  let bestAngle = 0, bestVar = -1;
  for (let i = 0; i <= steps; i++) {
    const angle = -range + (2*range*i)/steps;
    const rad = (angle*Math.PI)/180;
    ctx.clearRect(0,0,sw,sh);
    ctx.save(); ctx.translate(sw/2,sh/2); ctx.rotate(rad); ctx.drawImage(tmp,-sw/2,-sh/2,sw,sh); ctx.restore();
    const rot = ctx.getImageData(0,0,sw,sh);
    const hProj = new Float64Array(sh);
    for (let y=0;y<sh;y++) { let s=0; for(let x=0;x<sw;x++) s+=(255-rot.data[(y*sw+x)*4]); hProj[y]=s; }
    let mean=0; for(let y=0;y<sh;y++) mean+=hProj[y]; mean/=sh;
    let v=0; for(let y=0;y<sh;y++) v+=(hProj[y]-mean)**2; v/=sh;
    if(v>bestVar){bestVar=v;bestAngle=angle;}
  }
  return bestAngle;
}

function deskew(imageData, angle) {
  if (angle === undefined || angle === null) angle = estimateSkewAngle(imageData);
  if (Math.abs(angle) < 0.15) return imageData;
  const { width, height } = imageData;
  const rad = (-angle*Math.PI)/180;
  const cos=Math.abs(Math.cos(rad)), sin=Math.abs(Math.sin(rad));
  const nw=Math.round(width*cos+height*sin), nh=Math.round(width*sin+height*cos);
  const src = createCanvas(width, height); getCtx(src).putImageData(imageData, 0, 0);
  const dst = createCanvas(nw, nh); const ctx = getCtx(dst);
  ctx.fillStyle='#FFF'; ctx.fillRect(0,0,nw,nh);
  ctx.translate(nw/2,nh/2); ctx.rotate(rad); ctx.drawImage(src,-width/2,-height/2);
  return ctx.getImageData(0,0,nw,nh);
}

// ── Median Filter ──
function medianFilter(imageData, radius = 1) {
  const { data, width, height } = imageData;
  const out = new Uint8ClampedArray(data.length);
  const size = (2*radius+1)**2;
  const buf = new Uint8Array(size);
  for (let y=0;y<height;y++) for(let x=0;x<width;x++) {
    for(let ch=0;ch<3;ch++) {
      let cnt=0;
      for(let dy=-radius;dy<=radius;dy++) for(let dx=-radius;dx<=radius;dx++) {
        const ny=clamp(y+dy,0,height-1), nx=clamp(x+dx,0,width-1);
        buf[cnt++]=data[(ny*width+nx)*4+ch];
      }
      buf.subarray(0,cnt).sort();
      out[(y*width+x)*4+ch]=buf[cnt>>1];
    }
    out[(y*width+x)*4+3]=255;
  }
  return new ImageData(out, width, height);
}

// ── Scale Normalization ──
function normalizeScale(imageData, minH = 800, maxH = 2500) {
  const { width, height } = imageData;
  if (height >= minH && height <= maxH) return { imageData, factor: 1 };
  const scale = height < minH ? minH/height : maxH/height;
  const nw = Math.round(width*scale), nh = Math.round(height*scale);
  const src = createCanvas(width,height); getCtx(src).putImageData(imageData,0,0);
  const dst = createCanvas(nw,nh); const ctx = getCtx(dst);
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(src,0,0,nw,nh);
  return { imageData: ctx.getImageData(0,0,nw,nh), factor: scale };
}

// ── Composite Pipeline ──
const PRESETS = {
  auto:        { clahe:true,  deskew:true,  denoise:false, binarize:false, claheClip:2.0 },
  cleanScan:   { clahe:false, deskew:true,  denoise:false, binarize:false, claheClip:2.0 },
  phonePhoto:  { clahe:true,  deskew:true,  denoise:true,  binarize:false, claheClip:3.0 },
  degradedScan:{ clahe:true,  deskew:true,  denoise:true,  binarize:true,  claheClip:3.0 },
  receipt:     { clahe:true,  deskew:true,  denoise:false, binarize:true,  claheClip:4.0 },
};

function preprocess(imageData, preset = 'auto') {
  const p = PRESETS[preset] || PRESETS.auto;
  const meta = { skewAngle: null, scaled: false, factor: 1, denoised: false, binarized: false };
  let img = imageData;

  // Scale
  const sc = normalizeScale(img);
  img = sc.imageData; meta.factor = sc.factor; meta.scaled = sc.factor !== 1;

  // CLAHE
  if (p.clahe) img = clahe(img, 8, p.claheClip);

  // Grayscale
  img = toGrayscale(img);

  // Deskew
  if (p.deskew) {
    const angle = estimateSkewAngle(img);
    if (Math.abs(angle) >= 0.15) { img = deskew(img, angle); meta.skewAngle = angle; }
  }

  // Denoise
  if (p.denoise) { img = medianFilter(img, 1); meta.denoised = true; }

  // Binarize
  if (p.binarize) { img = sauvola(img, 15, 0.2); meta.binarized = true; }

  return { imageData: img, meta };
}


/* ═══════════════════════════════════════════════════════════════
   SECTION 2: PADDLEOCR ONNX INFERENCE (VENDORED)
   ═══════════════════════════════════════════════════════════════ */

const DEFAULT_MODELS = {
  det: 'https://raw.githubusercontent.com/niceandbright/ppu-paddle-ocr-models/main/models/det/en_PP-OCRv3_det_infer.onnx',
  rec: 'https://raw.githubusercontent.com/niceandbright/ppu-paddle-ocr-models/main/models/rec/en_PP-OCRv4_rec_infer.onnx',
  dict:'https://raw.githubusercontent.com/niceandbright/ppu-paddle-ocr-models/main/keys/en_dict.txt',
};

class PaddleOCR {
  constructor() { this.detSession = null; this.recSession = null; this.dictionary = []; }

  async init(onProgress) {
    onProgress?.('Loading ONNX Runtime...');
    // ort is loaded globally from CDN in index.html
    const sess = ort.InferenceSession;
    const opts = { executionProviders: ['wasm'], graphOptimizationLevel: 'all' };

    onProgress?.('Downloading detection model...');
    const detBuf = await fetch(DEFAULT_MODELS.det).then(r => r.arrayBuffer());
    this.detSession = await sess.create(detBuf, opts);

    onProgress?.('Downloading recognition model...');
    const recBuf = await fetch(DEFAULT_MODELS.rec).then(r => r.arrayBuffer());
    this.recSession = await sess.create(recBuf, opts);

    onProgress?.('Loading dictionary...');
    const dictText = await fetch(DEFAULT_MODELS.dict).then(r => r.text());
    this.dictionary = dictText.split('\n').filter(l => l.length > 0);

    onProgress?.('Ready.');
  }

  async recognize(imageData) {
    const { width, height, data } = imageData;

    // ── Detection preprocessing ──
    const maxSide = 960;
    let ratio = Math.max(width,height) > maxSide ? maxSide/Math.max(width,height) : 1;
    let rw = Math.max(32, Math.round(width*ratio/32)*32);
    let rh = Math.max(32, Math.round(height*ratio/32)*32);

    const tmpC = createCanvas(width,height); getCtx(tmpC).putImageData(imageData,0,0);
    const resC = createCanvas(rw,rh); getCtx(resC).drawImage(tmpC,0,0,rw,rh);
    const resD = getCtx(resC).getImageData(0,0,rw,rh);

    const mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225];
    const px = rw*rh;
    const chw = new Float32Array(3*px);
    for(let i=0;i<px;i++) {
      const pi=i*4;
      chw[i]=(resD.data[pi]/255-mean[0])/std[0];
      chw[px+i]=(resD.data[pi+1]/255-mean[1])/std[1];
      chw[2*px+i]=(resD.data[pi+2]/255-mean[2])/std[2];
    }

    const detT = new ort.Tensor('float32', chw, [1,3,rh,rw]);
    const detF = {}; detF[this.detSession.inputNames[0]] = detT;
    const detR = await this.detSession.run(detF);
    const heatmap = detR[this.detSession.outputNames[0]].data;

    // ── Extract boxes from heatmap ──
    const boxes = this._extractBoxes(heatmap, rw, rh, width, height);
    if (boxes.length === 0) return { text: '', lines: [] };

    // ── Recognition per box ──
    const lines = [];
    for (const box of boxes) {
      const crop = this._cropForRec(imageData, box);
      if (!crop) continue;
      const { tensor, cw, ch } = crop;
      const recT = new ort.Tensor('float32', tensor, [1,3,ch,cw]);
      const recF = {}; recF[this.recSession.inputNames[0]] = recT;
      const recR = await this.recSession.run(recF);
      const recOut = recR[this.recSession.outputNames[0]];
      const decoded = this._ctcDecode(recOut.data, recOut.dims[1], recOut.dims[2]);
      if (decoded.text.length > 0) lines.push({ text: decoded.text, score: decoded.score, box });
    }

    return { text: lines.map(l=>l.text).join('\n'), lines };
  }

  _extractBoxes(heatmap, w, h, origW, origH) {
    const binThresh=0.3, boxThresh=0.6, unclipRatio=1.5;
    const binary = new Uint8Array(w*h);
    for(let i=0;i<binary.length;i++) binary[i]=heatmap[i]>binThresh?1:0;

    const boxes=[], visited=new Uint8Array(w*h);
    for(let y=1;y<h-1;y++) for(let x=1;x<w-1;x++) {
      const idx=y*w+x;
      if(!binary[idx]||visited[idx]) continue;
      let minX=x,maxX=x,minY=y,maxY=y,sumS=0,cnt=0;
      const q=[idx]; visited[idx]=1;
      while(q.length>0) {
        const ci=q.pop(), cx=ci%w, cy=(ci/w)|0;
        minX=Math.min(minX,cx); maxX=Math.max(maxX,cx);
        minY=Math.min(minY,cy); maxY=Math.max(maxY,cy);
        sumS+=heatmap[ci]; cnt++;
        for(const [dx,dy] of [[1,0],[-1,0],[0,1],[0,-1]]) {
          const nx=cx+dx, ny=cy+dy;
          if(nx<0||nx>=w||ny<0||ny>=h) continue;
          const ni=ny*w+nx;
          if(!visited[ni]&&binary[ni]) { visited[ni]=1; q.push(ni); }
        }
      }
      if(sumS/cnt<boxThresh||cnt<10) continue;
      const bw=maxX-minX, bh=maxY-minY;
      const padX=bw*(unclipRatio-1)*0.5, padY=bh*(unclipRatio-1)*0.5;
      const sx=origW/w, sy=origH/h;
      const x0=Math.max(0,(minX-padX)*sx), y0=Math.max(0,(minY-padY)*sy);
      const x1=Math.min(origW,(maxX+padX)*sx), y1=Math.min(origH,(maxY+padY)*sy);
      if(x1-x0<3||y1-y0<3) continue;
      boxes.push([[x0|0,y0|0],[x1|0,y0|0],[x1|0,y1|0],[x0|0,y1|0]]);
    }
    boxes.sort((a,b) => { const dy=a[0][1]-b[0][1]; return Math.abs(dy)>10?dy:a[0][0]-b[0][0]; });
    return boxes;
  }

  _cropForRec(imageData, box) {
    const { width, height } = imageData;
    const x0=Math.max(0,Math.min(box[0][0],box[3][0])), y0=Math.max(0,Math.min(box[0][1],box[1][1]));
    const x1=Math.min(width,Math.max(box[1][0],box[2][0])), y1=Math.min(height,Math.max(box[2][1],box[3][1]));
    const cw=Math.round(x1-x0), ch_=Math.round(y1-y0);
    if(cw<2||ch_<2) return null;
    const rh=48, rw=Math.max(1,Math.round(cw*rh/ch_));
    const tmp=createCanvas(width,height); getCtx(tmp).putImageData(imageData,0,0);
    const crop=createCanvas(rw,rh); getCtx(crop).drawImage(tmp,x0,y0,cw,ch_,0,0,rw,rh);
    const cd=getCtx(crop).getImageData(0,0,rw,rh);
    const px=rw*rh, tensor=new Float32Array(3*px);
    for(let i=0;i<px;i++){const pi=i*4;tensor[i]=(cd.data[pi]/255-0.5)/0.5;tensor[px+i]=(cd.data[pi+1]/255-0.5)/0.5;tensor[2*px+i]=(cd.data[pi+2]/255-0.5)/0.5;}
    return { tensor, cw: rw, ch: rh };
  }

  _ctcDecode(output, seqLen, vocabSize) {
    let text='', prev=0, totalS=0, sCnt=0;
    for(let t=0;t<seqLen;t++) {
      let maxI=0,maxV=-Infinity;
      for(let v=0;v<vocabSize;v++){const val=output[t*vocabSize+v];if(val>maxV){maxV=val;maxI=v;}}
      const conf=1/(1+Math.exp(-maxV));
      if(maxI!==0&&maxI!==prev&&maxI-1<this.dictionary.length){text+=this.dictionary[maxI-1];totalS+=conf;sCnt++;}
      prev=maxI;
    }
    return { text: text.trim(), score: sCnt>0?totalS/sCnt:0 };
  }

  destroy() { this.detSession?.release?.(); this.recSession?.release?.(); }
}


/* ═══════════════════════════════════════════════════════════════
   SECTION 3: LAYOUT ANALYSIS + CONFIDENCE CALIBRATION
   ═══════════════════════════════════════════════════════════════ */

function analyzeLayout(lines, imgW, imgH) {
  if (lines.length === 0) return [];
  const bboxes = lines.map(l => {
    const xs=l.box.map(p=>p[0]), ys=l.box.map(p=>p[1]);
    return { x:Math.min(...xs), y:Math.min(...ys), w:Math.max(...xs)-Math.min(...xs), h:Math.max(...ys)-Math.min(...ys) };
  });
  const heights = bboxes.map(b=>b.h).sort((a,b)=>a-b);
  const medH = heights[heights.length>>1]||20;

  return lines.map((l,i) => {
    const bb = bboxes[i];
    let type = 'paragraph';
    if (bb.y < imgH*0.08) type = 'header';
    else if (bb.y+bb.h > imgH*0.92) type = 'footer';
    else if (bb.h/medH > 1.6 && l.text.length < 100) type = 'heading';
    else if (bb.h/medH < 0.7 && l.text.length < 80) type = 'caption';

    // Confidence calibration
    const logit = Math.log(Math.max(1e-8,l.score)/Math.max(1e-8,1-l.score));
    let conf = 1/(1+Math.exp(-logit));
    const len = l.text.trim().length;
    if(len===1) conf*=0.6; else if(len===2) conf*=0.8; else if(len<=4) conf*=0.9;
    conf = Math.min(1, Math.max(0, conf));

    return { text:l.text, rawScore:l.score, confidence:conf, box:l.box, bbox:bb, type, order:i };
  }).filter(r => r.confidence >= 0.25);
}




/* ═══════════════════════════════════════════════════════════════
   BARCODE SCANNING — TIER 1: Native BarcodeDetector API
   ═══════════════════════════════════════════════════════════════ */

var NativeBarcodeScanner = {
  available: false,
  detector: null,

  init: async function() {
    if (typeof BarcodeDetector === 'undefined') { this.available = false; return; }
    try {
      var formats = await BarcodeDetector.getSupportedFormats();
      if (formats.length > 0) {
        this.detector = new BarcodeDetector({ formats: formats });
        this.available = true;
      }
    } catch(e) { this.available = false; }
  },

  detect: async function(source) {
    if (!this.detector) return [];
    try {
      var results = await this.detector.detect(source);
      return results.map(function(r) {
        return { format: r.format, rawValue: r.rawValue, tier: 1,
                 boundingBox: r.boundingBox, cornerPoints: r.cornerPoints };
      });
    } catch(e) { return []; }
  }
};


/* ═══════════════════════════════════════════════════════════════
   BARCODE SCANNING — TIER 2: Canvas Heuristic Detection
   Scans horizontal lines looking for barcode-like patterns.
   If a pattern is found but can't be fully decoded, signals
   Tier 3 to try OCR on the printed digits below.
   ═══════════════════════════════════════════════════════════════ */

var HeuristicBarcodeScanner = {
  detect: function(source) {
    var imageData;
    if (source instanceof ImageData) {
      imageData = source;
    } else {
      var w = source.naturalWidth || source.videoWidth || source.width;
      var h = source.naturalHeight || source.videoHeight || source.height;
      var c = createCanvas(w, h);
      getCtx(c).drawImage(source, 0, 0);
      imageData = getCtx(c).getImageData(0, 0, w, h);
    }

    var data = imageData.data, width = imageData.width, height = imageData.height;
    var scanLines = [0.3, 0.4, 0.5, 0.6, 0.7];

    for (var s = 0; s < scanLines.length; s++) {
      var y = Math.round(height * scanLines[s]);
      // Build grayscale row
      var row = new Uint8Array(width);
      for (var x = 0; x < width; x++) {
        var i = (y * width + x) * 4;
        row[x] = Math.round(0.299 * data[i] + 0.587 * data[i+1] + 0.114 * data[i+2]);
      }
      // Binarize with local average
      var bin = new Uint8Array(width);
      var winSize = Math.max(15, Math.round(width * 0.05));
      for (var x2 = 0; x2 < width; x2++) {
        var lo = Math.max(0, x2 - winSize), hi = Math.min(width - 1, x2 + winSize);
        var sum = 0;
        for (var j = lo; j <= hi; j++) sum += row[j];
        bin[x2] = row[x2] < (sum / (hi - lo + 1)) ? 1 : 0;
      }
      // Count black runs
      var barRuns = 0, inBar = false;
      var widths = [];
      var runLen = 0;
      for (var x3 = 0; x3 < width; x3++) {
        if (bin[x3] === 1) {
          if (!inBar) { inBar = true; runLen = 0; }
          runLen++;
        } else {
          if (inBar) { barRuns++; widths.push(runLen); inBar = false; }
        }
      }
      if (inBar) { barRuns++; widths.push(runLen); }

      // 20+ bars with consistent widths = likely barcode
      if (barRuns >= 20 && widths.length > 0) {
        var avgW = widths.reduce(function(a,b){return a+b;},0) / widths.length;
        var variance = widths.reduce(function(a,w){return a+(w-avgW)*(w-avgW);},0) / widths.length;
        var cv = Math.sqrt(variance) / avgW;
        if (cv < 1.5) {
          return [{ format: 'linear_detected', rawValue: null, tier: 2,
                    _barcodeRegionY: y, _confidence: 1 - cv }];
        }
      }
    }
    return [];
  }
};


/* ═══════════════════════════════════════════════════════════════
   SMART TEXT EXTRACTION — Merchant, Dates, Promos
   ═══════════════════════════════════════════════════════════════ */

var DATE_PATTERNS = [
  /(?:exp(?:ir[ey]s?)?|valid\s*(?:thru|until|to)|best\s*before|use\s*by|good\s*thru)[:\s]*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})/i,
  /(?:exp(?:ir[ey]s?)?|valid\s*(?:thru|until|to))[:\s]*(\d{1,2}[\/\-\.]\d{2,4})/i,
  /(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})/
];

var PROMO_PATTERNS = [
  /(\d+%\s*off)/i, /(buy\s+\d+\s+get\s+\d+)/i, /(save\s+\$?\d+)/i,
  /(free\s+\w+)/i, /(\$\d+(?:\.\d{2})?\s*off)/i, /(bogo|b[12]g[12])/i,
  /(extra\s+\d+%)/i
];

function extractMerchantName(ocrLines) {
  if (!ocrLines || ocrLines.length === 0) return null;
  // Topmost, longest, non-numeric text = likely merchant
  var sorted = ocrLines.slice().sort(function(a, b) {
    var ay = Math.min(a.box[0][1], a.box[1][1]);
    var by = Math.min(b.box[0][1], b.box[1][1]);
    if (Math.abs(ay - by) > 30) return ay - by;
    return b.text.length - a.text.length;
  });
  for (var i = 0; i < Math.min(sorted.length, 5); i++) {
    var t = sorted[i].text.trim();
    if (t.length < 2) continue;
    if (/^\d+$/.test(t)) continue;
    if (/^https?:\/\//i.test(t)) continue;
    if (/^[\d\s\-\.\/]+$/.test(t)) continue;
    if (t.length > 2 && sorted[i].score > 0.4) return t;
  }
  return ocrLines[0] ? ocrLines[0].text.trim() : null;
}

function extractExpiryDate(ocrLines) {
  var fullText = ocrLines.map(function(l){return l.text;}).join(' ');
  for (var i = 0; i < DATE_PATTERNS.length; i++) {
    var match = fullText.match(DATE_PATTERNS[i]);
    if (match) return match[1] || match[0];
  }
  return null;
}

function extractPromoDetails(ocrLines) {
  var promos = [];
  var fullText = ocrLines.map(function(l){return l.text;}).join(' ');
  for (var i = 0; i < PROMO_PATTERNS.length; i++) {
    var match = fullText.match(PROMO_PATTERNS[i]);
    if (match) promos.push(match[1] || match[0]);
  }
  return promos.length > 0 ? promos : null;
}

function extractBarcodeDigits(ocrLines) {
  var candidates = [];
  for (var i = 0; i < ocrLines.length; i++) {
    var text = ocrLines[i].text.trim().replace(/\s/g, '');
    if (/^\d{8,20}$/.test(text)) {
      var fmt = text.length === 13 ? 'ean_13' :
                text.length === 12 ? 'upc_a' :
                text.length === 8 ? 'ean_8' : 'code_128';
      candidates.push({ value: text, format: fmt, score: ocrLines[i].score });
    }
  }
  candidates.sort(function(a,b){ return b.score - a.score; });
  return candidates[0] || null;
}


/* ═══════════════════════════════════════════════════════════════
   AUTO-CATEGORIZATION
   ═══════════════════════════════════════════════════════════════ */

var CATEGORY_ICONS = {
  membership:'🪪', loyalty:'⭐', coupon:'🏷️', boarding_pass:'✈️',
  ticket:'🎫', gift_card:'🎁', payment:'💳', id:'🆔', other:'📋'
};
var CATEGORY_LABELS = {
  membership:'Membership', loyalty:'Loyalty', coupon:'Coupon',
  boarding_pass:'Boarding Pass', ticket:'Ticket', gift_card:'Gift Card',
  payment:'Payment', id:'ID', other:'Other'
};

function autoCategorizeCombined(barcodeResult, ocrLines) {
  var fullText = (ocrLines||[]).map(function(l){return l.text;}).join(' ').toLowerCase();
  var barcodeData = (barcodeResult && barcodeResult.rawValue) ? barcodeResult.rawValue.toLowerCase() : '';
  var barcodeFormat = (barcodeResult && barcodeResult.format) ? barcodeResult.format : '';
  var combined = fullText + ' ' + barcodeData;

  if (/boarding|flight|gate|seat|airline|iata/i.test(combined)) return 'boarding_pass';
  if (/ticket|event|admit|concert|show|venue|stadium/i.test(combined)) return 'ticket';
  if (/coupon|promo|discount|%\s*off|save\s*\$|bogo|deal|offer/i.test(combined)) return 'coupon';
  if (/gift\s*card|gift\s*certificate|balance/i.test(combined)) return 'gift_card';
  if (/member|loyalty|reward|points|club|vip/i.test(combined)) return 'membership';
  if (/exp(?:ir|iry)|valid|good\s*thru|use\s*by/i.test(combined)) return 'coupon';
  if (barcodeFormat === 'pdf417' || barcodeFormat === 'aztec') return 'boarding_pass';
  if (barcodeFormat === 'ean_13' || barcodeFormat === 'upc_a') return 'loyalty';
  return 'other';
}


/* ═══════════════════════════════════════════════════════════════
   MAIN CLASS: InstantlyOCR
   ═══════════════════════════════════════════════════════════════ */

function InstantlyOCR(options) {
  this.options = options || {};
  this.ocrEngine = new PaddleOCR();
  this.ocrReady = false;
  this._ocrInitPromise = null;
  this._cameraLoop = null;
  this.barcodeTier = 0;
}

/** Initialize. Call once before any scan/extract. */
InstantlyOCR.prototype.init = async function(onProgress) {
  // Init native barcode scanner (instant, no downloads)
  await NativeBarcodeScanner.init();
  this.barcodeTier = NativeBarcodeScanner.available ? 1 : 2;
  if (onProgress) onProgress('Barcode: Tier ' + this.barcodeTier +
    (this.barcodeTier === 1 ? ' (native)' : ' (heuristic)'));

  // OCR is lazy-loaded on first use (downloads ~10MB models)
  if (this.options.eagerLoadOCR) {
    await this._initOCR(onProgress);
  }
  if (onProgress) onProgress('Ready.');
};

/** Internal: load OCR models (lazy, called on first text extraction). */
InstantlyOCR.prototype._initOCR = async function(onProgress) {
  if (this.ocrReady) return;
  if (this._ocrInitPromise) return this._ocrInitPromise;
  var self = this;
  this._ocrInitPromise = (async function() {
    // Load ONNX Runtime if not present
    if (typeof ort === 'undefined') {
      if (onProgress) onProgress('Loading ONNX Runtime...');
      await new Promise(function(resolve, reject) {
        var s = document.createElement('script');
        s.src = ORT_CDN; s.onload = resolve; s.onerror = reject;
        document.head.appendChild(s);
      });
    }
    await self.ocrEngine.init(onProgress);
    self.ocrReady = true;
    self._ocrInitPromise = null;
  })();
  return this._ocrInitPromise;
};

/**
 * Scan barcodes from an image source. Uses best available tier.
 * @param {HTMLImageElement|HTMLCanvasElement|HTMLVideoElement|ImageBitmap} source
 * @returns {Promise<Array<{format, rawValue, tier}>>}
 */
InstantlyOCR.prototype.scanBarcode = async function(source) {
  var results = [];

  // Tier 1: Native
  if (NativeBarcodeScanner.available) {
    var native = await NativeBarcodeScanner.detect(source);
    if (native.length > 0) return native;
  }

  // Tier 2: Heuristic
  var heuristic = HeuristicBarcodeScanner.detect(source);
  for (var i = 0; i < heuristic.length; i++) {
    if (heuristic[i].rawValue) results.push(heuristic[i]);
  }

  // Tier 3: OCR digit extraction
  if (results.length === 0 || results.every(function(r){return !r.rawValue;})) {
    await this._initOCR(this.options.onProgress);
    var imageData = (source instanceof ImageData) ? source :
      (function() {
        var w = source.naturalWidth || source.videoWidth || source.width;
        var h = source.naturalHeight || source.videoHeight || source.height;
        var c = createCanvas(w, h); getCtx(c).drawImage(source, 0, 0);
        return getCtx(c).getImageData(0, 0, w, h);
      })();
    var preprocessed = preprocess(imageData, 'auto');
    var ocrLines = await this.ocrEngine.recognize(preprocessed.imageData);
    var digits = extractBarcodeDigits(ocrLines);
    if (digits) {
      results.push({ format: digits.format, rawValue: digits.value, tier: 3, _ocrScore: digits.score });
    }
  }

  return results;
};

/**
 * Extract text from an image. Initializes OCR on first call.
 * @returns {Promise<{lines, fullText, merchantName, expiryDate, promoDetails}>}
 */
InstantlyOCR.prototype.extractText = async function(source) {
  await this._initOCR(this.options.onProgress);
  var imageData = (source instanceof ImageData) ? source :
    (function() {
      var w = source.naturalWidth || source.videoWidth || source.width;
      var h = source.naturalHeight || source.videoHeight || source.height;
      var c = createCanvas(w, h); getCtx(c).drawImage(source, 0, 0);
      return getCtx(c).getImageData(0, 0, w, h);
    })();
  var preprocessed = preprocess(imageData, 'phonePhoto');
  var lines = await this.ocrEngine.recognize(preprocessed.imageData);
  return {
    lines: lines,
    fullText: lines.map(function(l){return l.text;}).join('\n'),
    merchantName: extractMerchantName(lines),
    expiryDate: extractExpiryDate(lines),
    promoDetails: extractPromoDetails(lines)
  };
};

/**
 * Full analysis: barcode + text + auto-categorize.
 * Main method for !nstantly✓ PRO "smart import".
 * @returns {Promise<InstantlyAnalysisResult>}
 */
InstantlyOCR.prototype.analyze = async function(source) {
  // Run barcode and text extraction in parallel
  var barcodePromise = this.scanBarcode(source);
  var textPromise = this.extractText(source);
  var results = await Promise.all([barcodePromise, textPromise]);
  var barcodes = results[0];
  var textData = results[1];

  var bestBarcode = barcodes[0] || null;
  var category = autoCategorizeCombined(bestBarcode, textData.lines);

  return {
    barcode: bestBarcode ? {
      format: bestBarcode.format,
      rawValue: bestBarcode.rawValue,
      tier: bestBarcode.tier
    } : null,
    allBarcodes: barcodes,

    merchantName: textData.merchantName,
    expiryDate: textData.expiryDate,
    promoDetails: textData.promoDetails,
    fullText: textData.fullText,
    ocrLines: textData.lines,

    category: category,
    categoryIcon: CATEGORY_ICONS[category] || '📋',
    categoryLabel: CATEGORY_LABELS[category] || 'Other',

    suggestedCard: {
      name: textData.merchantName || 'Unknown Card',
      barcodeData: bestBarcode ? bestBarcode.rawValue : null,
      barcodeType: bestBarcode ? bestBarcode.format : null,
      category: category,
      expiryDate: textData.expiryDate || null,
      notes: textData.promoDetails ? textData.promoDetails.join(', ') : null
    }
  };
};

/**
 * Start continuous camera barcode scanning.
 * @param {HTMLVideoElement} videoElement — must have active stream
 * @param {function} onResult — callback(result)
 * @param {object} opts — { intervalMs: 200, autoStop: true }
 */
InstantlyOCR.prototype.startCameraScan = function(videoElement, onResult, opts) {
  opts = opts || {};
  var intervalMs = opts.intervalMs || 200;
  var autoStop = opts.autoStop !== false;
  var self = this;
  var scanning = true;

  function scan() {
    if (!scanning || videoElement.paused || videoElement.ended) return;
    self.scanBarcode(videoElement).then(function(barcodes) {
      if (barcodes.length > 0 && barcodes[0].rawValue) {
        onResult({ barcode: barcodes[0], allBarcodes: barcodes });
        if (autoStop) { scanning = false; return; }
      }
      if (scanning) self._cameraLoop = setTimeout(scan, intervalMs);
    }).catch(function() {
      if (scanning) self._cameraLoop = setTimeout(scan, intervalMs);
    });
  }
  scan();
};

/** Stop continuous camera scanning. */
InstantlyOCR.prototype.stopCameraScan = function() {
  if (this._cameraLoop) { clearTimeout(this._cameraLoop); this._cameraLoop = null; }
};

/** Platform capabilities report. */
InstantlyOCR.prototype.getCapabilities = function() {
  return {
    nativeBarcodeDetector: NativeBarcodeScanner.available,
    barcodeTier: this.barcodeTier,
    ocrLoaded: this.ocrReady,
    camera: !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia),
    platform: /iPhone|iPad|iPod/.test(navigator.userAgent) ? 'ios' :
              /Android/.test(navigator.userAgent) ? 'android' : 'desktop'
  };
};

/** Release resources. */
InstantlyOCR.prototype.destroy = function() {
  this.stopCameraScan();
  this.ocrEngine.destroy();
  this.ocrReady = false;
};

/* ═══════════════════════════════════════════════════════════════
   EXPORTS
   ═══════════════════════════════════════════════════════════════ */

global.InstantlyOCR = InstantlyOCR;
global.InstantlyOCR.CATEGORY_ICONS = CATEGORY_ICONS;
global.InstantlyOCR.CATEGORY_LABELS = CATEGORY_LABELS;

})(typeof globalThis !== 'undefined' ? globalThis : typeof window !== 'undefined' ? window : self);
