const CLASSES = ["plain", "comment", "string", "number", "keyword", "type", "function", "constant", "operator"];
let tokenizer;

export default function initializeTokenizer() {
  return tokenizer ??= import("/compact/pkg/tint_tokenizer.js").then(async module => (await module.default(), module.tokenize)).catch(error => {
    tokenizer = undefined;
    throw new Error(`Tokenizer initialization failed: ${error.message ?? error}`, { cause: error });
  });
}

export async function decodeModel(metadata, bytes) {
  if (typeof metadata === "string") {
    if (metadata.length >= 16384) throw new Error("Model metadata exceeds 16 KiB.");
    metadata = JSON.parse(metadata);
  }
  const keys = (value, expected) => value !== null && typeof value === "object" && !Array.isArray(value) &&
    Object.keys(value).length === expected.length && expected.every(key => Object.hasOwn(value, key));
  if (!keys(metadata, ["format_version", "feature_version", "architecture", "config", "classes", "quantization", "weights_sha256", "source_weights_sha256"]) ||
      metadata.format_version !== 1 || metadata.quantization !== "q4-block64-v1" ||
      JSON.stringify(metadata.classes) !== JSON.stringify(CLASSES) ||
      typeof metadata.weights_sha256 !== "string" || typeof metadata.source_weights_sha256 !== "string" ||
      !/^[a-f0-9]{64}$/.test(metadata.weights_sha256) || !/^[a-f0-9]{64}$/.test(metadata.source_weights_sha256)) {
    throw new Error("Unsupported model metadata.");
  }
  const state = metadata.config?.context_state === true;
  if (!((metadata.feature_version === 1 && metadata.architecture === "window-mlp-v1" && !state) ||
      (metadata.feature_version === 2 && metadata.architecture === "window-mlp-v2" && state)) ||
    !keys(metadata.config, state ? ["radius", "embedding_dim", "hidden_dim", "context_state"] : ["radius", "embedding_dim", "hidden_dim"])) {
    throw new Error("Unsupported model metadata.");
  }
  const { radius, embedding_dim: dim, hidden_dim: hidden } = metadata.config;
  if (!Number.isInteger(radius) || radius < 0 || radius > 16 ||
      !Number.isInteger(dim) || dim < 1 || dim > 128 ||
      !Number.isInteger(hidden) || hidden < 1 || hidden > 256) throw new Error("Unsupported model dimensions.");
  const count = 1365 * dim + ((2 * radius + 1) * dim + state * 9) * hidden + hidden * 10 + 9;
  if (!(bytes instanceof Uint8Array) || bytes.length !== ((count + 63) >> 6) * 4 + ((count + 1) >> 1)) {
    throw new Error("Model weight byte count mismatch.");
  }
  // Snapshot caller-owned input before the checksum await.
  const expectedChecksum = metadata.weights_sha256;
  bytes = bytes.slice();
  if (!crypto?.subtle) throw new Error("SHA-256 validation requires a secure context.");
  const checksum = Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)), value => value.toString(16).padStart(2, "0")).join("");
  if (checksum !== expectedChecksum) throw new Error("Model weight SHA-256 checksum mismatch.");
  const weights = new Float32Array(count);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let offset = 0;
  for (let start = 0; start < count; start += 64) {
    const length = Math.min(64, count - start);
    const scale = view.getFloat32(offset, true);
    offset += 4;
    if (!Number.isFinite(scale) || scale <= 0) throw new Error("Invalid q4 scale.");
    for (let i = 0; i < length; i++) {
      const code = (bytes[offset + (i >> 1)] >> ((i & 1) * 4)) & 15;
      if (code === 0) throw new Error("Invalid q4 code zero.");
      weights[start + i] = (code - 8) * scale;
      if (!Number.isFinite(weights[start + i])) throw new Error("Non-finite dequantized weight.");
    }
    if ((length & 1) && (bytes[offset + (length >> 1)] >> 4) !== 0) throw new Error("Invalid q4 padding nibble.");
    offset += (length + 1) >> 1;
  }
  return { config: { radius, embedding_dim: dim, hidden_dim: hidden, context_state: state }, weights };
}

function shader({ radius, embedding_dim: dim, hidden_dim: hidden, context_state: state }) {
  const base = (radius * 2 + 1) * dim;
  const hw = 1365 * dim, hb = hw + (base + state * 9) * hidden, ow = hb + hidden, ob = ow + hidden * 9;
  // Every entry point uses 64 threads, matching the (... / 64) divisors in
  // highlight() and the dispatch-limit asserts in gpu-check.js.
  if (!state) {
    return `
@group(0) @binding(0) var<storage,read> w:array<f32>;
@group(0) @binding(1) var<storage,read> ids:array<u32>;
@group(0) @binding(2) var<storage,read_write> e:array<f32>;
@group(0) @binding(3) var<storage,read_write> h:array<f32>;
@group(0) @binding(4) var<storage,read_write> labels:array<u32>;
@group(0) @binding(5) var<uniform> n:vec4<u32>;
@compute @workgroup_size(64) fn embedding(@builtin(global_invocation_id) g:vec3<u32>){
 let i=g.x;if(i>=n.w*${dim}u){return;}let t=i32(n.y+i/${dim}u)-${radius};let d=i%${dim}u;var sum=0.0;
 for(var f=0u;f<6u;f++){var id=0u;if(t>=0&&t<i32(n.z)){id=(ids[u32(t)*4u+1u+f/2u]>>((f%2u)*11u))&2047u;}sum+=w[id*${dim}u+d];}
 e[i]=sum/6.0;
}
@compute @workgroup_size(64) fn hidden(@builtin(global_invocation_id) g:vec3<u32>){
 let i=g.x;if(i>=n.x*${hidden}u){return;}let t=i/${hidden}u;let kind=ids[(n.y+t)*4u+1u]&2047u;
  if(kind==2u||kind==3u){return;}let d=i%${hidden}u;var sum=0.0;
  for(var k=0u;k<${base}u;k++){sum+=e[t*${dim}u+k]*w[${hw}u+k*${hidden}u+d];}
  h[i]=tanh(sum+w[${hb}u+d]);
}
@compute @workgroup_size(64) fn output(@builtin(global_invocation_id) g:vec3<u32>){
 let t=g.x;if(t>=n.x){return;}let absolute=n.y+t;let kind=ids[absolute*4u+1u]&2047u;
 if(kind==2u||kind==3u){labels[absolute]=0u;return;}
 var best=0u;var maximum=-3.402823466e+38;
 for(var c=0u;c<9u;c++){var sum=0.0;for(var k=0u;k<${hidden}u;k++){sum+=h[t*${hidden}u+k]*w[${ow}u+k*9u+c];}
 let value=sum+w[${ob}u+c];if(value>maximum){maximum=value;best=c;}}
 labels[absolute]=best;
}`;
  }
  const context = `let o=n.y+t;let w1=ids[o*4u+1u];let cst=(w1>>22u)&7u;let q=(w1>>25u)&3u;let r=(w1>>27u)&3u;var pst=0u;if(o>0u){pst=(ids[(o-1u)*4u+1u]>>22u)&7u;}var nxt=0u;if(o+1u<n.z){nxt=(ids[(o+1u)*4u+1u]>>22u)&7u;}sum+=f32(cst&1u)*w[${hw}u+${base}u*${hidden}u+d]+f32((cst>>1u)&1u)*w[${hw}u+${base + 1}u*${hidden}u+d]+f32((cst>>2u)&1u)*w[${hw}u+${base + 2}u*${hidden}u+d]+f32((cst&7u)!=(pst&7u))*w[${hw}u+${base + 3}u*${hidden}u+d]+f32((cst&7u)!=(nxt&7u))*w[${hw}u+${base + 4}u*${hidden}u+d]+f32(q&1u)*w[${hw}u+${base + 5}u*${hidden}u+d]+f32((q>>1u)&1u)*w[${hw}u+${base + 6}u*${hidden}u+d]+f32(r&1u)*w[${hw}u+${base + 7}u*${hidden}u+d]+f32((r>>1u)&1u)*w[${hw}u+${base + 8}u*${hidden}u+d];`;
  return `
@group(0) @binding(0) var<storage,read> w:array<f32>;
@group(0) @binding(1) var<storage,read> ids:array<u32>;
@group(0) @binding(2) var<storage,read_write> e:array<f32>;
@group(0) @binding(3) var<storage,read_write> h:array<f32>;
@group(0) @binding(4) var<storage,read_write> labels:array<u32>;
@group(0) @binding(5) var<uniform> n:vec4<u32>;
@compute @workgroup_size(64) fn embedding(@builtin(global_invocation_id) g:vec3<u32>){
 let i=g.x;if(i>=n.w*${dim}u){return;}let t=i32(n.y+i/${dim}u)-${radius};let d=i%${dim}u;var sum=0.0;
 for(var f=0u;f<6u;f++){var id=0u;if(t>=0&&t<i32(n.z)){id=(ids[u32(t)*4u+1u+f/2u]>>((f%2u)*11u))&2047u;}sum+=w[id*${dim}u+d];}
 e[i]=sum/6.0;
}
@compute @workgroup_size(64) fn hidden(@builtin(global_invocation_id) g:vec3<u32>){
 let i=g.x;if(i>=n.x*${hidden}u){return;}let t=i/${hidden}u;let kind=ids[(n.y+t)*4u+1u]&2047u;
  if(kind==2u||kind==3u){return;}let d=i%${hidden}u;var sum=0.0;
  for(var k=0u;k<${base}u;k++){sum+=e[t*${dim}u+k]*w[${hw}u+k*${hidden}u+d];}${context}
  h[i]=tanh(sum+w[${hb}u+d]);
}
@compute @workgroup_size(64) fn output(@builtin(global_invocation_id) g:vec3<u32>){
 let t=g.x;if(t>=n.x){return;}let absolute=n.y+t;let kind=ids[absolute*4u+1u]&2047u;
 if(kind==2u||kind==3u){labels[absolute]=0u;return;}
  var s0=0.0;var s1=s0;var s2=s0;var s3=s0;var s4=s0;var s5=s0;var s6=s0;var s7=s0;var s8=s0;
  for(var k=0u;k<${hidden}u;k++){let hv=h[t*${hidden}u+k];s0+=hv*w[${ow}u+k*9u];s1+=hv*w[${ow}u+k*9u+1u];s2+=hv*w[${ow}u+k*9u+2u];s3+=hv*w[${ow}u+k*9u+3u];s4+=hv*w[${ow}u+k*9u+4u];s5+=hv*w[${ow}u+k*9u+5u];s6+=hv*w[${ow}u+k*9u+6u];s7+=hv*w[${ow}u+k*9u+7u];s8+=hv*w[${ow}u+k*9u+8u];}
  var b=0u;var m=-3.402823466e+38;var v=0.0;
  v=s0+w[${ob}u];if(v>m){m=v;b=0u;}
  v=s1+w[${ob}u+1u];if(v>m){m=v;b=1u;}
  v=s2+w[${ob}u+2u];if(v>m){m=v;b=2u;}
  v=s3+w[${ob}u+3u];if(v>m){m=v;b=3u;}
  v=s4+w[${ob}u+4u];if(v>m){m=v;b=4u;}
  v=s5+w[${ob}u+5u];if(v>m){m=v;b=5u;}
  v=s6+w[${ob}u+6u];if(v>m){m=v;b=6u;}
  v=s7+w[${ob}u+7u];if(v>m){m=v;b=7u;}
  v=s8+w[${ob}u+8u];if(v>m){m=v;b=8u;}
  labels[absolute]=b;
}`;
}

export class Highlighter {
  static async load(metadata, bytes) {
    const { config, weights } = await decodeModel(metadata, bytes);
    if (!globalThis.navigator?.gpu) throw new Error("WebGPU is unavailable. There is no CPU fallback.");
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("WebGPU found no supported GPU adapter.");
    const tokenize = await initializeTokenizer();
    let device;
    const buffers = [];
    try {
      device = await adapter.requestDevice();
      const model = new Highlighter();
      const { radius, embedding_dim: dim, hidden_dim: hidden } = config;
      const invocations = device.limits.maxComputeWorkgroupsPerDimension * 64;
      const tileLimit = Math.min(32768, ~~(invocations / dim) - 2 * radius, ~~(invocations / hidden),
        ~~((16777216 - 8 * radius * dim) / (4 * (dim + hidden))));
      const tile = 2 ** Math.floor(Math.log2(tileLimit));
      const alignment = device.limits.minUniformBufferOffsetAlignment;
      Object.assign(model, { device, config, tokenize, buffers, tile, alignment, capacity: 0, busy: false, failure: null });
      device.lost.then(info => { model.failure ??= new Error(`WebGPU device lost: ${info.message || info.reason}`); });
      device.addEventListener("uncapturederror", event => { model.failure ??= new Error(`WebGPU error: ${event.error.message}`); });
      device.pushErrorScope("out-of-memory");
      device.pushErrorScope("validation");
      model.buffer = (size, usage) => {
        const value = device.createBuffer({ size, usage });
        buffers.push(value);
        return value;
      };
      const weightsBuffer = model.buffer(weights.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
      device.queue.writeBuffer(weightsBuffer, 0, weights);
      model.layout = device.createBindGroupLayout({ entries:
        ["read-only-storage", "read-only-storage", "storage", "storage", "storage", "uniform"].map((type, binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type, hasDynamicOffset: binding === 5 } })),
      });
      const module = device.createShaderModule({ code: shader(config) });
      const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [model.layout] });
      model.pipelines = await Promise.all(["embedding", "hidden", "output"].map(entryPoint => device.createComputePipelineAsync({ layout: pipelineLayout, compute: { module, entryPoint } })));
      const validation = await device.popErrorScope();
      const memory = await device.popErrorScope();
      if (validation || memory) throw new Error((validation || memory).message);
      if (model.failure) throw model.failure;
      return model;
    } catch (error) {
      for (const buffer of buffers) buffer.destroy();
      device?.destroy();
      throw new Error(`WebGPU initialization failed: ${error.message ?? error}`, { cause: error });
    }
  }

  async highlight(source) {
    if (this.failure) throw this.failure;
    if (this.busy) throw new Error("Concurrent highlight calls are not supported.");
    if (typeof source !== "string") throw new TypeError("Source must be a string.");
    if (source.length > 16777216) throw new Error("Source exceeds 16 MiB.");
    this.busy = true;
    try {
      const tokens = this.tokenize(source);
      const count = tokens.length / 4;
      if (!Number.isInteger(count) || count > 4194304) throw new Error("Source exceeds 4194304 tokens.");
      if (!count) return [];
      const spans = [];
      const { radius, embedding_dim: dim, hidden_dim: hidden } = this.config;
      const { device, tile, alignment } = this;
      device.pushErrorScope("validation");
      try {
        if (count > this.capacity) {
          // Previous readback is unmapped before growth; release old buffers before allocating replacements.
          for (const buffer of this.buffers.splice(1)) buffer.destroy();
          const capacity = 2 ** Math.ceil(Math.log2(count));
          const scratch = Math.min(capacity, tile);
          const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
          this.ids = this.buffer(capacity * 16, storage);
          const embedding = this.buffer((scratch + 2 * radius) * dim * 4, GPUBufferUsage.STORAGE);
          const hiddenBuffer = this.buffer(scratch * hidden * 4, GPUBufferUsage.STORAGE);
          this.labels = this.buffer(capacity * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
          this.uniforms = new Uint32Array(Math.ceil(capacity / tile) * alignment / 4);
          this.uniform = this.buffer(this.uniforms.byteLength, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
          this.readback = this.buffer(capacity * 4, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
          this.bindGroup = device.createBindGroup({ layout: this.layout, entries: [this.buffers[0], this.ids, embedding, hiddenBuffer, this.labels, this.uniform].map((buffer, binding) => ({ binding, resource: { buffer, size: binding === 5 ? 16 : buffer.size } })) });
          this.capacity = capacity;
        }
        for (let base = 0, offset = 0; base < count; base += tile, offset += alignment / 4) {
          const centers = Math.min(tile, count - base);
          this.uniforms.set([centers, base, count, centers + 2 * radius], offset);
        }
        device.queue.writeBuffer(this.ids, 0, tokens);
        device.queue.writeBuffer(this.uniform, 0, this.uniforms, 0, Math.ceil(count / tile) * alignment / 4);
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        for (let base = 0, offset = 0; base < count; base += tile, offset += alignment) {
          const centers = Math.min(tile, count - base);
          pass.setBindGroup(0, this.bindGroup, [offset]);
          const counts = [(centers + 2 * radius) * dim, centers * hidden, centers];
          for (let stage = 0; stage < 3; stage++) {
            pass.setPipeline(this.pipelines[stage]);
            pass.dispatchWorkgroups((counts[stage] + 63) >> 6);
          }
        }
        pass.end();
        encoder.copyBufferToBuffer(this.labels, 0, this.readback, 0, count * 4);
        device.queue.submit([encoder.finish()]);
      } finally {
        const validation = await device.popErrorScope();
        if (validation) throw this.failure ??= new Error(`WebGPU inference failed: ${validation.message}`);
      }
      await this.readback.mapAsync(GPUMapMode.READ, 0, count * 4).catch(error => {
        throw this.failure ?? new Error(`WebGPU readback failed: ${error.message ?? error}`, { cause: error });
      });
      try {
        if (this.failure) throw this.failure;
        const labels = new Uint32Array(this.readback.getMappedRange(0, count * 4));
        const ends = tokens, ids = labels;
        let index = 0, start = 0;
        while (index < count) {
          const id = ids[index];
          let next = index + 1;
          while (next < count && ids[next] === id) next++;
          const end = ends[(next - 1) * 4];
          const label = CLASSES[id];
          if (!label) throw new Error("WebGPU returned an invalid class.");
          spans.push({ start, end, class: label });
          start = end;
          index = next;
        }
      } finally {
        this.readback.unmap();
      }
      return spans;
    } finally {
      this.busy = false;
    }
  }

  dispose() {
    this.failure = new Error("Highlighter has been disposed.");
    for (const buffer of this.buffers) buffer.destroy();
    this.device.destroy();
  }
}
