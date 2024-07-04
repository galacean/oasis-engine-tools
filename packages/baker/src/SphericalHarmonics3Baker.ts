import { SphericalHarmonics3, TextureCube, TextureCubeFace } from "@galacean/engine";
import { WorkerManager } from "./WorkerManager";
import { DecodeMode } from "./enums/DecodeMode";

// multiply all elements of an array
const arrayProd = (arr) => arr.reduce((a, b) => a * b);

/**
 * Bake irradiance into spherical harmonics3 and use WebWorker.
 * @remarks
 * http://www.ppsloan.org/publications/StupidSH36.pdf
 */
export class SphericalHarmonics3Baker {
  private static async _decodeInWebGPU(
    dataPX: Uint8Array,
    dataNX: Uint8Array,
    dataPY: Uint8Array,
    dataNY: Uint8Array,
    dataPZ: Uint8Array,
    dataNZ: Uint8Array,
    textureSize: number,
    decodeMode: DecodeMode
  ): Promise<Float32Array> {
    console.time("WebGPU compute");
    const adapter = await navigator.gpu?.requestAdapter();
    const device = await adapter?.requestDevice();
    console.log(device);

    const workgroupSize = [1, 1, 1];
    const dispatchCount = [textureSize, textureSize, 1];
    const numThreadsPerWorkgroup = arrayProd(workgroupSize);

    const code = `
    @group(0) @binding(0) var<storage, read> dataPX: array<vec4u>;
    @group(0) @binding(1) var<storage, read> dataNX: array<vec4u>;
    @group(0) @binding(2) var<storage, read> dataPY: array<vec4u>;
    @group(0) @binding(3) var<storage, read> dataNY: array<vec4u>;
    @group(0) @binding(4) var<storage, read> dataPZ: array<vec4u>;
    @group(0) @binding(5) var<storage, read> dataNZ: array<vec4u>;
    // @group(0) @binding(6) var<storage, read_write> result: array<atomic<u32>>;
    @group(0) @binding(6) var<storage, read_write> solidAngleResult: array<f32>;
    @group(0) @binding(7) var<storage, read_write> shResult: array<f32>;


    fn RGBMToLinear(colorIn: vec4u) -> vec4f{
      var color = vec4f();
      let scale = colorIn.a / 13005; // (a * 5) / 255 / 255;

      color.r = f32(colorIn.r * scale);
      color.g = f32(colorIn.g * scale);
      color.b = f32(colorIn.b * scale);
      color.a = 1;

      return color;
    }

    fn computeSH(direction: vec3f, color: vec4f, deltaSolidAngle: f32, storeSHOffset: u32) {
      let x = direction[0];
      let y = direction[1];
      let z = direction[2];
      let r = color[0] * deltaSolidAngle;
      let g = color[1] * deltaSolidAngle;
      let b = color[2] * deltaSolidAngle;
      let bv0 = 0.282095; // basis0 = 0.886227
      let bv1 = -0.488603 * y; // basis1 = -0.488603
      let bv2 = 0.488603 * z; // basis2 = 0.488603
      let bv3 = -0.488603 * x; // basis3 = -0.488603
      let bv4 = 1.092548 * (x * y); // basis4 = 1.092548
      let bv5 = -1.092548 * (y * z); // basis5 = -1.092548
      let bv6 = 0.315392 * (3 * z * z - 1); // basis6 = 0.315392
      let bv7 = -1.092548 * (x * z); // basis7 = -1.092548
      let bv8 = 0.546274 * (x * x - y * y); // basis8 = 0.546274
    
      shResult[storeSHOffset] = r * bv0;
      shResult[storeSHOffset + 1] = g * bv0;
      shResult[storeSHOffset + 2] = b * bv0;
    
      shResult[storeSHOffset + 3] = r * bv1;
      shResult[storeSHOffset + 4] = g * bv1;
      shResult[storeSHOffset + 5] = b * bv1;
      shResult[storeSHOffset + 6] = r * bv2; 
      shResult[storeSHOffset + 7] = g * bv2;
      shResult[storeSHOffset + 8] = b * bv2;
      shResult[storeSHOffset + 9] = r * bv3; 
      shResult[storeSHOffset+ 10] = g * bv3;
      shResult[storeSHOffset+ 11] = b * bv3;
    
      shResult[storeSHOffset + 12] = r * bv4;
      shResult[storeSHOffset + 13] = g * bv4;
      shResult[storeSHOffset + 14] = b * bv4;
      shResult[storeSHOffset + 15] = r * bv5;
      shResult[storeSHOffset + 16] = g * bv5;
      shResult[storeSHOffset + 17] = b * bv5;
      shResult[storeSHOffset + 18] = r * bv6;
      shResult[storeSHOffset + 19] = g * bv6;
      shResult[storeSHOffset + 20] = b * bv6;
      shResult[storeSHOffset + 21] = r * bv7;
      shResult[storeSHOffset + 22] = g * bv7;
      shResult[storeSHOffset + 23] = b * bv7;
      shResult[storeSHOffset + 24] = r * bv8;
      shResult[storeSHOffset + 25] = g * bv8;
      shResult[storeSHOffset + 26] = b * bv8;
    }

    fn decodeFaceSH(faceIndex: u32, textureSize: u32, x:u32, y:u32){
      const channelLength = 4;
      let texelSize = f32(2 / textureSize); // convolution is in the space of [-1, 1]
      var color = vec4f();
      var direction = vec3f();
    
      let vStart = texelSize * 0.5 - 1;
      let uStart = vStart;
      let v = vStart + f32(y) * texelSize;
      let u = vStart + f32(y * textureSize + x) * texelSize;
      let dataOffset = y * textureSize * channelLength + x * channelLength;

      switch(faceIndex){
        case 0:{
          color = RGBMToLinear(dataPX[dataOffset]);
          direction[0] = 1;
          direction[1] = -v;
          direction[2] = -u;
          break;
        }
        case 1:{
          color = RGBMToLinear(dataNX[dataOffset]);
          direction[0] = -1;
          direction[1] = -v;
          direction[2] = u;
          break;
        }
        case 2:{
          color = RGBMToLinear(dataPY[dataOffset]);
          direction[0] = u;
          direction[1] = 1;
          direction[2] = v;
          break;
        }
        case 3:{
          color = RGBMToLinear(dataNY[dataOffset]);
          direction[0] = u;
          direction[1] = -1;
          direction[2] = -v;
          break;
        }
        case 4:{
          color = RGBMToLinear(dataPZ[dataOffset]);
          direction[0] = u;
          direction[1] = -v;
          direction[2] = 1;
          break;
        }
        case 5:{
          color = RGBMToLinear(dataNZ[dataOffset]);
          direction[0] = -u;
          direction[1] = -v;
          direction[2] = -1;
          break;
        }
        default :{}
      }

      /**
       * dA = cos = S / r = 4 / r
       * dw = dA / r2 = 4 / r / r2
       */
      let lengthSquared = direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2];
      let directionLength = sqrt(lengthSquared);
      let solidAngle = 4 / (directionLength * lengthSquared);

      // normalize
      direction[0] /= directionLength;
      direction[1] /= directionLength;
      direction[2] /= directionLength;

      let storeOffset = textureSize * textureSize * faceIndex + y * textureSize + x;
      let storeSHOffset = textureSize * textureSize * faceIndex * 27 + y * textureSize * 27 + x;

      solidAngleResult[storeOffset] = solidAngle;
      computeSH(direction, color, solidAngle, storeSHOffset);
    }

    @compute @workgroup_size(${workgroupSize})
    fn computeSomething(
        @builtin(workgroup_id) workgroup_id : vec3<u32>,
        @builtin(local_invocation_id) local_invocation_id : vec3<u32>,
        @builtin(global_invocation_id) global_invocation_id : vec3<u32>,
        @builtin(local_invocation_index) local_invocation_index: u32,
        @builtin(num_workgroups) num_workgroups: vec3<u32>
    ) {
      // let workgroup_index =
      //    workgroup_id.x +
      //    workgroup_id.y * num_workgroups.x +
      //    workgroup_id.z * num_workgroups.x * num_workgroups.y;

      // let global_invocation_index =
      //    workgroup_index * ${numThreadsPerWorkgroup} +
      //    local_invocation_index;

      let textureSize = num_workgroups.x;
      let x = workgroup_id.x;
      let y = workgroup_id.y;

      decodeFaceSH( 0, textureSize, x, y);
      decodeFaceSH( 1, textureSize, x, y);
      decodeFaceSH( 2, textureSize, x, y);
      decodeFaceSH( 3, textureSize, x, y);
      decodeFaceSH( 4, textureSize, x, y);
      decodeFaceSH( 5, textureSize, x, y);


    }
    `;

    const module = device.createShaderModule({ code });
    const pipeline = device.createComputePipeline({
      label: "compute pipeline",
      layout: "auto",
      compute: {
        module
      }
    });

    // Transform data to GPU
    const gpuWriteBuffer = [];
    [dataPX, dataNX, dataPY, dataNY, dataPZ, dataNZ].forEach(async (value, index) => {
      const u32Array = new Uint32Array(value);
      // Get a GPU buffer in a mapped state and an arrayBuffer for writing.
      gpuWriteBuffer[index] = device.createBuffer({
        mappedAtCreation: true,
        size: u32Array.byteLength,
        usage: GPUBufferUsage.STORAGE
      });
      const arrayBuffer = gpuWriteBuffer[index].getMappedRange();
      // Write bytes to buffer.
      new Uint32Array(arrayBuffer).set(u32Array);

      // Unmap buffer so that it can be used later for copy.
      gpuWriteBuffer[index].unmap();
    });

    // solidAngleResult buffer
    const solidAngleResultBuffer = device.createBuffer({
      size: textureSize * textureSize * 6 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });

    // shResult buffer
    const shResultBuffer = device.createBuffer({
      size: textureSize * textureSize * 6 * 27 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });

    // Create a bind group
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: gpuWriteBuffer[0] } },
        { binding: 1, resource: { buffer: gpuWriteBuffer[1] } },
        { binding: 2, resource: { buffer: gpuWriteBuffer[2] } },
        { binding: 3, resource: { buffer: gpuWriteBuffer[3] } },
        { binding: 4, resource: { buffer: gpuWriteBuffer[4] } },
        { binding: 5, resource: { buffer: gpuWriteBuffer[5] } },
        { binding: 6, resource: { buffer: solidAngleResultBuffer } },
        { binding: 7, resource: { buffer: shResultBuffer } }
      ]
    });

    // Encode commands to do the computation
    const encoder = device.createCommandEncoder({ label: "compute builtin encoder" });
    const pass = encoder.beginComputePass({ label: "compute builtin pass" });

    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(...dispatchCount);
    pass.end();

    // Get a GPU buffer for reading in an unmapped state.
    const solidAngleReadBuffer = device.createBuffer({
      size: textureSize * textureSize * 6 * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });

    // Encode commands for copying buffer to buffer.
    encoder.copyBufferToBuffer(
      solidAngleResultBuffer /* source buffer */,
      0 /* source offset */,
      solidAngleReadBuffer /* destination buffer */,
      0 /* destination offset */,
      textureSize * textureSize * 6 * 4 /* size */
    );

    //  Get a GPU buffer for reading in an unmapped state.
    const shReadBuffer = device.createBuffer({
      size: textureSize * textureSize * 6 * 27 * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });

    // Encode commands for copying buffer to buffer.
    encoder.copyBufferToBuffer(
      shResultBuffer /* source buffer */,
      0 /* source offset */,
      shReadBuffer /* destination buffer */,
      0 /* destination offset */,
      textureSize * textureSize * 6 * 27 * 4 /* size */
    );

    // // Finish encoding and submit the commands
    const commandBuffer = encoder.finish();
    device.queue.submit([commandBuffer]);

    // Read buffer.
    await Promise.all([solidAngleReadBuffer.mapAsync(GPUMapMode.READ), shReadBuffer.mapAsync(GPUMapMode.READ)]);

    const solidAngleResult = new Float32Array(solidAngleReadBuffer.getMappedRange());
    const shResult = new Float32Array(shReadBuffer.getMappedRange());
    const angleSum = solidAngleResult.reduce((a, b) => a + b);
    const scale = (4 * Math.PI) / angleSum;

    const result = new Float32Array(27);
    shResult.reduce((shResult, value, index) => {
      shResult[index % 27] += value * scale;
      return shResult;
    }, result);

    console.timeEnd("WebGPU compute");
    return result;
  }

  private static _decodeInCPU(
    dataPX: Uint8Array,
    dataNX: Uint8Array,
    dataPY: Uint8Array,
    dataNY: Uint8Array,
    dataPZ: Uint8Array,
    dataNZ: Uint8Array,
    textureSize: number,
    decodeMode: DecodeMode
  ): Float32Array {
    const result = new Float32Array(27);
    console.time("CPU compute");
    let solidAngleSum = 0;
    solidAngleSum = decodeFaceSH(dataPX, 0, decodeMode, textureSize, solidAngleSum, result);
    solidAngleSum = decodeFaceSH(dataNX, 1, decodeMode, textureSize, solidAngleSum, result);
    solidAngleSum = decodeFaceSH(dataPY, 2, decodeMode, textureSize, solidAngleSum, result);
    solidAngleSum = decodeFaceSH(dataNY, 3, decodeMode, textureSize, solidAngleSum, result);
    solidAngleSum = decodeFaceSH(dataPZ, 4, decodeMode, textureSize, solidAngleSum, result);
    solidAngleSum = decodeFaceSH(dataNZ, 5, decodeMode, textureSize, solidAngleSum, result);
    scaleSH(result, (4 * Math.PI) / solidAngleSum);

    console.timeEnd("CPU compute");
    return result;
  }

  /**
   * Bake from Cube texture and use WebWorker.
   * @param texture - Cube texture
   * @param out - SH3 for output
   * @param decodeMode - Mode of decoding texture cube, default DecodeMode.RGBM
   */
  static async fromTextureCube(
    texture: TextureCube,
    out: SphericalHarmonics3,
    decodeMode: DecodeMode = DecodeMode.RGBM
  ): Promise<SphericalHarmonics3> {
    const channelLength = 4;
    const textureSize = texture.width;

    // read pixel always return rgba
    const dataPX = new Uint8Array(textureSize * textureSize * channelLength);
    const dataNX = new Uint8Array(textureSize * textureSize * channelLength);
    const dataPY = new Uint8Array(textureSize * textureSize * channelLength);
    const dataNY = new Uint8Array(textureSize * textureSize * channelLength);
    const dataPZ = new Uint8Array(textureSize * textureSize * channelLength);
    const dataNZ = new Uint8Array(textureSize * textureSize * channelLength);
    texture.getPixelBuffer(TextureCubeFace.PositiveX, 0, 0, textureSize, textureSize, 0, dataPX);
    texture.getPixelBuffer(TextureCubeFace.NegativeX, 0, 0, textureSize, textureSize, 0, dataNX);
    texture.getPixelBuffer(TextureCubeFace.PositiveY, 0, 0, textureSize, textureSize, 0, dataPY);
    texture.getPixelBuffer(TextureCubeFace.NegativeY, 0, 0, textureSize, textureSize, 0, dataNY);
    texture.getPixelBuffer(TextureCubeFace.PositiveZ, 0, 0, textureSize, textureSize, 0, dataPZ);
    texture.getPixelBuffer(TextureCubeFace.NegativeZ, 0, 0, textureSize, textureSize, 0, dataNZ);

    let result;
    if (navigator.gpu) {
      // WebGPU is supported
      result = await SphericalHarmonics3Baker._decodeInWebGPU(
        dataPX,
        dataNX,
        dataPY,
        dataNY,
        dataPZ,
        dataNZ,
        textureSize,
        decodeMode
      );
      result = await SphericalHarmonics3Baker._decodeInCPU(
        dataPX,
        dataNX,
        dataPY,
        dataNY,
        dataPZ,
        dataNZ,
        textureSize,
        decodeMode
      );
    } else {
      // WebGPU is not supported, use WebWorker
      result = await WorkerManager.calculateSHFromTextureCube(
        dataPX,
        dataNX,
        dataPY,
        dataNY,
        dataPZ,
        dataNZ,
        textureSize,
        decodeMode
      );
    }

    out.copyFromArray(result);
    return out;
  }
}

export function RGBEToLinear(r: number, g: number, b: number, a: number, color: number[]) {
  if (a === 0) {
    color[0] = color[1] = color[2] = 0;
    color[3] = 1;
  } else {
    const scale = Math.pow(2, a - 128) / 255;
    color[0] = r * scale;
    color[1] = g * scale;
    color[2] = b * scale;
    color[3] = 1;
  }
}

export function RGBMToLinear(r: number, g: number, b: number, a: number, color: number[]) {
  const scale = a / 13005; // (a * 5) / 255 / 255;
  color[0] = r * scale;
  color[1] = g * scale;
  color[2] = b * scale;
  color[3] = 1;
}

export function gammaToLinearSpace(value: number): number {
  // https://www.khronos.org/registry/OpenGL/extensions/EXT/EXT_framebuffer_sRGB.txt
  // https://www.khronos.org/registry/OpenGL/extensions/EXT/EXT_texture_sRGB_decode.txt

  if (value <= 0.0) return 0.0;
  else if (value <= 0.04045) return value / 12.92;
  else if (value < 1.0) return Math.pow((value + 0.055) / 1.055, 2.4);
  else return Math.pow(value, 2.4);
}

export function addSH(direction: number[], color: number[], deltaSolidAngle: number, sh: Float32Array): void {
  const x = direction[0];
  const y = direction[1];
  const z = direction[2];
  const r = color[0] * deltaSolidAngle;
  const g = color[1] * deltaSolidAngle;
  const b = color[2] * deltaSolidAngle;
  const bv0 = 0.282095; // basis0 = 0.886227
  const bv1 = -0.488603 * y; // basis1 = -0.488603
  const bv2 = 0.488603 * z; // basis2 = 0.488603
  const bv3 = -0.488603 * x; // basis3 = -0.488603
  const bv4 = 1.092548 * (x * y); // basis4 = 1.092548
  const bv5 = -1.092548 * (y * z); // basis5 = -1.092548
  const bv6 = 0.315392 * (3 * z * z - 1); // basis6 = 0.315392
  const bv7 = -1.092548 * (x * z); // basis7 = -1.092548
  const bv8 = 0.546274 * (x * x - y * y); // basis8 = 0.546274

  (sh[0] += r * bv0), (sh[1] += g * bv0), (sh[2] += b * bv0);

  (sh[3] += r * bv1), (sh[4] += g * bv1), (sh[5] += b * bv1);
  (sh[6] += r * bv2), (sh[7] += g * bv2), (sh[8] += b * bv2);
  (sh[9] += r * bv3), (sh[10] += g * bv3), (sh[11] += b * bv3);

  (sh[12] += r * bv4), (sh[13] += g * bv4), (sh[14] += b * bv4);
  (sh[15] += r * bv5), (sh[16] += g * bv5), (sh[17] += b * bv5);
  (sh[18] += r * bv6), (sh[19] += g * bv6), (sh[20] += b * bv6);
  (sh[21] += r * bv7), (sh[22] += g * bv7), (sh[23] += b * bv7);
  (sh[24] += r * bv8), (sh[25] += g * bv8), (sh[26] += b * bv8);
}

export function scaleSH(array: Float32Array, scale: number): void {
  const src = array;
  (src[0] *= scale), (src[1] *= scale), (src[2] *= scale);
  (src[3] *= scale), (src[4] *= scale), (src[5] *= scale);
  (src[6] *= scale), (src[7] *= scale), (src[8] *= scale);
  (src[9] *= scale), (src[10] *= scale), (src[11] *= scale);
  (src[12] *= scale), (src[13] *= scale), (src[14] *= scale);
  (src[15] *= scale), (src[16] *= scale), (src[17] *= scale);
  (src[18] *= scale), (src[19] *= scale), (src[20] *= scale);
  (src[21] *= scale), (src[22] *= scale), (src[23] *= scale);
  (src[24] *= scale), (src[25] *= scale), (src[26] *= scale);
}

export function decodeFaceSH(
  faceData: Uint8Array,
  faceIndex: TextureCubeFace,
  decodeMode: DecodeMode,
  textureSize: number,
  lastSolidAngleSum: number,
  sh: Float32Array // length 27
): number {
  const channelLength = 4;
  const texelSize = 2 / textureSize; // convolution is in the space of [-1, 1]
  const color = [];
  const direction = [];

  let v = texelSize * 0.5 - 1;
  let solidAngleSum = lastSolidAngleSum;

  for (let y = 0; y < textureSize; y++) {
    let u = texelSize * 0.5 - 1;
    for (let x = 0; x < textureSize; x++) {
      const dataOffset = y * textureSize * channelLength + x * channelLength;
      switch (decodeMode) {
        case 0:
          color[0] = faceData[dataOffset];
          color[1] = faceData[dataOffset + 1];
          color[2] = faceData[dataOffset + 2];
          color[3] = 0;
          break;
        case 1:
          color[0] = gammaToLinearSpace(faceData[dataOffset] / 255);
          color[1] = gammaToLinearSpace(faceData[dataOffset + 1] / 255);
          color[2] = gammaToLinearSpace(faceData[dataOffset + 2] / 255);
          color[3] = 0;
          break;
        case 2:
          RGBEToLinear(
            faceData[dataOffset],
            faceData[dataOffset + 1],
            faceData[dataOffset + 2],
            faceData[dataOffset + 3],
            color
          );
          break;
        case 3:
          RGBMToLinear(
            faceData[dataOffset],
            faceData[dataOffset + 1],
            faceData[dataOffset + 2],
            faceData[dataOffset + 3],
            color
          );
          break;
      }

      switch (faceIndex) {
        case 0:
          direction[0] = 1;
          direction[1] = -v;
          direction[2] = -u;
          break;
        case 1:
          direction[0] = -1;
          direction[1] = -v;
          direction[2] = u;
          break;
        case 2:
          direction[0] = u;
          direction[1] = 1;
          direction[2] = v;
          break;
        case 3:
          direction[0] = u;
          direction[1] = -1;
          direction[2] = -v;
          break;
        case 4:
          direction[0] = u;
          direction[1] = -v;
          direction[2] = 1;
          break;
        case 5:
          direction[0] = -u;
          direction[1] = -v;
          direction[2] = -1;
          break;
      }

      /**
       * dA = cos = S / r = 4 / r
       * dw = dA / r2 = 4 / r / r2
       */
      const lengthSquared = direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2];
      const directionLength = Math.sqrt(lengthSquared);
      const solidAngle = 4 / (directionLength * lengthSquared);
      // normalize
      direction[0] /= directionLength;
      direction[1] /= directionLength;
      direction[2] /= directionLength;
      solidAngleSum += solidAngle;
      addSH(direction, color, solidAngle, sh);
      u += texelSize;
    }
    v += texelSize;
  }

  return solidAngleSum;
}
