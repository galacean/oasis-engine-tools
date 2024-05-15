import { type PackingItem } from "./interface/PackingItem";
import { type PackingOutput } from "./interface/PackingOutput";
import { type PackingOption } from "./interface/PackingOption";
import { ErrorCode } from "./enum/ErrorCode";
import { OptDrawing } from "./opt/OptDrawing";
import { OptLoadImage } from "./opt/OptLoadImage";
import { OptPacking } from "./opt/OptPacking";
import { MaxRectsMethod } from "@galacean/tools-atlas-algorithm";

const packingPipe = [new OptLoadImage(), new OptPacking(), new OptDrawing()];
const DefaultOption = {
  width: 1024,
  height: 1024,
  padding: 1,
  allowRotate: false,
  square: false,
  pot: false,
  method: MaxRectsMethod.BestLongSideFit
};

/**
 * 将散图打包成图集
 * @param images 散图数组
 * @param option 打包配置
 * @returns
 */
export async function pack(images: PackingItem[], option?: PackingOption): Promise<PackingOutput> {
  if (option === undefined) {
    option = DefaultOption;
  } else {
    const keys = Object.keys(DefaultOption);
    for (let i = keys.length - 1; i >= 0; i--) {
      const key = keys[i];
      if (option[key] === undefined) {
        option[key] = DefaultOption[key];
      }
    }
  }
  // 开始打包
  const outPut: PackingOutput = {
    code: ErrorCode.Success,
    msg: "打包成功！",
    format: 1,
    atlasItems: []
  };
  const context = { option, images, outPut };
  for (let i = 0; i < packingPipe.length; i++) {
    const optCode = await packingPipe[i].parse(context);
    if (optCode !== ErrorCode.Success) {
      context.outPut.code = optCode;
      context.outPut.msg = getErrorMessage(optCode);
      break;
    }
  }
  return outPut;
}

function getErrorMessage(code: ErrorCode) {
  switch (code) {
    case ErrorCode.Success:
      return "打包成功！";
    case ErrorCode.NoImage:
      return "传入图片长度为0.";
    case ErrorCode.ImageLoadError:
      return "图片加载失败.";
    case ErrorCode.CanvasBlobError:
      return "画布 blob 转换失败.";
    case ErrorCode.OffscreenBlobError:
      return "离线画布 blob 转换失败.";
    case ErrorCode.PackError:
      return "图集尺寸限制，打包失败。";
    default:
      break;
  }
}

export { ErrorCode } from "./enum/ErrorCode";
export type { PackingOption } from "./interface/PackingOption";
export type { PackingOutput } from "./interface/PackingOutput";
export type { PackingItem } from "./interface/PackingItem";
