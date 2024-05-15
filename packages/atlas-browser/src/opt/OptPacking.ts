import { MaxRectsBinPack, Rect } from "@galacean/tools-atlas-algorithm";
import { ErrorCode } from "../enum/ErrorCode";
import { PackingContext } from "../interface/PackingContext";
import { OptHandler } from "./OptHandler";

const POTS = [2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2];

export class OptPacking extends OptHandler {
  private pack: MaxRectsBinPack = new MaxRectsBinPack();
  parse(context: PackingContext): ErrorCode {
    const { option, images } = context;
    const { padding, square, pot, method, allowRotate } = option;
    let optionWidth: number;
    let optionHeight: number;
    if (square) {
      optionWidth = optionHeight = Math.min(option.width, option.height);
    } else {
      optionWidth = option.width;
      optionHeight = option.height;
    }
    if (pot) {
      let widthFlag = false;
      let heightFlag = false;
      for (let i = 0, l = POTS.length; i < l; ++i) {
        const curPot = POTS[i];
        if (!widthFlag && optionWidth >= curPot) {
          optionWidth = curPot;
          widthFlag = true;
        }
        if (!heightFlag && optionHeight >= curPot) {
          optionHeight = curPot;
          heightFlag = true;
        }

        if (widthFlag && heightFlag) {
          break;
        }
      }
    }

    const rects: Rect[] = [];
    const packCount = images.length;
    const doublePadding = padding * 2;
    for (let i = packCount - 1; i >= 0; i--) {
      const file = images[i];
      const image = file.image;
      if (
        !this.checkSizeLegality(image.width, image.height, optionWidth, optionHeight, doublePadding, option.allowRotate)
      ) {
        // 打包失败
        console.log("打包失败，", file.name, "单图尺寸超图集大小");
        return ErrorCode.PackError;
      }
      rects.push(new Rect(0, 0, image.width + doublePadding, image.height + doublePadding, file.name));
    }
    while (rects.length > 0) {
      this.pack.init(optionWidth, optionHeight, allowRotate);
      // 单次打包
      const atlasItem = {
        img: `./temp.png`,
        rects: <Rect[]>[],
        blob: null,
        width: 0,
        height: 0
      };
      const resArray = (atlasItem.rects = <Array<Rect>>this.pack.insert2(rects, method));
      let exportWidth = 0;
      let exportHeight = 0;
      const len = resArray.length;
      for (let i = 0; i < len; ++i) {
        const rect = resArray[i];
        const width = rect.x + rect.width;
        const height = rect.y + rect.height;
        exportWidth = Math.max(exportWidth, width);
        exportHeight = Math.max(exportHeight, height);

        if (padding > 0) {
          rect.x += padding;
          rect.y += padding;
          rect.width -= doublePadding;
          rect.height -= doublePadding;
        }
      }

      if (square) {
        exportWidth = exportHeight = Math.max(exportWidth, exportHeight);
      }

      if (pot) {
        let widthFlag = false;
        let heightFlag = false;
        for (let i = POTS.length - 1; i >= 0; --i) {
          const curPot = POTS[i];
          if (!widthFlag && exportWidth <= curPot) {
            exportWidth = curPot;
            widthFlag = true;
          }
          if (!heightFlag && exportHeight <= curPot) {
            exportHeight = curPot;
            heightFlag = true;
          }

          if (widthFlag && heightFlag) {
            break;
          }
        }
      }
      atlasItem.width = exportWidth;
      atlasItem.height = exportHeight;
      context.outPut.atlasItems.push(atlasItem);
    }
    return ErrorCode.Success;
  }

  /**
   * 检查单图尺寸是否合法
   * 如果单图的尺寸比整个图集的尺寸还大，需要直接返回错误
   * @param imgWidth - 单图宽度
   * @param imgHeight - 单图高度
   * @param atlasWidth - 图集宽度
   * @param atlasHeight - 图集高度
   * @param doublePadding - 2倍的padding
   * @param allowRotate - 是否支持旋转
   * @returns - 是否合法
   */
  checkSizeLegality(
    imgWidth: number,
    imgHeight: number,
    atlasWidth: number,
    atlasHeight: number,
    doublePadding: number,
    allowRotate: boolean
  ) {
    if (allowRotate) {
      // 支持旋转
      if (imgWidth + doublePadding > atlasWidth || imgHeight + doublePadding > atlasHeight) {
        return false;
      }
    } else {
      // 不支持旋转
      if (
        (imgWidth + doublePadding > atlasWidth || imgHeight + doublePadding > atlasHeight) &&
        (imgHeight + doublePadding > atlasWidth || imgWidth + doublePadding > atlasHeight)
      ) {
        return false;
      }
    }
    return true;
  }
}
