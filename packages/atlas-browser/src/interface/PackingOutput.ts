import { Rect } from "@galacean/tools-atlas-algorithm";

/**
 * 打包的阶段产物
 */
export interface PackingOutput {
  msg?: string;
  code?: number;
  usage?: number;
  format?: number;
  atlasItems?: {
    img: string;
    rects: Rect[];
    blob: Blob;
    width: number;
    height: number;
  }[];
}
