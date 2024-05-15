import { PackingContext } from "../interface/PackingContext";
import { ErrorCode } from "../enum/ErrorCode";
import { OptHandler } from "./OptHandler";
import { PackingItem } from "../interface/PackingItem";

export class OptLoadImage extends OptHandler {
  parse(context: PackingContext): Promise<ErrorCode> {
    return new Promise<ErrorCode>((resolve, reject) => {
      const { images } = context;
      // 第一步：加载所有的图片
      const imagesLength = images.length;
      if (imagesLength <= 0) {
        reject(ErrorCode.NoImage);
        return;
      }
      const promiseArray: Promise<HTMLImageElement>[] = [];
      for (let i = 0; i < imagesLength; i++) {
        const image = images[i];
        let promise: Promise<HTMLImageElement>;
        const imgSrc = image.src;
        if (!image.image) {
          if (imgSrc instanceof ArrayBuffer) {
            promise = this._loadImageFromBuffer(image);
          } else {
            promise = this._loadImageFromURL(image);
          }
          promiseArray.push(promise);
        }
      }
      Promise.all(promiseArray)
        .then(() => {
          resolve(ErrorCode.Success);
        })
        .catch((error: Error) => {
          console.error("OptLoadImage:Error", JSON.stringify(error));
          reject(ErrorCode.ImageLoadError);
        });
    });
  }

  private _loadImageFromBuffer(item: PackingItem): Promise<HTMLImageElement> {
    return new Promise((resolve, reject) => {
      const buffer = item.src as ArrayBuffer;
      const blob = new window.Blob([buffer], { type: item.type });
      const img = new Image();
      img.onerror = function () {
        reject(new Error("Failed to load image buffer"));
      };
      img.onload = function () {
        img.onload = null;
        img.onerror = null;
        img.onabort = null;
        item.image = img;
        resolve(img);
      };
      img.crossOrigin = "anonymous";
      img.src = URL.createObjectURL(blob);
    });
  }

  private _loadImageFromURL(item: PackingItem): Promise<HTMLImageElement> {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onerror = function () {
        reject(new Error("Failed to load image buffer"));
      };
      img.onload = function () {
        img.onload = null;
        img.onerror = null;
        img.onabort = null;
        item.image = img;
        resolve(img);
      };
      img.crossOrigin = "anonymous";
      img.src = item.src as string;
    });
  }
}
