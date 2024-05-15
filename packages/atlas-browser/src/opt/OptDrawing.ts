import { ErrorCode } from "../enum/ErrorCode";
import { PackingContext } from "../interface/PackingContext";
import { PackingItem } from "../interface/PackingItem";
import { OptHandler } from "./OptHandler";

export class OptDrawing extends OptHandler {
  private canvas: OffscreenCanvas | HTMLCanvasElement;
  private context2D: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D;

  async parse(context: PackingContext): Promise<ErrorCode> {
    const { atlasItems } = context.outPut;
    // 生成多张图集
    for (let i = 0; i < atlasItems.length; i++) {
      const atlasItem = atlasItems[i];
      this.initCanvas(atlasItem.width, atlasItem.height);
      const { context2D } = this;
      const { images } = context;
      const { padding } = context.option;
      const { rects } = atlasItem;
      for (let i = 0, l = rects.length; i < l; ++i) {
        const rect = rects[i];
        const name = rect.name;
        const image = this.getImage(images, name);
        if (image) {
          const { width: imgW, height: imgH } = image;
          const { x: sX, y: sY } = rect;
          context2D.drawImage(image, 0, 0, imgW, imgH, sX, sY, imgW, imgH);

          // Handle padding pixels.
          if (padding > 0) {
            try {
              for (let i = 1; i <= padding; ++i) {
                // Set top and bottom.
                const topY = sY - i;
                const bottomY = sY + imgH + i - 1;
                context2D.drawImage(image, 0, 0, imgW, 1, sX, topY, imgW, 1);
                context2D.drawImage(image, 0, imgH - 1, imgW, 1, sX, bottomY, imgW, 1);

                // Set left and right.
                const leftX = sX - i;
                const rightX = sX + imgW + i - 1;
                context2D.drawImage(image, 0, 0, 1, imgH, leftX, sY, 1, imgH);
                context2D.drawImage(image, imgW - 1, 0, 1, imgH, rightX, sY, 1, imgH);
              }
            } catch (error) {
              console.error(JSON.stringify(error));
            }
          }
        }
      }

      atlasItem.blob = await this.canvasToBlob();
    }
    return ErrorCode.Success;
  }

  canvasToBlob(): Promise<Blob> {
    return new Promise((resolve, reject) => {
      if (this.canvas instanceof HTMLCanvasElement) {
        this.canvas.toBlob(
          (blob) => {
            resolve(blob);
          },
          "image/png",
          1
        );
      } else {
        this.canvas
          .convertToBlob({ type: "image/png", quality: 1 })
          .then((blob) => {
            resolve(blob);
          })
          .catch((reason: any) => {
            console.error(JSON.stringify(reason));
            reject(ErrorCode.OffscreenBlobError);
          });
      }
    });
  }

  getImage(images: PackingItem[], name: string): HTMLImageElement {
    for (let i = images.length - 1; i >= 0; i--) {
      const image = images[i];
      if (image.name === name) {
        return image.image;
      }
    }
    return null;
  }

  initCanvas(exportWidth: number, exportHeight: number) {
    if (!this.canvas) {
      try {
        this.canvas = new OffscreenCanvas(exportWidth, exportHeight);
      } catch (error) {
        this.canvas = document.createElement("canvas");
        this.canvas.width = exportWidth;
        this.canvas.height = exportHeight;
      }
      this.context2D = this.canvas.getContext("2d", { willReadFrequently: true }) as
        | CanvasRenderingContext2D
        | OffscreenCanvasRenderingContext2D;
    } else {
      if (this.canvas.width !== exportWidth || this.canvas.height !== exportHeight) {
        this.canvas.width = exportWidth;
        this.canvas.height = exportHeight;
      } else {
        this.context2D.clearRect(0, 0, exportWidth, exportHeight);
      }
    }
  }
}
