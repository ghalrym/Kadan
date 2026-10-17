import { useEffect, useRef } from "react";

// Decode and draw once: animated uploads remain still in avatars and the picker.
export default function StillImage({
  src,
  onError,
}: {
  src: string;
  onError?: () => void;
}) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const failed = useRef(onError);
  failed.current = onError;
  useEffect(() => {
    const image = new Image();
    let current = true;
    const context = canvas.current?.getContext("2d");
    context?.clearRect(0, 0, 256, 256);
    image.onload = () => {
      if (!current || !context) return;
      const side = Math.min(image.naturalWidth, image.naturalHeight);
      context.drawImage(
        image,
        (image.naturalWidth - side) / 2,
        (image.naturalHeight - side) / 2,
        side,
        side,
        0,
        0,
        256,
        256,
      );
    };
    image.onerror = () => {
      if (current) failed.current?.();
    };
    image.src = src;
    return () => {
      current = false;
      image.onload = null;
      image.onerror = null;
      image.src = "";
    };
  }, [src]);
  return <canvas ref={canvas} width={256} height={256} aria-hidden="true" />;
}
