// @novnc/novnc ships JavaScript only; this covers the RFB surface Kadan uses.
declare module "@novnc/novnc" {
  export default class RFB extends EventTarget {
    constructor(target: HTMLElement, url: string, options?: { wsProtocols?: string[] });
    viewOnly: boolean;
    scaleViewport: boolean;
    clipViewport: boolean;
    resizeSession: boolean;
    focusOnClick: boolean;
    background: string;
    qualityLevel: number;
    disconnect(): void;
    focus(): void;
  }
}
