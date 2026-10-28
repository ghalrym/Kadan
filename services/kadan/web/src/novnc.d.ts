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
    sendKey(keysym: number, code: string | null, down?: boolean): void;
    disconnect(): void;
    focus(): void;
  }
}
