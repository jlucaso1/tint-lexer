export type SyntaxClassName =
  | "plain"
  | "comment"
  | "string"
  | "number"
  | "keyword"
  | "type"
  | "function"
  | "constant"
  | "operator";

export interface SyntaxSpan {
  type: SyntaxClassName;
  start: number;
  end: number;
}

export interface HtmlFormatOptions {
  classPrefix?: string;
  pre?: boolean;
}

export interface AnsiFormatOptions {
  theme?: Partial<Record<SyntaxClassName, string>>;
}

export declare const CLASSES: readonly SyntaxClassName[];
export declare function escapeHtml(str: string): string;
export declare function formatHtml(source: string, spans: SyntaxSpan[], options?: HtmlFormatOptions): string;
export declare const ANSI_COLORS: Readonly<Record<SyntaxClassName, string>>;
export declare function formatAnsi(source: string, spans: SyntaxSpan[], theme?: Partial<Record<SyntaxClassName, string>>): string;

export declare function parse(code: string): Promise<SyntaxSpan[]>;
export declare function highlight(code: string): Promise<SyntaxSpan[]>;
export declare function parseSync(code: string): SyntaxSpan[];
export declare function highlightSync(code: string): SyntaxSpan[];
export declare function parseFlat(code: string): Promise<Uint32Array>;
export declare function parseFlatSync(code: string): Uint32Array;
export declare function forEachSpan(code: string, callback: (start: number, end: number, classId: number, className: SyntaxClassName) => void): Promise<void>;
export declare function forEachSpanSync(code: string, callback: (start: number, end: number, classId: number, className: SyntaxClassName) => void): void;
export declare function highlightToHtml(code: string, options?: HtmlFormatOptions): Promise<string>;
export declare function highlightToHtmlSync(code: string, options?: HtmlFormatOptions): string;
export declare function highlightAnsi(code: string, options?: AnsiFormatOptions): Promise<string>;
export declare function highlightAnsiSync(code: string, options?: AnsiFormatOptions): string;
export declare function tokenize(source: string): Uint32Array;

export interface HighlighterOptions {
  metadata?: any;
  bytes?: Uint8Array;
  preferCpu?: boolean;
}

export declare class Highlighter {
  static load(options?: HighlighterOptions): Promise<Highlighter>;
  static load(metadata: any, bytes?: Uint8Array): Promise<Highlighter>;
  static loadSync(metadata?: any, bytes?: Uint8Array): Highlighter;
  readonly isCpu: boolean;
  highlight(source: string): Promise<Array<{ start: number; end: number; class: SyntaxClassName; type: SyntaxClassName }>>;
  highlightSync(source: string): Array<{ start: number; end: number; class: SyntaxClassName; type: SyntaxClassName }>;
  parseFlat(source: string): Promise<Uint32Array>;
  parseFlatSync(source: string): Uint32Array;
  forEachSpan(source: string, callback: (start: number, end: number, classId: number, className: SyntaxClassName) => void): Promise<void>;
  forEachSpanSync(source: string, callback: (start: number, end: number, classId: number, className: SyntaxClassName) => void): void;
  highlightToHtml(source: string, options?: HtmlFormatOptions): Promise<string>;
  highlightToHtmlSync(source: string, options?: HtmlFormatOptions): string;
  highlightAnsi(source: string, options?: AnsiFormatOptions): Promise<string>;
  highlightAnsiSync(source: string, options?: AnsiFormatOptions): string;
  dispose(): void;
}
export declare function getHighlighter(options?: HighlighterOptions): Promise<Highlighter>;

export declare class WorkerHighlighter {
  constructor(worker: any);
  highlight(source: string): Promise<Array<{ start: number; end: number; class: SyntaxClassName; type: SyntaxClassName }>>;
  parseFlat(source: string): Promise<Uint32Array>;
  highlightToHtml(source: string, options?: HtmlFormatOptions): Promise<string>;
  highlightAnsi(source: string, options?: AnsiFormatOptions): Promise<string>;
  dispose(): void;
}

export declare function createWorkerHighlighter(workerUrl?: string | URL): WorkerHighlighter;

