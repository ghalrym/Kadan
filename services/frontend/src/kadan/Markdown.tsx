import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { remarkMentions } from "./mentions";

export const markdownPlugins = [remarkGfm];

export default function Markdown({ children, mentions }: { children: string; mentions?: Set<string> }) {
  return (
    <div className="markdown">
      <ReactMarkdown
        remarkPlugins={mentions?.size ? [...markdownPlugins, remarkMentions(mentions)] : markdownPlugins}
        components={{
          img: ({ alt }) => (
            <span className="field-hint">[Image: {alt || "not loaded"}]</span>
          ),
          table: ({ children }) => (
            <div className="markdown-table">
              <table>{children}</table>
            </div>
          ),
          a: ({ children, href }) =>
            href?.startsWith("#work/") ? (
              <a href={href}>{children}</a>
            ) : (
              <a href={href} target="_blank" rel="noreferrer">
                {children}
              </a>
            ),
        }}
      >
        {children}
      </ReactMarkdown>
    </div>
  );
}
