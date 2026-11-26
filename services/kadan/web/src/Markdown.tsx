import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

export const markdownPlugins = [remarkGfm];

export default function Markdown({ children }: { children: string }) {
  return (
    <div className="markdown">
      <ReactMarkdown
        remarkPlugins={markdownPlugins}
        components={{
          img: ({ alt }) => (
            <span className="field-hint">[Image: {alt || "not loaded"}]</span>
          ),
          table: ({ children }) => (
            <div className="markdown-table">
              <table>{children}</table>
            </div>
          ),
          a: ({ children, href }) => (
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
