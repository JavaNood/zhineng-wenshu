import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

/** 轻量 Markdown 渲染：支持标题、加粗、列表、表格、代码块，保证专家结论可读。 */
export function Md({ text, className }: { text?: string; className?: string }) {
  if (!text) return null;
  return (
    <div className={`md-body ${className || ""}`}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          table: ({ children }) => (
            <div className="md-table-wrap"><table>{children}</table></div>
          ),
          code: ({ className, children, ...props }: any) => {
            const isBlock = /language-/.test(className || "");
            return isBlock ? (
              <pre className="md-code"><code {...props}>{children}</code></pre>
            ) : (
              <code className="md-inline-code" {...props}>{children}</code>
            );
          },
        }}
      >
        {text}
      </ReactMarkdown>
    </div>
  );
}
