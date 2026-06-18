import { motion } from "framer-motion";

// iMessage-style three-dot "typing" bubble. Styled to match the assistant
// chat bubble (border + ide-card background) so it reads as a pending reply.
export function TypingIndicator() {
  return (
    <div className="flex w-full justify-start">
      <div className="max-w-[82%] rounded-2xl border border-ide-border bg-ide-card px-3.5 py-2.5">
        <div className="flex items-center gap-1">
          {[0, 1, 2].map((i) => (
            <motion.span
              key={i}
              className="block h-1.5 w-1.5 rounded-full bg-ide-textDim"
              animate={{ opacity: [0.3, 1, 0.3], y: [0, -2, 0] }}
              transition={{
                duration: 1,
                repeat: Infinity,
                ease: "easeInOut",
                delay: i * 0.15,
              }}
            />
          ))}
        </div>
      </div>
    </div>
  );
}
