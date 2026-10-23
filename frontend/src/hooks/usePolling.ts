import { useEffect, useRef } from "react";

export function usePolling(callback: () => void, intervalMs: number): void {
  const savedCallback = useRef(callback);
  savedCallback.current = callback;

  useEffect(() => {
    savedCallback.current();

    const id = window.setInterval(() => {
      if (!document.hidden) savedCallback.current();
    }, intervalMs);

    const onVisibilityChange = () => {
      if (!document.hidden) savedCallback.current();
    };
    document.addEventListener("visibilitychange", onVisibilityChange);

    return () => {
      window.clearInterval(id);
      document.removeEventListener("visibilitychange", onVisibilityChange);
    };
  }, [intervalMs]);
}
