export const STALE_THRESHOLD_MS = 15 * 60 * 1000;

export function elapsedMs(fromIso: string, nowMs: number = Date.now()): number {
  return nowMs - new Date(fromIso).getTime();
}

export function isStale(decisionTimestampIso: string, nowMs: number = Date.now()): boolean {
  return elapsedMs(decisionTimestampIso, nowMs) > STALE_THRESHOLD_MS;
}

export function formatElapsed(ms: number): string {
  if (!Number.isFinite(ms)) return "unknown";
  if (ms < 0) return "just now";
  const totalSeconds = Math.floor(ms / 1000);
  if (totalSeconds < 60) return `${totalSeconds}s ago`;
  const totalMinutes = Math.floor(totalSeconds / 60);
  if (totalMinutes < 60) return `${totalMinutes}m ago`;
  const totalHours = Math.floor(totalMinutes / 60);
  const remMinutes = totalMinutes % 60;
  if (totalHours < 24) return `${totalHours}h ${remMinutes}m ago`;
  const totalDays = Math.floor(totalHours / 24);
  return `${totalDays}d ago`;
}

export function formatTimestamp(iso: string | null): string {
  if (!iso) return "—";
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}
