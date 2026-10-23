import type { NotModeledStatus } from "../api/types";
import { isNotModeledStatus } from "../api/types";

export function Value({ value, suffix }: { value: string | number | null | undefined; suffix?: string }) {
  if (value === null || value === undefined || value === "") {
    return <span className="null-value">not available</span>;
  }
  return (
    <>
      {value}
      {suffix ?? ""}
    </>
  );
}

export function NotModeledOrValue({ value }: { value: NotModeledStatus | Record<string, unknown> | null }) {
  if (value === null) return <span className="null-value">not available</span>;
  if (isNotModeledStatus(value)) {
    return (
      <span className="not-modeled-value" title={value.detail}>
        Not modeled
      </span>
    );
  }
  return <span>{JSON.stringify(value)}</span>;
}
