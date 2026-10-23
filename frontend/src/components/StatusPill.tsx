interface Props {
  kind: string;
  label?: string;
}

export function StatusPill({ kind, label }: Props) {
  return (
    <span className="status-pill" data-kind={kind}>
      {label ?? kind}
    </span>
  );
}
