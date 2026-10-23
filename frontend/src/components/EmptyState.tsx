export function EmptyState() {
  return (
    <div className="empty-state" role="status">
      <p>No recommendation has been generated yet for NIFTY.</p>
      <p>This is not an error - the live paper session may not have produced a decision yet.</p>
    </div>
  );
}
