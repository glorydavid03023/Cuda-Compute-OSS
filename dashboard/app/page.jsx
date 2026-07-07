import fs from "node:fs";
import path from "node:path";

function readJson(file, fallback) {
  try {
    return JSON.parse(fs.readFileSync(file, "utf8"));
  } catch {
    return fallback;
  }
}

function fmt(value, digits = 4) {
  if (value === null || value === undefined || value === "") return "-";
  if (typeof value === "number") return Number.isInteger(value) ? value : value.toFixed(digits);
  return String(value);
}

function stateClass(value) {
  return String(value || "").replace(/[^a-zA-Z0-9_-]/g, "_");
}

function PrLink({ item }) {
  if (!item.url) return <>#{item.pr || item.num}</>;
  return <a href={item.url}>#{item.pr || item.num}</a>;
}

function Stat({ label, value, tone = "" }) {
  return (
    <div className={`stat ${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function QueueTable({ queue }) {
  if (!queue.length) {
    return <div className="empty">No PRs are queued for the next RTX 5090 batch.</div>;
  }
  return (
    <div className="tableWrap">
      <table>
        <thead>
          <tr>
            <th>Pos</th>
            <th>PR</th>
            <th>Title</th>
            <th>Author</th>
            <th>State</th>
            <th>Head</th>
          </tr>
        </thead>
        <tbody>
          {queue.map((item) => (
            <tr key={`${item.pr}-${item.head_sha}`}>
              <td>{item.position}</td>
              <td><PrLink item={item} /></td>
              <td>{item.title}</td>
              <td>{item.author}</td>
              <td><span className={`pill ${stateClass(item.state)}`}>{item.state}</span></td>
              <td className="mono">{String(item.head_sha || "").slice(0, 12)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ResultsTable({ prs }) {
  if (!prs.length) {
    return <div className="empty">No verified GPU results have landed yet.</div>;
  }
  return (
    <div className="tableWrap">
      <table>
        <thead>
          <tr>
            <th>PR</th>
            <th>Track</th>
            <th>Verdict</th>
            <th>Accuracy</th>
            <th>Latency</th>
            <th>VRAM</th>
            <th>FLOP Ratio</th>
          </tr>
        </thead>
        <tbody>
          {prs.map((item) => (
            <tr key={`${item.num}-${item.label}-${item.track}`}>
              <td><PrLink item={item} /></td>
              <td>{item.track}</td>
              <td><span className={`pill eval_${stateClass(item.label)}`}>eval:{item.label}</span></td>
              <td>{fmt(item.accuracy)}</td>
              <td>{fmt(item.latency_s, 5)}s</td>
              <td>{fmt(item.peak_vram_mib, 1)} MiB</td>
              <td>{fmt(item.flop_ratio, 2)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function TrackGrid({ tracks }) {
  const entries = Object.entries(tracks || {});
  if (!entries.length) return <div className="empty">No frontier tracks yet.</div>;
  return (
    <div className="trackGrid">
      {entries.map(([name, track]) => (
        <div className="track" key={name}>
          <span>{name}</span>
          <strong>{fmt(track.frontier_score)}</strong>
          <em>accuracy floor {fmt(track.accuracy_floor, 2)}</em>
        </div>
      ))}
    </div>
  );
}

export default function Dashboard() {
  const queueData = readJson(
    path.join(process.cwd(), "data.json"),
    { queue: [], open_prs: [], gpu_policy: {}, updated: "" },
  );
  const resultsData = readJson(
    path.join(process.cwd(), "results.json"),
    { prs: [], status: { gpu: "RTX 5090", tracks: {} }, updated: "" },
  );
  const queue = queueData.queue || [];
  const open = queueData.open_prs || [];
  const results = resultsData.prs || [];
  const rejected = results.filter((item) => item.label === "REJECT").length;
  const admitted = results.filter((item) => ["BASELINE", "XS", "S", "M", "L", "XL"].includes(item.label)).length;

  return (
    <main>
      <header>
        <div>
          <p className="eyebrow">Cuda-Compute-OSS</p>
          <h1>PR evaluation control room</h1>
        </div>
        <div className="meta">
          <span>Queue updated: {queueData.updated || "pending"}</span>
          <span>Results updated: {resultsData.updated || "pending"}</span>
        </div>
      </header>

      <section className="stats">
        <Stat label="Queued for RTX 5090" value={queue.length} tone="queued" />
        <Stat label="Open PRs tracked" value={open.length} />
        <Stat label="Verified results" value={results.length} />
        <Stat label="Admitted" value={admitted} tone="good" />
        <Stat label="Rejected" value={rejected} tone="bad" />
      </section>

      <section>
        <div className="sectionHead">
          <h2>Sequential GPU Queue</h2>
          <p>{queueData.gpu_policy?.cadence || "Maintainer-controlled RTX 5090 windows."}</p>
        </div>
        <QueueTable queue={queue} />
      </section>

      <section>
        <div className="sectionHead">
          <h2>Verified Results</h2>
          <p>Final labels are written by Phase 3 after GPU or mock result processing.</p>
        </div>
        <ResultsTable prs={results} />
      </section>

      <section>
        <div className="sectionHead">
          <h2>Frontier Tracks</h2>
          <p>Current admitted score frontier by evaluation track.</p>
        </div>
        <TrackGrid tracks={resultsData.status?.tracks || {}} />
      </section>
    </main>
  );
}
