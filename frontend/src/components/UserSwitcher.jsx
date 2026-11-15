import { useState } from "react";

export default function UserSwitcher({ users, activeUser, sources, sourceLabels, onSwitch }) {
  const [newUserInput, setNewUserInput] = useState("");
  const [source, setSource] = useState("");

  const handleAddUser = (e) => {
    e.preventDefault();
    const trimmed = newUserInput.trim().replace(/[^A-Za-z0-9_-]/g, "");
    if (!trimmed) return;
    onSwitch(trimmed, source || sources[0]?.key);
    setNewUserInput("");
  };

  return (
    <div className="panel user-switcher">
      <div className="panel-label">
        <span className="eyebrow">WEARER PROFILE</span>
        <span className="panel-sub">
          each profile replays one real person&rsquo;s recordings; the model has never seen that person
        </span>
      </div>

      <div className="user-chip-row">
        {users.map((u) => (
          <button
            key={u.user_id}
            className={`user-chip ${u.user_id === activeUser ? "user-chip-active" : ""}`}
            onClick={() => onSwitch(u.user_id)}
          >
            {u.user_id} · {sourceLabels[u.source] || u.source}
          </button>
        ))}
      </div>

      <form className="user-add-form" onSubmit={handleAddUser}>
        <input
          className="user-add-input mono"
          type="text"
          aria-label="New profile id"
          placeholder="new profile id, e.g. patient_02"
          maxLength={40}
          value={newUserInput}
          onChange={(e) => setNewUserInput(e.target.value)}
        />
        <select
          className="user-add-input mono"
          aria-label="Signal source"
          value={source}
          onChange={(e) => setSource(e.target.value)}
        >
          {sources.map((s) => (
            <option key={s.key} value={s.key}>
              signals from {s.label}
            </option>
          ))}
        </select>
        <button className="btn btn-ghost" type="submit">
          Add &amp; switch
        </button>
      </form>
    </div>
  );
}
