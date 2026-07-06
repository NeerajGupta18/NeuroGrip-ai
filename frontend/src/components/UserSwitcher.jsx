import { useState } from "react";

export default function UserSwitcher({ users, activeUser, onSwitch }) {
  const [newUserInput, setNewUserInput] = useState("");

  const handleAddUser = (e) => {
    e.preventDefault();
    const trimmed = newUserInput.trim();
    if (!trimmed) return;
    onSwitch(trimmed);
    setNewUserInput("");
  };

  return (
    <div className="panel user-switcher">
      <div className="panel-label">
        <span className="eyebrow">USER PROFILE</span>
        <span className="panel-sub">calibration is per-user — switch to see it generalize</span>
      </div>

      <div className="user-chip-row">
        {users.map((u) => (
          <button
            key={u}
            className={`user-chip ${u === activeUser ? "user-chip-active" : ""}`}
            onClick={() => onSwitch(u)}
          >
            {u}
          </button>
        ))}
      </div>

      <form className="user-add-form" onSubmit={handleAddUser}>
        <input
          className="user-add-input mono"
          type="text"
          placeholder="new user id, e.g. patient_02"
          value={newUserInput}
          onChange={(e) => setNewUserInput(e.target.value)}
        />
        <button className="btn btn-ghost" type="submit">
          Add &amp; switch
        </button>
      </form>
    </div>
  );
}
