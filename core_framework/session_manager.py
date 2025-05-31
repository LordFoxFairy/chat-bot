# core_framework/session_manager.py
import time
from typing import Dict, Any, Optional, List
from collections import deque


class SessionManager:
    def __init__(self, default_max_history_turns: int = 5):
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.default_max_history_turns = default_max_history_turns
        self.correlation_to_session_map: Dict[str, str] = {}
        # print("SessionManager: Initialized with empty correlation_to_session_map.")

    def create_session(self, session_id: Optional[str] = None, initial_data: Optional[Dict[str, Any]] = None) -> str:
        if session_id is None:  # pragma: no cover
            session_id = f"session_{time.time_ns()}_{hash(str(time.perf_counter_ns())) % 10000}"

        if session_id not in self.sessions:
            base_data = {
                "id": session_id,
                "created_at_ns": time.time_ns(),
                "dialogue_history": deque(maxlen=self.default_max_history_turns * 2),  # type: ignore
                "current_pipeline_name": None,
                "current_correlation_id": None,
                "current_pipeline_status": "idle",
                "correlation_id_counter": 0,
                "last_activity_ns": time.time_ns(),
            }
            if initial_data:  # pragma: no cover
                base_data.update(initial_data)
            self.sessions[session_id] = base_data
            print(f"SessionManager: Created new session [{session_id}]")
        # else: # pragma: no cover
        # self.sessions[session_id]["last_activity_ns"] = time.time_ns()
        return session_id

    def get_session_data(self, session_id: str) -> Optional[Dict[str, Any]]:
        session = self.sessions.get(session_id)
        if session:
            session["last_activity_ns"] = time.time_ns()
        return session

    def update_session_data(self, session_id: str, key: str, value: Any) -> bool:
        session = self.get_session_data(session_id)
        if session:
            session[key] = value
            return True
        # print(f"SessionManager: Warning - Attempt to update non-existent session [{session_id}] for key '{key}'") # pragma: no cover
        return False

    def add_to_dialogue_history(self, session_id: str, entry: Dict[str, Any], max_history_turns: Optional[int] = None):
        session = self.get_session_data(session_id)
        if session:
            # print(f"SessionManager: Adding to history for session [{session_id}]: {str(entry)[:100]}...")
            session["dialogue_history"].append(entry)  # type: ignore
        # else: # pragma: no cover
        # print(f"SessionManager: Warning - Attempt to add history to non-existent session [{session_id}]")

    def get_dialogue_history(self, session_id: str) -> List[Dict[str, Any]]:
        session = self.get_session_data(session_id)
        history = list(session["dialogue_history"]) if session else []  # type: ignore
        # print(f"SessionManager: Retrieved history for session [{session_id}], length: {len(history)}")
        return history

    def end_session(self, session_id: str) -> bool:
        if session_id in self.sessions:
            # print(f"SessionManager: Ending session [{session_id}]")
            correlations_to_remove = [
                cid for cid, sid in self.correlation_to_session_map.items() if sid == session_id
            ]
            for cid in correlations_to_remove:  # pragma: no cover
                # print(f"SessionManager: Removing correlation map entry: {cid} -> {session_id}")
                del self.correlation_to_session_map[cid]
            del self.sessions[session_id]
            return True
        return False  # pragma: no cover

    def get_next_correlation_id(self, session_id: str) -> str:
        session = self.get_session_data(session_id)
        if not session:  # pragma: no cover
            self.create_session(session_id)
            session = self.get_session_data(session_id)
            if not session: return f"error_session_corr_{time.time_ns()}"

        counter = session["correlation_id_counter"]
        session["correlation_id_counter"] += 1
        correlation_id = f"{session_id}_corr_{counter}"

        self.correlation_to_session_map[correlation_id] = session_id
        print(f"SessionManager: Generated and mapped Correlation ID '{correlation_id}' to Session ID '{session_id}'")
        return correlation_id

    def set_external_correlation_id_for_session(self, session_id: str, correlation_id: str):
        if session_id not in self.sessions:  # pragma: no cover
            print(
                f"SessionManager: Warning - Attempt to set external Correlation ID '{correlation_id}' for non-existent session [{session_id}]. Creating session.")
            self.create_session(session_id)  # Create if not exists, essential for this to work

        self.correlation_to_session_map[correlation_id] = session_id
        print(f"SessionManager: Mapped external Correlation ID '{correlation_id}' to Session ID '{session_id}'")
        # Also update the session data with this as the current correlation ID
        self.update_session_data(session_id, "current_correlation_id", correlation_id)

    def get_session_id_for_correlation(self, correlation_id: str) -> Optional[str]:
        session_id = self.correlation_to_session_map.get(correlation_id)
        # if not session_id: # pragma: no cover
        # print(f"SessionManager: Correlation ID '{correlation_id}' not found in map. Current map: {self.correlation_to_session_map}")
        # else:
        # print(f"SessionManager: Found Session ID '{session_id}' for Correlation ID '{correlation_id}'")
        return session_id

    def cleanup_inactive_sessions(self, inactive_threshold_s: int = 3600):  # pragma: no cover
        current_time_ns = time.time_ns()
        inactive_threshold_ns = inactive_threshold_s * 1_000_000_000
        sessions_to_remove = [sid for sid, data in self.sessions.items() if
                              current_time_ns - data.get("last_activity_ns", 0) > inactive_threshold_ns]
        for session_id in sessions_to_remove: self.end_session(session_id)
