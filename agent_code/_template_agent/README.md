# Typed agent template

Copy this whole directory to `agent_code/<your_agent_name>` and implement the
decision logic in `callbacks.py`. All engine-facing types are local to the
directory in `types.py`, so the copied agent remains self-contained.

When adding state to the callback context (for example `self.model`), declare
the corresponding attribute on `AgentContext` in `types.py`. The training
callbacks in `train.py` are intentionally no-ops and can be replaced as needed.
