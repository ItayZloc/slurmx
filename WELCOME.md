========================================================================
 slurmx is ready.
========================================================================

WHAT IT DOES
  Lets Claude Code (and you) submit, monitor, and manage SLURM GPU jobs
  through a small set of MCP tools and CLI commands. Auto-picks the
  smallest GPU that fits your VRAM budget, prefers golden tickets, and
  falls back to cluster-wide.

MCP TOOLS (invoked by Claude in chat)
  cluster_summary          your jobs + golden + cluster-wide GPU view
                           (lists the waiting queue when a golden ticket is full)
  submit_job               submit GPU/CPU jobs (auto-selects GPU by VRAM;
                           golden_only=true forces a preemption-immune slot)
  select_gpu               recommend a GPU for a VRAM requirement
  get_job_status           detailed status of a specific job
  wait_for_job             block until a job finishes
  read_job_log             read a job's SLURM log
  diagnose_job             classify failures (OOM/timeout/missing module/etc)
  cancel_jobs              cancel by ID, all, or pending-only
  job_history              recent finished jobs from sacct
  launch_remote_session    run `claude remote-control` as a SLURM job;
                           returns the claude.ai/code session URL inline

CLI COMMANDS
  slurmx <subcommand>      umbrella, like `git` or `aws-cli`:
                             slurmx status                one-shot dashboard
                             slurmx submit [opts] -- CMD  submit a job
                             slurmx remote-session        launch claude RC
                             slurmx setup                 = ./setup.sh
                             slurmx update                = ./update.sh
                             slurmx --help                list subcommands
                             slurmx <cmd> --help          per-subcommand help

HOW TO USE WITH AGENTS
  Once the MCP server is registered (step 4 in the README), any Claude
  Code session in this terminal can call the tools by name. Phrase
  requests naturally — the tool docstrings tell the agent what to ask:
      "Check GPU availability."
      "Submit a training job that needs 48GB of VRAM."
      "Diagnose job 12345."
      "Launch a remote-control session on CPU for 1 day."
  For launch_remote_session, the agent will ALWAYS ask you for hardware
  (cpu/gpu), time limit (1/2/3/7 days), and permission_mode
  (default/acceptEdits/plan) before submitting. If you pick gpu, it
  also asks for gpu_type (gtx_1080/rtx_2080/rtx_3090/rtx_4090/rtx_6000/
  rtx_pro_6000) and warns about idle-GPU cancellation. After submit,
  the tool waits up to 90s for the session URL to appear in the SLURM
  log and returns it inline (no need for a follow-up read_job_log).

NEXT STEPS
  1. Verify config.py has your MAIL_USER and the right GOLDEN_QOS list.
  2. Register the MCP server with Claude Code:
       claude mcp add slurmx \
         "$(pwd)/.venv/bin/python" "$(pwd)/server.py"
  3. Run `claude mcp list` to confirm it shows up.
  4. Start a session: `claude` — then ask "show me a cluster summary".

ONLY IF YOU PLAN TO USE launch_remote_session
  Remote Control needs a full-scope OAuth token, not the long-lived
  inference-only kind that `claude setup-token` writes. Run once:
       claude auth login                       # fresh install
       claude auth logout && claude auth login # already on setup-token
  ~/.claude/ is NFS-shared, so the credential is available to every
  compute node automatically. Symptom that you're on the wrong token:
  "Remote Control requires a full-scope login token" in the SLURM log.
