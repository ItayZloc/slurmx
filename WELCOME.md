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
                           (lists pending jobs by user, in order, when a ticket is full)
  submit_job               submit GPU/CPU jobs (auto-selects GPU by VRAM;
                           golden-only by default — golden_only=false to
                           allow the preemptible main-pool fallback;
                           blocks until the job is RUNNING)
  select_gpu               recommend a GPU for a VRAM requirement (advisory;
                           reports the non-golden pick, so it can differ
                           from what a default submit_job uses)
  get_job_status           one job's status as JSON, incl. the pending reason
  wait_for_job             block until a job reaches a terminal state
  read_job_log             read a job's SLURM log (output_dir must be the
                           exact directory the job logs to)
  diagnose_job             classify a finished job's failure + log tail
  cancel_jobs              cancel by ID, all, or pending-only
  job_history              recent jobs from sacct, finished ones included

  None of them raise on failure — they return it. Read what comes back.

CLI COMMANDS
  slurmx <subcommand>      umbrella, like `git` or `aws-cli`:
                             slurmx status                live colorized dashboard
                                                          (--once for a text snapshot;
                                                           -n N sets refresh seconds)
                             slurmx submit [opts] -- CMD  submit a job (golden-only;
                                                          --after JOBID to chain,
                                                          --allow-main for main pool)
                             slurmx select-gpu --vram N   recommend a GPU
                             slurmx job-status ID         status of one job (alias: job)
                             slurmx wait ID               block until a job finishes
                             slurmx log ID                read a job's SLURM log
                             slurmx diagnose ID           classify a job failure
                             slurmx history               recent finished jobs (sacct)
                             slurmx cancel ID|--all       cancel jobs
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
      "Cancel my pending jobs."

NEXT STEPS
  1. Verify config.py has your MAIL_USER and the right GOLDEN_QOS list.
  2. Register the MCP server with Claude Code:
       claude mcp add slurmx \
         "$(pwd)/.venv/bin/python" "$(pwd)/server.py"
  3. Run `claude mcp list` to confirm it shows up.
  4. Start a session: `claude` — then ask "show me a cluster summary".
