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
                           an omitted golden_only follows your GOLDEN_POLICY,
                           and under the "ask" policy Claude has to ask you
                           first; blocks until the job is RUNNING)
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
                             slurmx submit [opts] -- CMD  submit a job (pool from
                                                          GOLDEN_POLICY; --after JOBID
                                                          to chain, --golden-only or
                                                          --allow-main to pick a pool)
                             slurmx select-gpu --vram N   recommend a GPU
                             slurmx job-status ID         status of one job (alias: job)
                             slurmx wait ID               block until a job finishes
                             slurmx log ID                read a job's SLURM log
                             slurmx diagnose ID           classify a job failure
                             slurmx history               recent finished jobs (sacct)
                             slurmx cancel ID|--all       cancel jobs
                             slurmx config                edit config.py in a form
                                                          (--show prints it as text;
                                                           creates it on first run)
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
  1. Run `slurmx config` to check MAIL_USER, MAIL_TYPE, your GOLDEN_QOS list
     and GOLDEN_POLICY (set it to "ask" if you want to be asked, per job,
     whether a job may land on the preemptible main pool).
  2. Register the MCP server with Claude Code:
       claude mcp add slurmx \
         "$(pwd)/.venv/bin/python" "$(pwd)/server.py"
  3. Run `claude mcp list` to confirm it shows up.
  4. Start a session: `claude` — then ask "show me a cluster summary".
