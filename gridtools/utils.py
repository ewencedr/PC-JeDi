from typing import Mapping


def standard_job_array(
    job_name: str,
    work_dir: str,
    image_path: str,
    command: str,
    log_dir: str,
    n_gpus: int,
    n_cpus: int,
    time_hrs: int,
    mem_gb: int,
    script_save_path: str,
    rename_dict: Mapping,
    opt_dict: Mapping,
    double_dash: bool = True,
):

    # Calculate the total number of jobs to perform
    n_jobs = 1
    for key, vals in opt_dict.items():
        if not isinstance(vals, list):
            vals = [vals]
            opt_dict[key] = vals
        n_jobs *= len(vals)
    print(f"Generating gridsearch with {n_jobs} subjobs")

    # Creating the slurm submision file
    f = open(f"{script_save_path}/{job_name}.sh", "w", newline="\n", encoding="utf-8")
    f.write("#!/bin/sh\n\n")
    f.write(f"#SBATCH --cpus-per-task={n_cpus}\n")
    f.write(f"#SBATCH --mem={mem_gb}GB\n")
    f.write(f"#SBATCH --time={time_hrs}:00:00\n")
    f.write(f"#SBATCH --job-name={job_name}\n")
    f.write(f"#SBATCH --output={log_dir}/%A_%a.out\n")
    if n_gpus:
        f.write(f"#SBATCH --gpus={n_gpus}\n")
        f.write("#SBATCH --partition=shared-gpu,private-dpnc-gpu\n")
    else:
        f.write("#SBATCH --partition=shared-cpu,private-dpnc-cpu\n")

    # The job array setup using the number of jobs
    f.write(f"\n#SBATCH -a 0-{n_jobs-1}\n\n")

    # Creating the bash lists of the job arguments
    for (opt, vals) in opt_dict.items():
        f.write('{}=({}'.format(rename_dict[opt], vals[0]))
        for v in vals[1:]:
            f.write(' {}'.format(v))
        f.write(')\n')
    f.write("\n")

    # The command line arguments
    f.write('export XDG_RUNTIME_DIR=""\n')
    # f.write("module load GCC/9.3.0 Singularity/3.7.3-GCC-9.3.0-Go-1.14\n")

    # Creating the base singularity execution script
    f.write(f"cd {work_dir}\n")
    f.write("srun singularity exec --nv -B /srv,/home \\\n")
    f.write(f"   {image_path} \\\n")
    f.write(f"   {command} \\\n")

    # Now include the job array options using the bash lists
    run_tot = 1
    for opt, vals in opt_dict.items():
        if double_dash:
            f.write(f"       --{opt} ${{{rename_dict[opt]}")
        else:
            f.write(f"       {opt}=${{{rename_dict[opt]}")
        f.write(f"[`expr ${{SLURM_ARRAY_TASK_ID}} / {run_tot} % {len(vals)}`]")
        f.write("} \\\n")
        run_tot *= len(vals)
    f.close()
