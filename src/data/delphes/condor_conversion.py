# Stdlib packages
import logging
import os
import subprocess

# HEP packages
#import eos_utils as eos

################################


logger = logging.getLogger(__name__)

class LPCVanillaSubmitter:
    """
    A class for submitting jobs on the FNAL's LPC cluster using HTCondor, one job per file in a list of filepaths.
    All jobs for a given era are submitted to the same cluster.

    Parameters:
        :param era_filepaths: Map of eras-to-filepaths, with tuples for input and output filepaths.
        :type era_filepaths: dict[str, list[tuple[str, str]]]
        :param datatype: Type of data (\'MC\' or \'Data\') for era_filepaths.
        :type datatype: str
        :param queue: HTCondor queue to submit the job to. Defaults to "longlunch".
        :type queue: str, optional
        :param memory: Memory request for the job. Defaults to "10GB".
        :type memory: str, optional
    """

    def __init__(
        self,
        dataset_filepaths: list[list[str]], out_file: str, 
        queue="longlunch", memory="4GB"
    ):
        self.queue = queue
        self.memory = memory
        
        self.git_repo = (
            subprocess.Popen(["git", "rev-parse", "--show-toplevel"], stdout=subprocess.PIPE)
            .communicate()[0].rstrip().decode("utf-8")
        )
        self.current_dir = os.getcwd()
        self.base_name = ".condor_preprocess"
        self.datetime_extension = subprocess.getoutput("date +%Y%m%d_%H%M%S")
        self.condor_dir = os.path.join(self.current_dir, self.base_name, self.datetime_extension)
        self.inputs_dir = os.path.join(self.condor_dir, 'inputs')
        self.jobs_dir = os.path.join(self.condor_dir, 'jobs')
        os.makedirs(self.inputs_dir); os.makedirs(self.jobs_dir)

        self.job_files = []

        # Get proxy information (required in executable script for this method of running)
        try:
            stat, out = subprocess.getstatusoutput("voms-proxy-info -e --valid 5:00")
        except:
            logger.exception(
                "voms proxy not found or validity less that 5 hours:\n%s",
                out
            )
            raise
        try:
            stat, out = subprocess.getstatusoutput("voms-proxy-info -p")
            out = out.strip().split("\n")[-1]
        except:
            logger.exception(
                "Unable to voms proxy:\n%s",
                out
            )
            raise
        proxy = out

        base_name = f"h5conv_{self.datetime_extension}"
        job_file_executable = os.path.join(self.jobs_dir, f"{base_name}.sh")
        job_file_submit = os.path.join(self.jobs_dir, f"{base_name}.sub")
        job_file_out = os.path.join(self.jobs_dir, f"{base_name}.$(ClusterId).$(ProcId).out")
        job_file_err = os.path.join(self.jobs_dir, f"{base_name}.$(ClusterId).$(ProcId).err")
        job_file_log = os.path.join(self.jobs_dir, f"{base_name}.$(ClusterId).log")
        n_jobs = len(dataset_filepaths)
        job_out_file = '$1.'.join(out_file.rsplit('.', 1))
        srv_out_file = os.path.normpath('/srv/'+job_out_file)

        with open(job_file_executable, "w") as executable_file:
            # Shabang and x509 proxy
            executable_file.write("#!/bin/bash\n")
            executable_file.write(f"export X509_USER_PROXY={'/srv'+proxy[proxy.rfind('/'):]}\n")

            executable_file.write("echo \"Start of job $1\"\n")
            executable_file.write("echo \"-------------------------------------\"\n")

            # Setting up python environment
            executable_file.write("echo \"Pulling python to node\"\n")
            executable_file.write("wget https://www.python.org/ftp/python/3.12.6/Python-3.12.6.tar.xz\n")
            
            executable_file.write("echo \"Building python \"\n")
            executable_file.write("cd /srv\n")
            executable_file.write("tar -xvf Python-3.12.6.tar.xz\n")
            executable_file.write("cd /srv/Python-3.12.6\n")
            executable_file.write("./configure --prefix=/srv/python3.12\n")
            executable_file.write("make\n")
            executable_file.write("make install\n")
            executable_file.write("export PATH=\"/srv/python3.12/bin:$PATH\"\n")
            
            executable_file.write("echo \"Pulling git repo to node\"\n")
            executable_file.write("cd /srv\n")
            executable_file.write("git clone https://github.com/tcoulvert/SPAtop\n")
            executable_file.write("cd /srv/SPAtop\n")

            executable_file.write("echo \"Building python environment\"\n")
            executable_file.write("python3.12 -m venv /srv/spatop_venv\n")
            executable_file.write("source /srv/spatop_venv/bin/activate\n")
            executable_file.write("python3.12 -m pip install -r requirements.txt\n")
            executable_file.write("python3.12 -m pip install -e .\n")
            executable_file.write("echo \"-------------------------------------\"\n")
            
            # Running preprocessing code for era
            executable_file.write(f"echo \"Running conversion for $1\"\n")
            executable_file.write("echo \"-------------------------------------\"\n")
            
            for i, filepaths in enumerate(dataset_filepaths):
                executable_file.write(f"if [ $1 -eq {i} ]; then\n")
                executable_file.write(f"    python3.12 /srv/SPAtop/src/data/delphes/convert_to_h5.py {', '.join(filepaths)} --out-file {srv_out_file}\n")
                executable_file.write(f"    xrdcp -f {srv_out_file} {job_out_file}\n")
                executable_file.write("fi\n")
            os.system(f"chmod 775 {job_file_executable}")
            
            with open(job_file_submit, "w") as submit_file:
                submit_file.write(f"executable = {job_file_executable}\n")
                submit_file.write("arguments = $(ProcId)\n")
                submit_file.write(f"output = {job_file_out}\n")
                submit_file.write(f"error = {job_file_err}\n")
                submit_file.write(f"log = {job_file_log}\n")
                submit_file.write(f"request_memory = {self.memory}\n")
                submit_file.write("getenv = True\n")
                submit_file.write(f'+JobQueue = "{self.queue}"\n')
                submit_file.write(f"should_transfer_files = YES\n")
                submit_file.write(f"Transfer_Input_Files = {proxy}\n")
                submit_file.write(f"Transfer_Output_Files = \"\"\n")
                submit_file.write(f'when_to_transfer_output = ON_EXIT\n')

                submit_file.write('on_exit_remove = (ExitBySignal == False) && (ExitCode == 0)\n')
                submit_file.write('max_retries = 0\n')
                submit_file.write(f"queue {n_jobs}\n")
            self.job_files.append(job_file_submit)

    def update_git(self):
        def run_git_cmd(cmd):
            """Helper function to run Git commands and handle errors."""
            result = subprocess.run(cmd, cwd=self.git_repo, capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception(f"Command {' '.join(cmd)} failed:\n{result.stderr}")
            return result.stdout
            
        try:
            run_git_cmd(["git", "commit", "-a", "-m", f"commit before condor preprocess at {self.datetime_extension}"])
            run_git_cmd(["git", "push"])
        except Exception as e:
            logger.exception(e)
            if not 'branch is up to date' in str(e): 
                raise e

    
    def update_git(self):
        
        def run_git_cmd(cmd):
            """Helper function to run Git commands and handle errors."""
            result = subprocess.run(cmd, cwd=self.git_repo, capture_output=True, text=True)
            return result.returncode, result.stdout, result.stderr
            
        comm_code, comm_out, comm_err = run_git_cmd(["git", "commit", "-a", "-m", f"commit before condor preprocess at {self.datetime_extension}"])
        if comm_code == 0:
            push_code, push_out, push_err = run_git_cmd(["git", "push"])
            if push_code != 0: raise Exception(f'Git push failed:\n\nOut: {push_out}\n\nError: {push_err}')
        elif comm_code != 0 and 'branch is up to date' in comm_out: return
        else: raise Exception(f'Git commit failed:\n\nOut: {comm_out}\n\nError: {comm_err}')

    def submit(self):
        """
        A method to submit all the jobs in the jobs_dir to the cluster
        """
        # commit and push to gitrepo
        self.update_git()

        # submit jobs
        for jf in self.job_files:
            if self.current_dir.startswith("/eos"):
                # see https://batchdocs.web.cern.ch/troubleshooting/eos.html#no-eos-submission-allowed
                subprocess.run(["condor_submit", "-spool", jf])
            else:
                subprocess.run("condor_submit {}".format(jf), shell=True)
        return None

    
