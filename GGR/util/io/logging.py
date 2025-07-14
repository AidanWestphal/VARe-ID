import logging
import os
import select
import sys
import subprocess


class ConsoleFormatter(logging.Formatter):
    '''
    A custom formatter for console output.
    Critical messages have a header of [CRITICAL].
    Error messages have a header of [ERROR].
    All other messages have no header.
    '''

    FORMAT_INFO = "%(message)s"
    FORMAT_ERROR = "[ERROR] %(message)s"
    FORMAT_CRITICAL = "[CRITICAL] %(message)s"

    def format(self, record):
        if record.levelno >= logging.CRITICAL:
            self._fmt = self.FORMAT_CRITICAL
        elif record.levelno >= logging.ERROR:
            self._fmt = self.FORMAT_ERROR
        else:
            self._fmt = self.FORMAT_INFO
        
        # CALL PARENT FORMAT
        return super().format(record)


def setup_logging(logfile):
    '''
    Configurates a logger with the the proper formatting for the pipeline.
    '''
    logger = logging.getLogger("smk_pipeline")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = ConsoleFormatter()

    # LOGFILE
    file_handler = logging.FileHandler(logfile)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO) 
    logger.addHandler(file_handler)

    # STDOUT
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO) 
    logger.addHandler(console_handler)

    return logger


def log_subprocess(command, logger):
    '''
    Executes a subprocess with real-time logging. Output will be logged to the console 
    and logging file. 
    '''
    try:
        process = subprocess.Popen(command,
                                 shell=True,
                                 text=True,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE,
                                 universal_newlines=True,
                                 bufsize=1)
        
        # MAPPINGS FROM FD TO IMPORTANT INFO
        fd_map = {
            process.stdout.fileno(): {"stream": process.stdout, "level": logger.info},
            process.stderr.fileno(): {"stream": process.stderr, "level": logger.error}
        }
        active_fds = list(fd_map.keys())

        while active_fds or process.poll() is None: 
            # USE SELECT TO PICK FIRST STREAM WITH READABLE DATA
            ready_fds, _, _ = select.select(active_fds, [], [], 0.1)
            
            # IF THE PROCESS IS STILL RUNNING AND OUR FDS ARE EMPTY, SKIP
            if not ready_fds:
                continue

            # LOOP OVER READY FDS
            for fd in ready_fds:
                stream_info = fd_map[fd]
                # Strip after checking the line, we don't print EOF but do print deliberate whitespace
                line = stream_info["stream"].readline()
                if line:
                    line = line.rstrip() 
                    stream_info["level"](line)
                else:
                    # IF THE LINE WAS EMPTY STRING WE ARE AT EOF
                    active_fds.remove(fd)

            # FLUSH HANDLERS        
            for handler in logger.handlers:
                handler.flush()

        if process: 
            process.wait() 
            
        if process.returncode != 0:
            logger.critical(f"Subprocess failed with exit code {process.returncode}")

    except Exception as e:
        logger.critical(f"An unexpected error occurred: {e}")