# Ohio Supercomputer Instructions
This file provides comprehensive instructions on how to run an experiment on the Ohio Supercomputer (OSC). It assumes you already have access.

### Starting the job
1. Sign into your OSC OnDemand account at [https://www.example.com](https://ondemand.osc.edu/).
2. Head to **"My Interactive Sessions"** and scroll down on the left under **"Interactive Apps"**. Select **"Code Server"**.
3. Select the above and start a job. The cluster type does not matter. Same thing with the working directory, you can leave it blank.
4. When you create the job, wait for it to start and click **"Connect to VS Code"**.

### Getting the code
You'll now need to set up your Python environment and get the source code for this repository. I used Miniconda but you can use whatever. Also note you can drag and drop files into the VS Code file system from your computer.
1. In the VS Code terminal, clone this repository through `git clone https://github.com/ziesski/GGR.git`
2. Download your desired conda installer and drag it into the VS Code file system. Download the Linux x86 version. You can get Anaconda/Miniconda at [https://www.anaconda.com/download/success](https://www.anaconda.com/download/success).
3. In the VS Code terminal, execute this via `./[name of installer].sh` *If you have a `Permission Denied` error you need to give execute permissions to the executable via `chmod +rwx [name of installer].sh`*

### Running the code
Now that you have conda installed along with the code, you can directly follow the instructions in the **README**. Make sure to do it in the VS Code terminal and remember you can drag and drop files. Note that you might also have an issue with video permissions. If so, repeat the same `chmod` step for the names of the videos.
