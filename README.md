# Video-Based Animal Re-Identification from Multiview Spatio-Temporal Track Clustering

This work is a modular software pipeline and end-to-end workflow for video-based animal re-identification that clusters multiview spatio-temporal tracks to assign consistent individual IDs with minimal human review. From raw video, we detect and track animals, score and select informative left/right views, compute embeddings, cluster annotations/embeddings by viewpoint, and then link clusters across time and disparate views using spatio-temporal track continuity plus automated consistency checks to resolve ambiguities; preliminary experiments show the approach can reach near-perfect identification accuracy with very little manual verification. This workflow is designed to be generalizable across different species. Currently, the trained models support Grevy's and Plains Zebras but it will be expanded to work with variety of other animal species.

### Tags: 
- Software
- CI4AI
- Animal-Ecology

---

## Definitions of key terms and concepts


* **Animal Re-Identification (re-id)**: The process of determining if an animal has been seen before by matching it against a database of images with known identity labels. The paper addresses this problem in the context of long video sequences.
* **Multiview Spatio-Temporal Track Clustering**: A novel framework introduced for animal re-identification. It works by clustering tracked animal detections from different viewpoints (multiview) and across time (spatio-temporal) to correctly identify individuals.
* **Identifiable Annotation (IA)**: An annotation, or detected animal image, that contains sufficient distinguishing information for reliable individual identification. For Grévy's zebras, an IA must show both the hip and chevron patterns on either the left or right side.
* **Human-in-the-loop**: The involvement of human decisions to confirm animal identities when the automated system is uncertain or to correct algorithmic errors.





---

# Tutorials


### High-level Introduction
### Prerequisites (e.g. software requirements, configurations)
### A sequence of steps that guide users through accomplishing a goal
### Visual aids such as screenshots or GIFs, if necessary, to clarify complex steps
### End results, showcasing the outcome of following the tutorial.



---

# How-To Guides

### Problem Description 
- Problem description: A brief overview of the task or issue at hand.

### Instructions
- Step-by-step instructions on how to complete the task. [Let us include how to run entire pipeline and also separate components guiding detailed steps]

### Variations and Advanced Tips
- Potential variations or advanced tips to enhance the process.

### Troubleshooting
- Troubleshooting advice for common pitfalls.

### Code snippets, commands or configuration examples
- Relevant code snippets, commands, or configuration examples.


---

# Explanation

### High-level Overview
- High-level overview of core concepts and principles.

### Explanation of Working of the System
- Explanations of how the system works and why certain design patterns or approaches were chosen.

### Visualizations
- Diagrams, flowcharts, or illustrations to visualize key concepts.

### Background Information
- Background information to help users grasp the context of the project’s design and architecture.

### Readings and Resources
- Suggested readings or resources for further exploration.

---



### License
- MIT [![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---
  
## References

### Links to related resources (libraries, tools, etc.) or external documentation
* [YOLOv10](https://github.com/THU-MIG/yolov10)
* [BioCLIP](https://github.com/Imageomics/bioclip)
* [MiewID](https://github.com/WildMeOrg/wbia-plugin-miew-id)
* [LCA](https://github.com/WildMeOrg/lca)

---
   
## Acknowledgements

* **National Science Foundation (NSF)** funded AI institute for Intelligent Cyberinfrastructure with Computational Learning in the Environment (ICICLE) (OAC 2112606).
* **Imageomics Institute (A New Frontier of Biological Information Powered by Knowledge-Guided Machine Learning)** is funded by the US National Science Foundation's Harnessing the Data Revolution (HDR) program under Award (OAC 2118240).
* Support from **Rensselaer Polytechnic Institute (RPI)**.
* Support from **Finnish Cultural Foundation**.
* Resources from **Ohio Supercomputer Center** made it possible to train and test algorithmic components.

