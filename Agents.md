## Strict Design Rules for Notebook Creation

All Jupyter Notebooks created or modified in this repository must strictly adhere to the following design rules:

### 1 Succinct and Elegant Titles
* Notebook titles must be succinct, elegant, and highly professional.
* You may add a subtitle for clarity and precision.
* Treat the notebook title and subtitle as if the notebook was the starting point for a published academic paper.

### 2 Friendly and Context-Rich Introductions
* The intro section must be friendly and accessible.
* Provide deep context for readers, assuming no previous specialized background.
* Clearly state the goal (hypothesis) and the core motivation of the notebook.
* Push complex LaTeX and heavy mathematical rigor to the deeper parts of the notebook to maintain a welcoming entry point.

### 3 Detailed Cell-Level Documentation
* Every cell (both markdown and code) must have a clear title and a paragraph describing its methodology, implementation details, and what the code is doing.
* Include LaTeX where it helps provide clarity.

### 4 High Precision in Parameter Documentation
* Be exceptionally precise on all parameters.
* This includes documenting the model architecture, input/output dimensions, dataset formation/splits, and the training process.
* Precise parameter documentation ensures experiments can be easily modified and verified by external researchers.

### 5 Visual Examples and Computation Validation
* Provide concrete examples to show how computations are performed and how datasets are constructed.
* Use logs, inline printed outputs, images, and videos where appropriate. This helps readers instantly verify the methodology and results.

### 6 Consistent and Transparent Metrics
* Be clear about the selected testing metrics and how they relate to the results.
* The metrics must be highly consistent with the notebook's motivation and accompanied by an explanation of why they were selected and how they facilitate interpretation.
* Training, validation, and testing metrics must be consistent with each other.
* Include training details, wall-clock execution times, and step-by-step logs to track these metrics.
* Include a justification for the selected model architecture in the relevant section.
* Use plenty of charts to show the results visually.

### 7 Summary Contribution to README
* Each notebook contributes to a larger research project.
* After completing a notebook, include a complete summary in the main project `README.md` so that the results provide context for future experiments and can be easily consolidated into a formal research paper.
