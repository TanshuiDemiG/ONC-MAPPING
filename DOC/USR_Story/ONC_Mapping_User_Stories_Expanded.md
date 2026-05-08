# Expanded User Stories — ONC Mapping Project

## 1. Field Research Scientist Story

**As a** field ecological researcher
**I want** to quickly identify and assess rock habitat quality across large regions using drone imagery and AI-based analysis
**So that** I can reduce manual survey time and improve the efficiency of identifying suitable pink-tailed worm-lizard habitats.

### Acceptance Criteria

* System identifies rock formations automatically from aerial imagery
* Habitat quality grading is generated visually
* Detection confidence scores are displayed
* Results can be exported for field validation
* Large datasets can be processed in batches

---

## 2. Conservation Manager Story

**As a** conservation manager at the ACT Office of Nature Conservation
**I want** habitat quality maps and statistical reports
**So that** I can make evidence-based conservation and land management decisions.

### Acceptance Criteria

* GIS-compatible outputs are generated
* High-quality habitats are highlighted
* Distribution statistics are included
* Reports support environmental planning
* Results are reproducible and documented

---

## 3. GIS Specialist Story

**As a** GIS technical specialist
**I want** automated ArcGIS-compatible workflows
**So that** drone imagery and habitat data can integrate directly into existing GIS systems.

### Acceptance Criteria

* Outputs support ArcGIS layers and shapefiles
* Scripts automate preprocessing workflows
* Spatial maps are generated correctly
* Coordinates remain geographically accurate
* Deployment scripts are documented

---

## 4. Drone Operator Story

**As a** drone operator
**I want** standardised image collection procedures
**So that** collected aerial imagery is suitable for machine learning analysis.

### Acceptance Criteria

* Flight altitude recommendations are defined
* Image overlap standards are documented
* Weather limitations are considered
* Image quality validation exists
* Data collection workflows are repeatable

---

## 5. Machine Learning Developer Story

**As a** machine learning developer
**I want** to train and compare multiple detection models
**So that** the team can identify the most accurate and efficient solution for habitat analysis.

### Acceptance Criteria

* Multiple YOLO models can be trained
* Model metrics are compared systematically
* Detection confidence is evaluated
* False positives and false negatives are analysed
* Training results are reproducible

---

## 6. Model Comparison & Validation Story

**As a** project developer
**I want** to compare different training configurations and preprocessing techniques
**So that** model performance can be optimised for real-world deployment.

### Acceptance Criteria

* Different image sizes (640 vs 512) are tested
* Confidence thresholds are compared
* Performance metrics are logged
* Results are reviewed collaboratively
* Best-performing configuration is selected

---

## 7. Code Review & Collaboration Story

**As a** software development team member
**I want** structured code reviews and collaborative debugging
**So that** the project remains maintainable, reproducible, and technically reliable.

### Acceptance Criteria

* Team members review each other's scripts
* Repository commits are documented
* Bugs are identified collaboratively
* Training pipelines are reproducible
* Shared coding standards are maintained

---

## 8. Data Annotation Story

**As a** data annotation team member
**I want** to label rock habitats accurately using Roboflow
**So that** the machine learning models receive high-quality training data.

### Acceptance Criteria

* Bounding boxes are consistent
* Labels follow agreed standards
* Multiple annotation passes are completed
* Dataset versions are tracked
* Label quality is reviewed internally

---

## 9. ArcGIS Deployment Story

**As a** GIS deployment developer
**I want** the AI outputs integrated into ArcGIS workflows
**So that** conservation staff can visualise habitat distribution spatially.

### Acceptance Criteria

* Detection outputs are imported into ArcGIS
* Spatial maps render correctly
* Distribution layers are generated
* Scripts automate map generation
* GIS exports are compatible with stakeholder systems

---

## 10. CUDA GPU Acceleration Story

**As a** machine learning developer
**I want** CUDA-enabled GPU acceleration during training and inference
**So that** model experimentation and deployment become significantly faster.

### Acceptance Criteria

* CUDA GPUs are detected automatically
* GPU training reduces epoch runtime
* Inference speed improves
* Training logs confirm CUDA usage
* CPU fallback exists if GPU unavailable

---

## 11. EXE Deployment Story

**As a** non-technical conservation staff member
**I want** to run the detection system through a simple executable application
**So that** I can use the software without programming knowledge.

### Acceptance Criteria

* Software launches from a standalone `.exe`
* Users can upload drone images directly
* Detection outputs display automatically
* Results export locally
* No Python installation is required

---

## 12. Student Team Collaboration Story

**As a** student project team member
**I want** clearly documented sprint goals, responsibilities, and meeting records
**So that** collaboration and accountability remain effective throughout the project lifecycle.

### Acceptance Criteria

* Meeting minutes are recorded consistently
* Sprint tasks are assigned clearly
* Deliverables are tracked
* Risks are reviewed regularly
* Team progress is documented

---

## 13. Client Communication Story

**As a** project stakeholder
**I want** regular progress updates and prototype demonstrations
**So that** the project remains aligned with conservation requirements and stakeholder expectations.

### Acceptance Criteria

* Sprint updates are communicated clearly
* Prototype demos are presented
* Client feedback is documented
* Technical limitations are explained
* Project goals remain aligned with stakeholder needs

---

## 14. Habitat Mapping Story

**As a** conservation researcher
**I want** rock density and vegetation distribution maps generated automatically
**So that** habitat suitability can be analysed efficiently over large regions.

### Acceptance Criteria

* Distribution heatmaps are generated
* Rock counts are calculated
* Vegetation overlays are supported
* Outputs are geographically accurate
* Maps are exportable

---

## 15. Risk Management Story

**As a** project team member
**I want** technical and project risks reviewed regularly
**So that** issues affecting model performance, deployment, or collaboration can be mitigated early.

### Acceptance Criteria

* Risks are documented
* Technical blockers are tracked
* Mitigation plans are proposed
* Dataset limitations are reviewed
* Deployment risks are discussed during sprint reviews

---

## 16. Real-Time Demonstration Story

**As a** project presenter or stakeholder
**I want** live demonstrations of the detection system
**So that** the functionality and impact of the project can be validated visually.

### Acceptance Criteria

* Demo images process successfully
* Detection overlays appear correctly
* Outputs generate within acceptable runtime
* Confidence scores display live
* Prototype remains stable during presentations
