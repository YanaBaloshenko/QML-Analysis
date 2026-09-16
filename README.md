# QML-Analysis

This is a part of my Master's Thesis. Project created to evaluate QML algorithms performance in comparison to their classical counterparts in an anomaly-based classification problem.

## Technical details

QML algorithms: VQC, QSVC, PegasosQSVC. ML algorithms: SVC, Pegasos.
Used IBM Qiskit Python library.
Experiments conducted on both simulation (StateVector) and real hardware (20-qubit IQM Garnet Quantum Processor) environments.
Specifically prepared NSL-KDD dataset was used.

## Results

QML algorithms showed better accuracy (92%), however they faced certain technological barriers, such as transpilation and middleware problems. Cost of implementing QML stays too high to use it in a production environment.
