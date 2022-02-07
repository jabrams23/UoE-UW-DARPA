---
#cover: cover.jpg
title: Program Summary
permalink: /summary/
---
## Introduction
This program is based on combining two components. The first is a conventional knowledge-based model of the evolution of the climate. The second is a machine learning component employing a spatial grid of many, relatively small, reservoir computers trained in parallel on observation-based state evolution data. This approach has been formulated by us through previous DARPA-funded research. We call our scheme Combined Hybrid Parallel Prediction (CHyPP, pronounced "chip").

In past work, we have successfully implemented CHyPP for weather forecasting. For this implementation, the knowledge-based component of the hybrid was a low-resolution, but realistic, fully 3-dimensional atmospheric global circulation model known as SPEEDY. SPEEDY was formulated for climate studies, and incorporates terrestrial geography, including representations of the Earth's continents, oceans, ice covered regions, and mountain ranges. This preliminary weather forecasting work has demonstrated the great potential of the CHyPP approach. In particular, it uses the data-driven machine learning component to effectively compensate for the inability of conventional, purely knowledge-based geophysical models to appropriately capture sub-grid scale dynamics, which are known to have crucial fundamental effects on the dynamics at the resolved scale. Such results lead us to anticipate that CHyPP may also facilitate large improvements in climate and tipping-point prediction relative to what is attainable using conventional, purely knowledge-based modeling.

## Tasks
### **Task I**: Extension of our current SPEEDY-based CHyPP implementation from weather (i.e., short-term atmospheric state prediction) to climate (i.e., long-term prediction of the statistical properties of atmospheric states).
### **Task II**: Development of new climate and tipping-point prediction concepts and methods through theoretical analysis and numerical exploration on appropriate "small" test models.
### **Task III**: Incorporation of those discoveries from Task II that are found to be appropriate into the global atmospheric model implemented in Task I, for testing, evaluation, modification, and further development of techniques and concepts.

## Key Issues
- Coupling the atmospheric model to a model of the ocean (part of Task I).
- Scalability to very large, high-resolution systems (part of Task III).
- Computational speed and accuracy of predictions (part of Task III).
- Methods for enhancing prediction system stability for long-term climate runs (part of Task II).
- Methods for enhancing prediction of non-stationary dynamical systems and tipping points (building on our previous DARPA-funded work) (part of Task II).
- Tipping-point concepts for large, spatially heterogeneous, complex systems (part of Task II).
- Capturing behavior near and across tipping-points (part of Task II).
- Potential for CHyPP proxies to be used for computational speed-up relative to conventional purely knowledge-based models at high resolution (part of Task II).
- Identification of high-value data opportunities (part of Task III).
