---
layout: post
title: Month 1 Milestone Progress Report
category: reports
image-folder: 2022-01-13-ACTM-UMD-progress-report1-media
---
{% include base.html %}

# ACTM University of Maryland

## 1. Identification of Hybrid Model

We plan to develop a hybrid physics-based/machine-learning-based model
for prediction of climate change evolution and tipping point
prediction.

The physics-based component will be the publicly available Simplified
Parameterizations, primativE-Equation Dynamics (SPEEDY) model code,
which, although reduced resolution, incorporates relevant physics and
realistic terrestrial geography (e.g. mountain ranges, ice covered
regions, oceans, etc.), and is three dimensional, employing a grid in
latitude, longitude, and height above the surface of the earth. (See
pages 14-16 of the PowerPoint in Sec. 4.) We will also couple the
atmospheric dynamics with a slab ocean model.

The machine learning component will be based on a reservoir computing
to take advantage of its ability for rapid training. In addition, for
purposes of scaling to large scale systems, we will employ a parallel
scheme utilizing many, relatively small reservoir computers combined
via a convolutional architecture.

For further details see Sec. 4 and the references therein.

## 2. Planned Datasets

For the training, tuning, and evaluation of the hybrid
physics-based/machine-learning-based model, we plan on using the [ERA 5
reanalysis dataset](https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5)
. ERA 5 is the latest observation-based dataset produced by the
European Centre for Medium-Range Forecasts (ECMWF). ERA5 has hourly
data from 1979 till the present and contains numerous atmospheric and
oceanic variables relevant to climate change (e.g. sea-surface
temperature, winds, and moisture).

Once the data is acquired, we will regrid the data to the SPEEDY model
grid and begin training the hybrid model.

## 3. Problems and Effects to be Investigated

See Sec. 4 for a list of problems and effects that will be addressed
(pages 17-19 of the PowerPoint in Sec. 4). To begin within the next few
months, we will concentrate on the following three things:

-   Extending your present hybrid scheme implementation (which currently
    is based on atmospheric dynamics) to self-consistently incorporate
    coupling between ocean and atmospheric dynamics.

-   Development and testing of theory and techniques for insuring hybrid
    operation that avoids "numerical instabilities".

-   Extensions of our previous work on purely machine-learning-based
    climate and tipping point prediction to hybridization of the
    machine-learning-based component with a physics-based component.

## 4. Kickoff Meeting PowerPoint Presentation

![]({{base}}/images/{{page.image-folder}}/image1.png)

![]({{base}}/images/{{page.image-folder}}/image2.png)

![]({{base}}/images/{{page.image-folder}}/image3.png)
![]({{base}}/images/{{page.image-folder}}/image4.png)

![]({{base}}/images/{{page.image-folder}}/image5.png)
![]({{base}}/images/{{page.image-folder}}/image6.png)

![]({{base}}/images/{{page.image-folder}}/image7.png)
![]({{base}}/images/{{page.image-folder}}/image8.png)
![]({{base}}/images/{{page.image-folder}}/image9.png)
![]({{base}}/images/{{page.image-folder}}/image10.png)
![]({{base}}/images/{{page.image-folder}}/image11.png)
![]({{base}}/images/{{page.image-folder}}/image12.png)
![]({{base}}/images/{{page.image-folder}}/image13.png)
![]({{base}}/images/{{page.image-folder}}/image14.png)
![]({{base}}/images/{{page.image-folder}}/image15.png)
![]({{base}}/images/{{page.image-folder}}/image16.png)
![]({{base}}/images/{{page.image-folder}}/image17.png)
![]({{base}}/images/{{page.image-folder}}/image18.png)
![]({{base}}/images/{{page.image-folder}}/image19.png)
![]({{base}}/images/{{page.image-folder}}/image20.png)
![]({{base}}/images/{{page.image-folder}}/image21.png)
