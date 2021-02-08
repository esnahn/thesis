<div lang=en>
<!-- :::lang=en -->
<!-- fenced_divs don't work exactly like real divs? -->

\cleardoublepage

\snunoheaderchapter{Abstract}

\begin{center}
\snutitlelarge
\snuentitle
\end{center}

\vspace{20pt}

\begin{flushright}
\snularge
\setlength{\parskip}{0pt}

AHN Euisoon

Department of Architecture

The Graduate School

Seoul National University

\end{flushright}

\vspace{20pt}

\normalsize

This study aims to develop a methodology for analyzing the configuration of residential spaces in Korean apartment unit plans using deep learning.
The goal for the methodology is to be capable of performing quantitative analysis on the total number of Korean apartments.

Since apartments became the mainstream housing type in Korea, the explosive increase of apartments made it hard to study. As a result, the study on apartment unit plans has been fragmented by various criteria.
Spatial analysis, including spatial syntax, is a quantitative analysis methodology for the structure of an architectural space.
It provided the basis for an objective and reproducible analysis of Korean apartments.
However, even in spatial analysis methodology, manual work required for analyzing each case is the limiting factor.
Because of that, There has been no spatial analysis study on the total cases of Korean apartments since the late 1990s.
To overcome this limitation, this study attempted to develop a methodology that can inductively learn the researcher's intention from data and apply it to perform analysis on the whole of Korean apartments.
To develop such an analysis methodology, deep learning, one of the machine learning methodologies, was used in this study.
Deep learning has the advantage of being able to learn complicated concepts using a deep neural network model.

In Chapter 2, this study reviewed previous research that applied deep learning methodology to architectural spaces.
The review confirmed that CNN (convolutional neural network) models could be applied successfully to architectural space by representing it as a two-dimensional matrix.
Also, studies showed that the rules that the deep learning model learned from the data could be analyzed through LDA (latent space analysis) or deconvolutional neural networks.
However, the deep learning model learned for other architectural spaces could not be directly applied to Korean apartments.
Therefore, to apply deep learning to Korean apartments for analysis, the review suggested that it is necessary to construct a dataset for the total number of Korean apartments.

In Chapter 3, the thesis developed an analysis methodology on the configuration of residential spaces in Korean apartment unit plans using deep learning.
First, reflecting the review on spatial analysis and deep learning, the spatial configuration model was developed in the form of a two-dimensional image.
The image represents the space as a two-dimensional grid and describes the residential characteristics of each point in the depth dimension.
Next, this study selected as the deep learning methodology for analyzing the spatial configuration.
The CAM (class activation mapping) methodology visualizes the relationship between the configuration types from the deep learning model and the layout of the unit plans.
With the CAM method, it is possible to visualize the planning characteristics of each configuration type.

The limitation of CAM is that it can only be applied to the classification set in advance by the researchers.
This study extended it into the BAM (bicluster activation map) methodology that visualizes the relationship between inductively classified types and the unit plans.
BAM classifies the configuration types of residential space by pairing nodes of the latent layer inside the deep learning model with unit plans that activate it using biclustering analysis.

Then, this study designed a process to examine the analysis methodology developed in this study.
Using a deep learning model that trained on the classification of Korean apartment unit plan dataset,
the unit plans were classified based on the characteristics of the spatial configuration that differentiate the period.
Finally,
This study compared the result with findings from previous studies and examined the effectiveness of the analysis methodology.

In Chapter 4, this study constructed the Korean apartment unit plan dataset, which is necessary for analyzing the evolution of the Korean apartment unit plans and verifying the analysis methodology through the process.
The unit plans and associated data were collected from publically accessible data. The configuration of residential spaces, such as entrance, LDK (integration of living room, kitchen, and dining room), bedroom, balcony, and restroom, was extracted from the plans.
Through the process, a dataset for the total available data of Korean apartments was constructed.

Next, the study regularized the unit plans for consistent analysis.
The main facing, the position of the entrance, and the scale of the plan were normalized.
Also, the analysis area for comparison between unit plans of different sizes was set.
Associated data such as completion year and regional jurisdiction were normalized to fit as input data of the deep learning model.
In particular, reflecting the number of previous studies that set the time period a decade or a half, the completion year, from 1969 to 2019, was divided into ten periods of 5 years.
Through this process, about 50 thousand unit plans and associated data were constructed into the dataset.

Chapter 5 analyzed the changes through time in the spatial configuration of the Korean apartment unit plans applying the methodology developed in this study.
The VGG-GAP model, the model selected in Chapter 3, was modified to fit the dataset built in Chapter 4.
Then the model was trained on the classification of the unit plans by time periods.
Through this process, the spatial configuration analysis model predicted 90.51% of 50,252 plans within 5-year error.

Next, the learned representation of the spatial configuration by the periods was analyzed using the BAM methodology developed in Chapter 3.
The biclustering analysis between about 50 thousand unit plans and 512 latent representation nodes derived 16 spatial configuration types.
Each type co-clustered unit plans and latent nodes that are activated by those.
However, the correlation of the activation was high between the types of similar periods, which means that the relationship can not be interpreted as mutually exclusive.

The unit plans of each spatial configuration type were differentiated by the time period.
However, there was no significant difference in most types by region and size.
BAM visualization on the model for each type showed that
the models of different types learned various characteristics in the unit plans of Korean apartments at different time periods.

In Chapter 6, this study reconstructed the evolutionary process of Korean apartments by applying the spatial configuration type derived in Chapter 5 to the unit plans from different periods and various building types.
Then, the results were compared with previous studies to verify the analysis methodology developed in this study.
Ultimately, the study investigated the possibility of utilizing an inductive analysis methodology based on deep learning for research on architectural space.

The subject of the analysis is the unit plans
of various building types
from the 1980s to the 2010s,
including
corridor-type and staircase-type plans
from flat-type and mixed-type buildings,
different plans from tower-type buildings,
and
one-room-type (studio) urban-type housing plans.
The activation areas for each type of the period were analyzed by BAM visualization on the plans.
The results from the analysis showed that the types and their counterpart partial model  learned the configuration in the staircase-type unit plans from the 1990s.
Also,
BAM derived unique configurations of the corridor-type plans and various plans from tower-type buildings.
However, it showed the limit of ignoring the overall configuration when there are clearly distinguishable differences in the plans of the configuration type.

The evolutionary process of Korean apartments derived the analysis mostly consistent with the findings of previous studies,
including
standardization of the staircase-type planning,
domination of fewer plan types in flat-type buildings,
trickle-down of the configuration from large to small plans,
introduction of tower-type building,
maximization of balcony area,
and
standardization of the "balcony extension" (modification of the balcony as indoor space).
The results confirmed the possibility of utilizing the analysis methodology developed in this study for the research in the evolution of Korean apartments.

The unit plan analysis methodology developed in this study does not require a manual process for individual cases.
The significance of this study is that the methodology enables analysis of every unit plans of Korean apartments.
Also, by demonstrating that the deep learning model learned only by unit plans and completion year can learn configuration types of various Korean apartments, the methodology showed that it could analyze abstract criteria that are not directly related to the spatial configuration.
This study offers new possibilities by establishes the theoretical foundation for further research,
including but not limited to,
analyzing the spatial configuration of Korean apartments based on various criteria,
expansion of analysis methodology for architectural spaces other than apartments,
and
developing generative design methodology of Korean apartment unit plans.

\vspace{16pt}

**Keywords: Apartment, Unit plan, Spatial configuration, Machine learning, Deep learning, Typology**

**Student Number: 2011-30177**

<!-- ::: -->
</div>

\newpage
<!-- 페이지가 바뀌어야 다음 파일 페이지 번호 양식에 영향받지 않음 -->