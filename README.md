# A Tool-kit Of Techniques To Make Computer Vision Models Easier To Train And Evaluate
AIExpo Africa 2022 - Poster Presentation 

[Poster at the event](https://github.com/TRex22/MakeAISpeedy/raw/main/Final%20Poster%20Design.pdf)

## About me
I am a senior software engineer at a Fin-tech called DocFox and a master’s candidate at the University of the Witwatersrand focusing on computer vision, explainable AI and self-driving cars.

My blog: https://jasonchalom.com/blog

## Thanks
- My girlfriend :)
- My master's supervisor: Dr Richard Klein
- The Mathematical Sciences Support unit at the University of the Witwatersrand.
- PRIME Lab (Located at the University of the Witwatersrand, specialising in Computer Vision)
- Rail lab (https://www.raillab.org/)

# Abstract (150)
Machine learning techniques, especially in the area of computer vision, have opened a new frontier of research and industry - however the initial cost of entry into the field is a boundary, especially in developing economies. There are many techniques and approaches to optimising these work-flows in order to make them more computationally efficient, and therefore cheaper. Such techniques include traditional software engineering, data and signals processing, as well as more advanced optimisations to the model structures and training pipelines to make the process more efficient. These techniques have trade-offs and are task dependant but when combined appropriately can be a powerful tool-kit to improve computability of large work-flows. This poster evaluates a selection of techniques aimed at making computer vision easier to compute in memory usage, processor load, and time. These techniques are not always appropriate, and therefore several different types of tasks are considered.

# Introduction
Artificial intelligence (AI) is a fundamental part of the next industrial revolution. There is a lot of opportunity for innovation. However the cost, and access to AI - especially in computer vision - is a limiting factor in developing economies. AI on the edge is also limited by the practicality of the AI being used and such low power and embedded devices.

Many AI projects and state of the art models ignore training and evaluation cost in favour of the best results of the model. There are diminishing returns to obtaining the absolute best performance out of a model. Whilst having better accuracy and low loss when training AI models is desirable, practical use of such models does not always require the trade-offs of obtaining the best results and an acceptable level of error for more efficient use of computer resources can be acceptable.

# Methodology
For this investigation a classification problem, determining the brand of shoe from a given image is chosen[1]. This is a fairly common use case of such AI systems. The dataset is small at 16.53 MB in total, with three categories of shoes.

## Examples from the data set[1]
![Converse Shoe](images/176.jpg)
![Adidas Shoe](images/266.jpg)
![Nike Shoe](images/271.jpg)

## Scope of experiments
Each experiment has a specific focus on a component of a simple computer vision pipeline:
1. This experiment looks at different models, their performance, time cost, and memory cost
2. Here I look at the effect of pre-processing the input image dimensional size and associated metrics of the chosen baseline model
3. The last experiment applies platform and library specific optimisations and evaluates the effect on the metrics of the chosen baseline model

## Metrics which are evaluated
1. The Accuracy of both training and test dataset splits
2. The time to complete an individual run of an experiment
3. Amount of memory is used by the AI model
4. The effective use of memory by an experimental run*

\*This was only done for experiment 3, to evaluate the effect of those optimisations on memory usage

Only a synthetic performance metric of the accuracy of the test set is used. More advanced metrics of performance should be evaluated in the future.

## Hardware used
I made use of my workstation which has 64GB of RAM, 6 core CPU with 12 threads and an Nvidia RTX 3060 with 12GB of RAM.

## Specific implementation details
Each experiment uses 100 epoch runs. Over training and over shooting is not considered. The specific training loops make use of an ADAM optimiser (with standard parameters used in the PyTorch documentation[2]). The loss function used is Cross Entropy Loss[3]. A custom data loader is used which randomises the order of the images each run. Random cropping is used, and the image pixels are automatically unscaled by the TorchVision transforms.

# Experiment 1 - A look at different models
This experiment looks at the impact of model size on the accuracy and resource use of each experimental run. The family of ResNet models as well as a simple 2-layer convolutional neural network (CNN) with two linear neural networks (NN) at the end are evaluated.

A notable outcome of this experiment is that ResNet18 performed the best of all models and the small CNN performed slightly better than random guessing whist both used considerably less memory and took considerably less time to train and evaluate.

## Results
### Time impact
The smallest graphs took the least amount of time to train and evaluate
![Experiment 1 Time Graph](images/diagrams/Model%20Training/Amount%20of%20time%20each%20model%20took%20to%20train.svg)

### Memory Impact
The smallest graphs consumed the least amount of memory
![Experiment 1 Memory Graph](images/diagrams/Model%20Training/Amount%20of%20Memory%20Used%20Per%20Model.svg)

### Accuracy of the final models for each run
ResNet18 performed the best (in terms of test accuracy) over other models
![Experiment 1 Accuracy Graph](images/diagrams/Model%20Training/Accuracy%20of%20Models%20for%20100%20Epochs.svg)

# Experiment 2 - A look at image pre-processing

## Results
### Time Impact
![Experiment 2 Time Graph](images/diagrams/Image%20Training/Amount%20of%20time%20each%20model%20took%20to%20train.svg)

### Memory Impact
![Experiment 2 Memory Graph](images/diagrams/Image%20Training/Amount%20of%20Memory%20Used%20Per%20Model.svg)

### Accuracy of the final models for each run
![Experiment 2 Accuracy Graph](images/diagrams/Image%20Training/Accuracy%20of%20Models%20for%20100%20Epochs.svg)

# Experiment 3 - The application of programmatic optimisations
## Results
### Time Impact
![Experiment 3 Time Graph](images/diagrams/Optimisations/Amount%20of%20time%20each%20model%20took%20to%20train.svg)

### Memory Impact
![Experiment 3 Memory Graph](images/diagrams/Optimisations/Amount%20of%20Memory%20Used%20Per%20Model.svg)

### Accuracy of the final models for each run
![Experiment 3 Accuracy Graph](images/diagrams/Optimisations/Accuracy%20of%20Models%20for%20100%20Epochs.svg)

# Conclusion

# Limitations of the work done

# Future work

# References
- [1] Nike, Adidas and Converse Shoes Images, https://www.kaggle.com/datasets/die9origephit/nike-adidas-and-converse-imaged, Accessed: 18 September 2022
- [2] ADAM, https://pytorch.org/docs/stable/generated/torch.optim.Adam.html, Accessed: 18 September 2022
- [3] CROSSENTROPYLOSS, https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html, Accessed: 18 September 2022
