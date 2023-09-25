library(tidyverse)

setwd("/Users/seibi/projects/bmi550/assignment1")

p = tibble(
    value = c(0.529, 0.589, 0.480),
    metric = c("F1","Recall","Precision")) %>% 
    ggplot(aes(metric, value, color = metric, fill =metric)) + 
    geom_bar(stat = "identity") +
    xlab("Metric") + 
    ylab("Performance estimates") +
    theme_bw()+
    scale_fill_discrete(name ="Metric") + 
    scale_color_discrete(name = "Metric") + 
    theme(
        axis.text = element_text(size = 12)
    ) 

ggsave("metric.pdf", p, unit = "in", height = 5, width = 7, dpi = 72)
