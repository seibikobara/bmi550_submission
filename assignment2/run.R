
# This R code was used for figure generation and descriptive analysis



library(tidyverse)
library(magrittr)
library(janitor)

setwd("/Users/seibi/projects/bmi550/assignment2/")


# training set
data = read_csv("./fallreports_2023-9-21_train.csv")


# demographics of the train data
data %>% tabyl(fog_q_class)


# duration of PD
data %>% summarise(median = median(pd_duration),
Q1 = quantile(pd_duration, 0.25), 
Q3 = quantile(pd_duration, 0.75) 
)

categorical = function(x){
    var_ = dplyr::enquo(x)

    data %>% group_by(fog_q_class, !!var_) %>%
    dplyr::summarise(n = n()) %>% 
    mutate(prop = round(n / sum(n)*100, 1)) %>% 
    rowwise() %>% 
    mutate(n_p = paste0(n," (", prop, ")")) %>% 
    dplyr::select(fog_q_class, !!var_, n_p) %>% 
    pivot_wider(names_from = fog_q_class, values_from = n_p)
}

numerical = function(x){
    var_ = dplyr::enquo(x)

    data %>% group_by(fog_q_class) %>%
    dplyr::summarise(median = quantile(!!var_, 0.5),
                Q1 = quantile(!!var_, 0.25),
                Q3 = quantile(!!var_, 0.75)) %>% 
    mutate(out = paste0(median, " [", Q1, "-", Q3, "]")) %>% 
    dplyr::select(fog_q_class, out)
}


categorical(gender)
categorical(race)
categorical(ethnicity)
categorical(education)
categorical(num_falls_6_mo)
categorical(previous_falls)
numerical(age_at_enrollment)
numerical(pd_duration)



# missing values
missing_check = function(x){
    var_ = dplyr::enquo(x)

    data %>% filter(is.na(!!var_)) %>% nrow()
}

missing_check(gender)
missing_check(race)
missing_check(ethnicity)
missing_check(education)
missing_check(num_falls_6_mo)
missing_check(previous_falls)
missing_check(age_at_enrollment)
missing_check(pd_duration)

missing_check(fall_description) # 3






# summarize results

# results of CV
data = read_csv("best_model.csv")
data %>% names()
data %>% 
    dplyr::select(classifier, params, feature, F1_micro_in_cv_test, F1_macro_in_cv_test, accuracy_in_cv_test) %>% 
    arrange(desc(F1_micro_in_cv_test)) %>% 
    write_csv("CV_model_performance_result.csv")


# training sample vs performance
data = read_csv("sample_size.csv")
p = data %>% 
    group_by(size) %>% 
    summarise(f1_average = mean(f1)) %>% 
    ggplot(aes(size, f1_average)) + 
    geom_point(size= 0.7) +
    geom_smooth()+
    theme_bw() +
    xlab("Training set size") + 
    ylab("F1 score with micro average")+
    theme(
        text = element_text(size = 13)
    )
ggsave("sample_performance.pdf", p , width = 4, height = 4, unit = "in", dpi = 300)


# ablation study
data = read_csv("ablation.csv")
p = data %>% 
    group_by(removed) %>% 
    summarize_all(mean) %>% 
    mutate(re_f = factor(removed, 
        levels = c("ngram","cluster","word2vec","demographics","fall_location"),
        labels = c("N-gram","Word cluster", "word2vec","Demographics","Fall location"))) %>% 
    dplyr::select(re_f, f1_macro, f1_micro, accuracy) %>% 
    pivot_longer(!re_f, names_to = "metric", values_to = "value") %>% 
    ggplot(aes(re_f, value, color = metric)) + 
        geom_point(aes(group = metric)) + 
        geom_line(aes(group = metric)) + 
        theme_bw() +
        scale_color_discrete(name = "Metrics", labels = c("Accuracy","F1 score (macro)", "F1 score (micro)")) + 
        xlab("Removed feature set") + 
        ylab("Model performance")+
        theme(
            text = element_text(size = 6)
        )
ggsave("ablation.pdf", p , width = 4, height = 4, unit = "in", dpi = 300)
