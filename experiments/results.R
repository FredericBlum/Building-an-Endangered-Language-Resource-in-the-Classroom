library(tidyverse)
library(viridis)
library(xtable)

upos <- read_delim('results/results_pos.csv', delim = '\t') %>% filter(!is.na(shp_acc))

upos_summary <- upos %>% pivot_longer(cols = c(shp_acc:cbr_f1), 
                      names_to = "Eval", values_to = "Value") %>% 
  group_by(Model, embeddings, Eval, train, ft) %>% 
  summarize(avg = mean(Value), sd = sd(Value))


upos_summary %>% mutate(score = paste0(round(avg, 1),"±",round(sd, 1))) %>% 
  select(-avg, -sd) %>% 
  pivot_wider(names_from = Eval, values_from = score) %>% 
  xtable(caption = "Results of the POS tagging experiment")

dep <- read_delim('results/results_dep.csv', delim = '\t')

dep_summary <- dep %>% 
  pivot_longer(cols = c(UAS_cbr:LAS_shp), 
               names_to = "Eval", 
               values_to = "Value") %>% 
  filter(!is.na(Value)) %>% 
  group_by(model, train, Eval) %>% 
  summarize(avg = mean(Value), sd = sd(Value)) %>% 
  arrange(match(model, c("delex_to_lex", "delex_to_delex", "mono", "mono_full")), 
          match(train, c("ktb", "shp", "cbr")))

dep_summary %>%
  mutate(score = paste0(round(avg, 1),"±",ifelse(is.na(sd),0, round(sd, 1)))) %>% 
  select(-avg, -sd) %>% 
  pivot_wider(names_from = Eval, values_from = score) %>% 
  relocate(model, train, UAS_cbr, LAS_cbr, UAS_shp, LAS_shp) %>% 
  xtable(caption = "Results of the Dependency parsing experiment")


dep_plot <- dep_summary %>%
  filter(Eval == "UAS_cbr") %>% ungroup() %>% mutate(experiment = factor(seq(1:7))) %>% 
  ggplot(aes(x=experiment, y = avg, fill = experiment)) +
  geom_bar(stat="identity") +
  geom_errorbar(aes(x=experiment, ymin = avg-sd, ymax=avg+sd), width = 0.4, color = "black", size = 1) +
  scale_fill_viridis(discrete = TRUE, begin = 0, end = 0.8) +
  scale_y_continuous(limits = c(0, 85), breaks = seq(from = 0, to = 85, by = 10)) +
  xlab("Experiment number") + ylab("Kakataibo UAS") + 
  theme_bw(base_size = 18) + theme(legend.position="none")

ggsave("images/dep_results.png", dep_plot)

