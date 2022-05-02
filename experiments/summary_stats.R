library(tidyverse)
library(xtable)
library(ggdist)
library(gghalves)
library(viridis)

shp <- read_delim('../data/Shipibo/features/all_in_one.txt', delim = " ",
                        col_names = c("Token", "upos", "head", "deprel"))  %>% 
  mutate(Language = "Shipibo")

cbr <- read_delim('../data/Kakataibo/features/all_in_one.txt', delim = " ",
                        col_names = c("Token", "upos", "head", "deprel"))  %>% 
  mutate(Language = "Kakataibo")

shp_upos <- shp %>% group_by(Language, upos) %>% count() %>% 
  mutate(Freq = round(n/4680, 2))

cbr_upos <- cbr %>% group_by(Language, upos) %>% count() %>% 
  mutate(Freq = round(n/1047, 2))

full_upos <- rbind(shp_upos, cbr_upos) %>%   
  pivot_wider(names_from = Language, values_from = c(n, Freq)) %>% 
  xtable()

shp_deprel <- shp %>% group_by(Language, deprel) %>% count() %>% 
  mutate(Freq = round(n/4680, 2))

cbr_deprel <- cbr %>% group_by(Language, deprel) %>% count() %>% 
  mutate(Freq = round(n/1047, 2))

full_deprel <- rbind(shp_deprel, cbr_deprel) %>%   
  pivot_wider(names_from = Language, values_from = c(n, Freq)) %>% 
  xtable()

cbr <- read_csv('../data/Kakataibo/utterance_lengths.csv', col_names = c("id", "length"))  %>% 
  mutate(Language = "Kakataibo")
shp <- read_csv('../data/Shipibo/utterance_lengths.csv', col_names = c("id", "length"))  %>% 
  mutate(Language = "Shipibo")

utt_length <- rbind(cbr, shp)

length_pic <- utt_length %>% 
  ggplot(aes(x = Language, y = length, color = Language, fill = Language)) +
  geom_boxplot(width = .2, fill = "white", size = 1.5, outlier.shape = NA) +
  geom_half_point(side = "l", width=1, range_scale = 0.4, alpha = 0.4, size = 1.5) +
  coord_flip()+
  #geom_jitter(color="black", size=0.5, alpha=0.3, width=0.3) +
  scale_color_viridis(discrete = TRUE, begin = 0.1, end = 0.7) +
  scale_y_continuous(limits = c(0, 20), breaks = seq(from = 0, to = 30, by = 5)) +
  xlab("") + ylab("Number of tokens in utterance") + theme_bw(base_size = 14) + theme(legend.position="none")

ggsave("images/utterance_length.png", length_pic)
