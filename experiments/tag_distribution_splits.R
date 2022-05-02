library(tidyverse)
library(xtable)

train_60 <- read_delim('../data/Kakataibo/features/train_60.txt', delim = " ",
                        col_names = c("Token", "upos", "head", "deprel"))  %>% 
  mutate(Language = "Kakataibo", Set = "Training")

test_20 <- read_delim('../data/Kakataibo/features/test_20.txt', delim = " ",
                       col_names = c("Token", "upos", "head", "deprel"))  %>% 
  mutate(Language = "Kakataibo", Set = "Test")

dev_20 <- read_delim('../data/Kakataibo/features/dev_20.txt', delim = " ",
                      col_names = c("Token", "upos", "head", "deprel"))  %>% 
  mutate(Language = "Kakataibo", Set = "Dev")

full_upos <- rbind(train_60, test_20, dev_20) %>% 
  group_by(Set, upos) %>% count() %>% 
  pivot_wider(names_from = "Set", values_from = "n", values_fill = 0) %>% 
  relocate(upos, Training, Dev, Test) %>% 
  arrange(upos) %>% xtable()

train_dep <- read_delim('../data/Kakataibo/features/train.txt', delim = " ",
                        col_names = c("Token", "upos", "head", "deprel"))  %>% 
  mutate(Language = "Kakataibo", Set = "Training")

test_dep <- read_delim('../data/Kakataibo/features/test.txt', delim = " ",
                       col_names = c("Token", "upos", "head", "deprel"))  %>% 
  mutate(Language = "Kakataibo", Set = "Test")

dev_dep <- read_delim('../data/Kakataibo/features/dev.txt', delim = " ",
                      col_names = c("Token", "upos", "head", "deprel"))  %>% 
  mutate(Language = "Kakataibo", Set = "Dev")

full_deprel <- rbind(train_dep, test_dep, dev_dep) %>% 
  group_by(Set, deprel) %>% count() %>% 
  pivot_wider(names_from = "Set", values_from = "n", values_fill = 0) %>% 
  relocate(deprel, Training, Dev, Test) %>% 
  arrange(deprel)

rbind(rbind(train_60, test_20, dev_20)) %>% 
  group_by(Set, deprel) %>% count() %>% 
  pivot_wider(names_from = "Set", values_from = "n", values_fill = 0) %>% 
  relocate(deprel, Training, Dev, Test) %>% 
  arrange(deprel) %>% 
  left_join(full_deprel, by = "deprel", suffix = c(" 60", " 80")) %>% 
  xtable()
