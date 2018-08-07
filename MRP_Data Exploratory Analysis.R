#import data
require(xlsx)
reddit_data <- read.xlsx("C:\\Users\\ali\\Downloads\\reddit_posts.xlsx", sheetName = "combine_3coders_merge_and_posts")

# distribution
class <- c("c1","c2","c3","c4","c5","c6","c7","c8")

count <- c(
sum(reddit_data$c1),
sum(reddit_data$c2),
sum(reddit_data$c3),
sum(reddit_data$c4),
sum(reddit_data$c5),
sum(reddit_data$c6),
sum(reddit_data$c7),
sum(reddit_data$c8))


class_dist <- classes <- data.frame(class, count)

barplot(class_dist$count, names.arg = class_dist$class, width = 0.85, ylim = c(0,600),
              main = "class distribution", 
              ylab = "Frequency")

plot(reddit_data$taskinfo__askhistorians_type, ylim=c(0,1000), main='Post types' , ylab='Counts')

reply_cnt <- data.frame(table(reddit_data$taskinfo__askhistorians_submissions_title))
reply_cnt$Freq <- reply_cnt$Freq - 1
barplot(reply_cnt$Freq)

barplot(reply_cnt[order(reply_cnt[,2],decreasing=TRUE),][,2], ylim=c(0,150), xlab="Reply", ylab="Count", main="Reply count for comments")

# pattern
library(ggplot2)
library(lubridate)
library(scales)
reddit_data$taskinfo__askhistorians_date <- ymd_hms(reddit_data$taskinfo__askhistorians_date)
reddit_data$taskinfo__askhistorians_date <- with_tz(reddit_data$taskinfo__askhistorians_date, "America/Toronto")

ggplot(data = reddit_data , aes(x = taskinfo__askhistorians_date)) +
  geom_histogram(aes(fill = ..count..)) +
  theme(legend.position = "none") +
  xlab("Time") + ylab("Number of posts") + 
  scale_fill_gradient(low = "midnightblue", high = "aquamarine4") + ggtitle("distribution of posts over time")


ggplot(data = reddit_data, aes(x = wday(reddit_data$taskinfo__askhistorians_date, label = TRUE))) +
  stat_count(width = 0.75) +
  theme(legend.position = "none") +
  xlab("Day of the Week") + ylab("Number of posts") + 
  scale_fill_gradient(low = "midnightblue", high = "aquamarine4") + ggtitle("Comment") +
facet_wrap("taskinfo__askhistorians_type")

ggplot(data = reddit_data, aes(x = month(reddit_data$taskinfo__askhistorians_date, label = TRUE))) +
  stat_count(width = 0.75) +
  theme(legend.position = "none") +
  xlab("Month") + ylab("Number of Posts") + 
  scale_fill_gradient(low = "midnightblue", high = "aquamarine4") +
  facet_wrap("taskinfo__askhistorians_type")

reddit_data$timeonly <- as.numeric(reddit_data$taskinfo__askhistorians_date - trunc(reddit_data$taskinfo__askhistorians_date, "days"))

class(reddit_data$timeonly) <- "POSIXct"
ggplot(data = reddit_data, aes(x = timeonly)) +
  geom_histogram(aes(fill = ..count..)) +
  theme(legend.position = "none") +
  xlab("Time") + ylab("Number of Posts") + 
  scale_x_datetime(breaks = date_breaks("3 hours"), 
                   labels = date_format("%H:00")) +
  scale_fill_gradient(low = "midnightblue", high = "aquamarine4")


# content
ggplot(reddit_data, aes(factor(grepl("#", reddit_data$taskinfo__askhistorians_text)))) +
  geom_bar(fill = "midnightblue") + 
  theme(legend.position="none", axis.title.x = element_blank()) +
  ylab("Number of posts") + 
  ggtitle("Posts with Hashtags") +
  scale_x_discrete(labels=c("No hashtags", "Posts with hashtags"))

ggplot(reddit_data, aes(factor(grepl("http|www", reddit_data$taskinfo__askhistorians_text)))) +
  geom_bar(fill = "midnightblue") + 
  theme(legend.position="none", axis.title.x = element_blank()) +
  ylab("Number of posts") + 
  ggtitle("Posts with a Link") +
  scale_x_discrete(labels=c("No link", "Posts with a Link"))

reddit_data$charsinpost <- sapply(as.character(reddit_data$taskinfo__askhistorians_text), function(x) nchar(x))
ggplot(data = reddit_data, aes(x = charsinpost)) +
  geom_histogram(aes(fill = ..count..), binwidth = 8) +
  theme(legend.position = "none") +
  xlab("Characters per post") + ylab("Number of posts") + 
  scale_fill_gradient(low = "midnightblue", high = "aquamarine4")

# scrubbing the post texts
# Remove punctuation, numbers, html-links and unecessary spaces:

reddit_text <- data.frame(text = reddit_data$taskinfo__askhistorians_text)

textScrubber <- function(dataframe) {
  
  dataframe$text <-  gsub("http\\S+\\s*", "", dataframe$text)
  dataframe$text <-  gsub("\n", " ", dataframe$text)
  dataframe$text <-  gsub("https\\w+", "", dataframe$text)
  dataframe$text <-  gsub("http\\w+", "", dataframe$text)
  dataframe$text <-  gsub("-", " ", dataframe$text)
  dataframe$text <-  gsub("&amp;", " ", dataframe$text)
  dataframe$text <-  gsub("[[:punct:]]", " ", dataframe$text)
  dataframe$text <-  gsub("[[:digit:]]", " ", dataframe$text)
  dataframe$text <-  tolower(dataframe$text)
  
  return(dataframe)
}

reddit_text <- textScrubber(reddit_text)

library(tm)
tdmCreator <- function(dataframe, stemDoc = F, rmStopwords = T){
  
  tdm <- Corpus(VectorSource(dataframe$text))
  if (isTRUE(rmStopwords)) {
    tdm <- tm_map(tdm, removeWords, c(stopwords(), "well","can","one","also", "like", "much", "just", "even", "make", "use","know", "get"))
  }
  if (isTRUE(stemDoc)) {
    tdm <- tm_map(tdm, stemDocument)
  }
  tdm <- TermDocumentMatrix(tdm, control = list(wordLengths = c(3, Inf)))
  tdm <- rowSums(as.matrix(tdm))
  tdm <- sort(tdm, decreasing = T)
  df <- data.frame(term = names(tdm), freq = tdm)
  return(df)
}

reddit_text <- tdmCreator(reddit_text)

reddit_text <- reddit_text[-c(4),]

# Selects the 15 most used words.
reddit_text_15 <- reddit_text[1:15,]

library(ggplot2)
# Create bar graph with appropriate colours and use coord_flip() to help the labels look nicer.
ggplot(reddit_text_15, aes(x = reorder(term, freq), y = freq)) +
  geom_bar(stat = "identity", fill = "red") +
  xlab("Most Used") + ylab("How Often") +
  coord_flip() + theme(text=element_text(size=25,face="bold"))

#wordcloud
library(wordcloud)
set.seed(123)

wordcloud(words = reddit_text$term, freq = reddit_text$freq, min.freq = 100,
          max.words=200, random.order=FALSE, rot.per=0.35, 
          colors=brewer.pal(8, "Dark2"))

# word association & Term Network
reddit_text <- data.frame(text = reddit_data$taskinfo__askhistorians_text)

textScrubber <- function(dataframe) {
  
  dataframe$text <-  gsub("http\\S+\\s*", "", dataframe$text)
  dataframe$text <-  gsub("\n", " ", dataframe$text)
  dataframe$text <-  gsub("https\\w+", "", dataframe$text)
  dataframe$text <-  gsub("http\\w+", "", dataframe$text)
  dataframe$text <-  gsub("-", " ", dataframe$text)
  dataframe$text <-  gsub("&amp;", " ", dataframe$text)
  dataframe$text <-  gsub("[[:punct:]]", " ", dataframe$text)
  dataframe$text <-  gsub("[[:digit:]]", " ", dataframe$text)
  dataframe$text <-  tolower(dataframe$text)
  
  return(dataframe)
}

reddit_text <- textScrubber(reddit_text)

library(tm)
tdmCreator1 <- function(dataframe, stemDoc = F, rmStopwords = T){
  
  tdm <- Corpus(VectorSource(dataframe$text))
  if (isTRUE(rmStopwords)) {
    tdm <- tm_map(tdm, removeWords, c(stopwords(), "well","can","one","also", "like", "much", "just", "even", "make", "use","know", "get"))
  }
  if (isTRUE(stemDoc)) {
    tdm <- tm_map(tdm, stemDocument)
  }
  tdm <- TermDocumentMatrix(tdm, control = list(wordLengths = c(3, Inf)))
  return(tdm)
}

tdm <- tdmCreator1(reddit_text)
inspect(tdm)

#find associated word with top frequent words
findAssocs(tdm, "german", 0.4)


#Network of Terms
#source("https://bioconductor.org/biocLite.R")
#biocLite("graph")
#biocLite("Rgraphviz")
#freq.terms <- findFreqTerms(tdm, lowfreq = 170)
#plot(tdm, term = freq.terms, corThreshold = 0.1, weighting = T)

# change it to a Boolean matrix
tdm[tdm>=1] <- 1
m <- as.matrix(tdm)

# transform into a term-term adjacency matrix
termmatrix <- m %*% t(m)
# inspect terms numbered 5 to 10
termmatrix[5:10,5:10]

library(igraph)
# build a graph from the above matrix
g <- graph.adjacency(termmatrix, weighted=T, mode = "undirected")
# remove loops
g <- simplify(g)
# set labels and degrees of vertices
V(g)$label <- V(g)$name
V(g)$degree <- degree(g)

# set seed to make the layout reproducible
set.seed(3952)
layout1 <- layout.fruchterman.reingold(g)
plot(g, layout=layout1)


#Topic Modelling

dtm <- as.DocumentTermMatrix(tdm)
install.packages("topicmodels")
library(topicmodels)
rowTotals <- apply(dtm , 1, sum) #Find the sum of words in each Document
dtm.new   <- dtm[rowTotals> 0, ]           #remove all docs without words
lda <- LDA(dtm.new, k = 8)
# find 8 topics
term <- terms(lda, 7)
# first 7 terms of every topic
(term <- apply(term, MARGIN = 2, paste, collapse = ", "))

topics <- topics(lda)
# 1st topic identified for every document (post)
library(data.table)
reddit_data.2 <- reddit_data[-c(234),] 
topics <- data.frame(date=as.IDate(reddit_data.2$taskinfo__askhistorians_date), topic=topics)
ggplot(topics, aes(date, fill = term[topic])) +
  geom_density(position = "stack")
