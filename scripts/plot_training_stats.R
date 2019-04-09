#!/usr/bin/env Rscript

#' script for plotting training data
#' ---------------------------------
#' takes two arguments, the path to the training data file and the output file.
#' if training remotely, it can be a good idea to mount the output directory
#' with `sshfs`.

library(tidyverse)
library(ggpubr)
library(ggrepel)
library(grid)
library(gridExtra)
library(gtable)
library(zeallot)


take <- function(n, xs) {
    if (n == 0 || length(xs) == 0) {
        return(list(list(), xs))
    }
    c(rest, ys) %<-% take(n - 1, tail(xs, -1))
    return(list(c(head(xs, 1), rest), ys))
}


chunksOf <- function(n, xs) {
    if (length(xs) == 0) {
        return(list())
    }
    c(ys, zs) %<-% take(n, xs)
    zs_ <- chunksOf(n, zs)
    return(c(list(ys), zs_))
}


gracePeriod <- 600

args <- commandArgs(trailingOnly = T)
stopifnot(length(args) > 0 && length(args) <= 2)

inputFile <- args[1]
outputFile <- (
    if (length(args) == 2) {
        args[2]
    } else {
        bns <- strsplit(basename(inputFile), '\\.')[[1]]
        bn <- (
            if (length(bns) == 1) bns
            else invoke(paste, head(bns, -1), sep='.')
        )
        file.path(dirname(inputFile), sprintf('%s.pdf', bn))
    }
)

message(sprintf('input file=%s, output file=%s', inputFile, outputFile))

repeat {
    message('reading data...')
    df = (
        read.csv(inputFile, header=F)
        %>% as.data.frame()
        %>% do({
            if (ncol(.) == 4) {
                colnames(.) <- c('epoch', 'iteration', 'type', 'value')
                .['validation'] <- 0
            } else {
              colnames(.) <- c('epoch', 'iteration', 'validation', 'type', 'value')
            }
            .
        })
        %>% mutate(validation = as.factor(validation))
        %>% as_tibble()
        %>% filter(!is.nan(value))
    )

    message('making plot...')
    c(plots, legends) %<-% (
        unique(df$type)
        %>% map(function(t) {
            data <- (
                df
                %>% filter(type == t, epoch > 0.1 * dplyr::last(epoch))
                %>% group_by(epoch, validation)
                %>% summarize(
                        v=mean(value)
                      , min=quantile(value, 0.25)
                      , max=quantile(value, 0.75)
                    )
                %>% ungroup()
            )
            annotations <- (
                data
                %>% group_by(validation)
                %>% summarize(min = min(v), max = max(v))
                %>% ungroup()
                %>% gather(k, v, -validation)
            )
            plot <- (
                ggplot(data)
                + aes(
                      epoch
                    , v
                    , ymin=min
                    , ymax=max
                    , fill=validation
                    , color=validation
                    , group=validation
                  )
                + geom_point(size=0.5)
                + geom_ribbon(alpha=0.15, linetype='blank')
                + geom_smooth(linetype='solid', size=0.3, se=F)
                + geom_hline(
                      aes(yintercept=v, color=validation)
                    , annotations
                    , linetype='dashed'
                    , size=0.15
                  )
                + geom_text_repel(
                      aes(
                        , y=v
                        , label=format(v, digits=3)
                        , color=validation
                      )
                    , annotations
                    , x=min(data$epoch)
                    , hjust=0
                    , alpha=0.8
                    , fontface='bold'
                    , size=2
                    , direction='y'
                    , inherit.aes=F
                  )
                + ylab(t)
                + theme_minimal()
                + theme_bw()
                + rremove('grid')
            )
            legend <- get_legend(plot)
            return(list(plot + rremove('legend'), legend))
        })
        %>% transpose()
    )

    message('saving plot...')
    # png(outputFile, height=5, width=10, units='in', res=200)
    pdf(outputFile, height=5, width=10)
    grid.draw(
        chunksOf(ceiling(sqrt(length(plots))), plots)
        %>% map(function(xs) {
            top <- head(xs, -1) %>% map(~. + rremove('x.title'))
            do.call(gtable_rbind, c(top, tail(xs, 1)) %>% map(ggplotGrob))
        })
        %>% invoke(arrangeGrob, ., nrow = 1)
        %>% arrangeGrob(first(legends), widths=c(0.9, 0.1))
    )
    dev.off()

    message(sprintf('done. now sleeping for %ds', gracePeriod))
    Sys.sleep(gracePeriod)
}
