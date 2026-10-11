library(stringr)
library(tidyverse)
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) stop("Supply the cohort raw directory")
setwd(args[1])
count_files = dir("miRNA/",pattern = "isoforms[.]quantification[.]txt$",recursive = T)
exp = list()
for(i in 1:length(count_files)){
  exp[[i]] <- read.table(paste0("miRNA/",count_files[[i]]),sep = "\t",header = T) %>%
    dplyr::select(c(6,4)) %>%
    group_by(miRNA_region) %>%
    summarise(reads_per_million_miRNA_mapped = sum(reads_per_million_miRNA_mapped))
}
m = Reduce(function(x, y) merge(x, y, by= 'miRNA_region',all = T), exp)
m[is.na(m)]=0
exp <- column_to_rownames(m,var = "miRNA_region")
if (nrow(exp) < 4) stop("Insufficient miRNA regions")
removed_regions <- tail(rownames(exp), 3)
if (any(grepl("MIMAT", removed_regions))) stop("Historical last-three-row exclusion would remove mature miRNAs; inspect the source format")
exp = exp[-((nrow(exp)-2):nrow(exp)), , drop = FALSE]
rownames(exp) = str_remove(rownames(exp),"mature,")
library(miRBaseVersions.db)
mh <- AnnotationDbi::select(miRBaseVersions.db,
             keys = rownames(exp),
             keytype = "MIMAT",
             columns = c("ACCESSION","NAME","VERSION"))
mh = mh[mh$VERSION=="21",]
mh = mh[match(rownames(exp),mh$ACCESSION),]
rownames(exp) = mh$NAME
meta <- jsonlite::fromJSON("miRNA_metadata.json")
ID = sapply(meta$associated_entities,
            function(x){x$entity_submitter_id})
file2id = data.frame(file_name = meta$file_id,
                     ID = ID)
count_files2 = stringr::str_split(count_files,"/",simplify = T)[,1]
file2id = file2id[match(count_files2,file2id$file_name),]
colnames(exp) = file2id$ID
k <- rowMeans(exp > 0) >= 0.10
table(k)
exp <- exp[k, , drop = FALSE]
exp <- as.matrix(exp)
exp1 = data.frame(ID=rownames(exp),exp)
colnames(exp1) = gsub('[.]', '-', colnames(exp1))
write.table(exp1,'miRNA.RPM.txt', sep="\t", quote=F, row.names = F)
