#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
# custom colors for LMN extension on housestyle 2024                 #
# Lydia de Haan                                                      #
# June 2024                                                          #
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
#### LMN colors                    ####
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
# original colorscheme
lmncolors <- c("#007CE1", '#035891', '#CCE5F9', '#EB5496', '#A477F5', '#03C78D')

# add more colors
palette_blues_1   <- colorRampPalette(colors = c("white", "#CCE5F9"))(4)
palette_blues_2   <- colorRampPalette(colors = c("white", "#007CE1"))(4)
palette_blues_3   <- colorRampPalette(colors = c("white", "#035891"))(4)
palette_pinks_1   <- colorRampPalette(colors = c("white", "#EB5496"))(4)
palette_purples_1 <- colorRampPalette(colors = c("white", "#A477F5"))(4)
palette_greens_1  <- colorRampPalette(colors = c("white", "#03C78D"))(4)
palette_oranges_1 <- colorRampPalette(colors = c("white", "#FF9900"))(4)

# join into list to obtain color hexadecimal codes and remove all whites
lmnpalette <-c(palette_blues_1[2:4],palette_blues_2[2:4],palette_blues_3[2:4],
               palette_pinks_1[2:4],palette_purples_1[2:4],palette_greens_1[2:4], 
               palette_oranges_1[2:4]) 

# reorder colors
lmnpalette <- c("#007CE1","#EB5496","#03C78D","#A477F5","#035891","#CCE5F9","#FF9900", 
                "#55A7EB","#F18DB9","#57D9B3","#C2A4F8","#578FB5","#DDEDFB","#FFBB55", 
                "#AAD3F5","#F8C6DC","#ABECD9","#E0D1FB","#ABC7DA","#EEF6FD","#FFDDAA")

# check colors
# scales::show_col(lmncolors)
# scales::show_col(lmnpalette)

# remove all other object from environment
rm(list = ls(pattern ="palette_" ))

# clean up
gc()

 

