sort -u $1 | \
    while read name
    do
        printf $name
        printf "\t"
        grep $name $1 | wc -l
    done
