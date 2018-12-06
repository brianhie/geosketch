sort -u $1 | \
    while read name
    do
        printf $name
        printf "\t"
        cat $1 | awk -v var=$name '$1 == var' | wc -l
    done
