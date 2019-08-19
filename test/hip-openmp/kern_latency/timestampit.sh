#!/bin/bash

while read -r line; do
        timestamp=`date | colrm 1 11 | colrm 9 30`
        echo "[$timestamp] $line"
done


